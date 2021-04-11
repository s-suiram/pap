
#include "easypap.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline(s):
// ./run -l images/1024.png -k scrollup -v seq
//
unsigned scrollup_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int i = 0; i < DIM; i++) {
      int src = (i < DIM - 1) ? i + 1 : 0;
      for (int j = 0; j < DIM; j++)
        next_img (i, j) = cur_img (src, j);
    }

    swap_images ();
  }

  return 0;
}

// Tile inner computation
static void do_tile_reg (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++) {
    int src = (i < DIM - 1) ? i + 1 : 0;

    for (int j = x; j < x + width; j++)
      next_img (i, j) = cur_img (src, j);
  }
}

static void do_tile (int x, int y, int width, int height, int who)
{
  monitoring_start_tile (who);

  do_tile_reg (x, y, width, height);

  monitoring_end_tile (x, y, width, height, who);
}

///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline(s):
// ./run -l images/1024.png -k scrollup -v tiled
//
unsigned scrollup_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H, 0 /* CPU id */);

    swap_images ();

  }

  return 0;
}


//////////// OpenCL version using mask (ocl_ouf)
// Suggested cmdlines:
// ./run -l images/shibuya.png -k scrollup -o -v ocl_ouf
// or
// ./run -l images/1024.png -k scrollup -o -v ocl_ouf -a data/misc/mask.bin
//

static cl_mem twin_buffer = 0, mask_buffer = 0;
static char *mask_file = NULL;

void scrollup_config_ocl_ouf (char *param)
{
  if (param && (access (param, R_OK) != -1)) {
    // The parameter is a filename, so we guess it's a 1024x1024 mask
    if (DIM != 1024)
      exit_with_error ("scrollup-OpenCL-de-Ouf requires 1024x1024 images");
    mask_file = param;
  }
}

void scrollup_init_ocl_ouf (void)
{
  const int size = DIM * DIM * sizeof (unsigned);

  mask_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, size, NULL, NULL);
  if (!mask_buffer)
    exit_with_error ("Failed to allocate mask buffer");

  twin_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, size, NULL, NULL);
  if (!twin_buffer)
    exit_with_error ("Failed to allocate second input buffer");
}

void scrollup_draw_ocl_ouf (char *param)
{
  const int size = DIM * DIM * sizeof (unsigned);
  cl_int err;

  if (easypap_image_file == NULL)
    exit_with_error ("scrollup is prettier when applied to an image!");

  unsigned *tmp = malloc (size);
  if (mask_file) {
    int fd = open (mask_file, O_RDONLY);
    if (fd == -1)
      exit_with_error ("Cannot open file %s", mask_file);

    int n = read (fd, tmp, size);
    if (n != size)
      exit_with_error ("Cannot read from file %s", mask_file);
    close (fd);
  } else {
    // Draw a quick-n-dirty circle
    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++) {
        const int mid = DIM / 2;
        int dist2     = (i - mid) * (i - mid) + (j - mid) * (j - mid);
        const int r1  = (DIM / 4) * (DIM / 4);
        const int r2  = (DIM / 2) * (DIM / 2);
        if (dist2 < r1)
          tmp[i * DIM + j] = 0xFF;
        else if (dist2 < r2)
          tmp[i * DIM + j] = ((r2 - dist2) * 255 / (r2 - r1)) & 0xFF;
        else
          tmp[i * DIM + j] = 0;
      }
  }
  // We send the mask buffer to GPU
  // (not need to send twin_buffer : its content will be erased during 1st
  // iteration)
  err = clEnqueueWriteBuffer (queue, mask_buffer, CL_TRUE, 0, size, tmp, 0,
                              NULL, NULL);
  check (err, "Failed to write to extra buffer");

  free (tmp);

  img_data_replicate (); // Perform next_img = cur_img
}

unsigned scrollup_invoke_ocl_ouf (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y}; // global domain size for our calculation
  size_t local[2]  = {GPU_TILE_W, GPU_TILE_H}; // local domain size for our calculation
  cl_int err;

  monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned color = 0xFF0000FF;
    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &next_buffer);
    err |= clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &twin_buffer);
    err |= clSetKernelArg (compute_kernel, 2, sizeof (cl_mem), &cur_buffer);
    err |= clSetKernelArg (compute_kernel, 3, sizeof (cl_mem), &mask_buffer);
    err |= clSetKernelArg (compute_kernel, 4, sizeof (unsigned), &color);
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");

    // Swap buffers
    {
      cl_mem tmp  = twin_buffer;
      twin_buffer = next_buffer;
      next_buffer = tmp;
    }
  }

  clFinish (queue);

  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  return 0;
}
