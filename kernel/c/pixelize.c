
#include "easypap.h"

#include <omp.h>

static unsigned PIX_BLOC = 16;
static unsigned LOG_BLOC = 4; // LOG2(PIX_BLOC)
static unsigned LOG_BLOCx2 = 8; // LOG2(PIX_BLOC)^2 : pour diviser par PIX_BLOC^2 en faisant un shift right

static unsigned log2_of_power_of_2 (unsigned v)
{
  const unsigned b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
  const unsigned S[] = {1, 2, 4, 8, 16};

  unsigned r = 0;

  for (int i = 4; i >= 0; i--)
    if (v & b[i]) {
      v >>= S[i];
      r |= S[i];
    }

  return r;
}

// Tile inner computation
static void do_tile_reg (int x, int y, int width, int height)
{
  unsigned r = 0, g = 0, b = 0, a = 0, mean;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) {
      unsigned c = cur_img (i, j);
      r += c >> 24 & 255;
      g += c >> 16 & 255;
      b += c >> 8 & 255;
      a += c & 255;
    }

  r >>= LOG_BLOCx2;
  g >>= LOG_BLOCx2;
  b >>= LOG_BLOCx2;
  a >>= LOG_BLOCx2;

  mean = (r << 24) | (g << 16) | (b << 8) | a;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      cur_img (i, j) = mean;
}

static void do_tile (int x, int y, int width, int height, int who)
{
  monitoring_start_tile (who);

  do_tile_reg (x, y, width, height);

  monitoring_end_tile (x, y, width, height, who);
}

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline:
// ./run -l images/1024.png -k pixelize -a 16
//
unsigned pixelize_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += PIX_BLOC)
      for (int x = 0; x < DIM; x += PIX_BLOC)
        do_tile_reg (x, y, PIX_BLOC, PIX_BLOC);
  }

  return 0;
}


// The parameter is used to fix the size of pixelized blocks
void pixelize_config (char *param)
{
  unsigned n;

  if (param != NULL) {

    n = atoi (param);
    if (n > 0) {
      PIX_BLOC = n;
      if (PIX_BLOC & (PIX_BLOC - 1))
        exit_with_error ("PIX_BLOC is not a power of two");

      LOG_BLOC   = log2_of_power_of_2 (PIX_BLOC);
      LOG_BLOCx2 = 2 * LOG_BLOC;
    }
  }
}

///////////////////////////// OpenCL big variant (ocl_big)

unsigned pixelize_invoke_ocl (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y};
  size_t local[2]  = {GPU_TILE_W, GPU_TILE_H};
  cl_int err;

  monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {

    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");
  }

  clFinish (queue);

  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  return 0;
}

void pixelize_config_ocl (char *param)
{
  pixelize_config (param);

  if (PIX_BLOC > 16)
    exit_with_error ("PIX_BLOC too large (> 16) for OpenCL variant.");
}

void pixelize_init_ocl (void)
{
  if (GPU_TILE_W != GPU_TILE_H)
    exit_with_error ("Tiles should have a square shape (%d != %d)", GPU_TILE_W,
                     GPU_TILE_H);

  if (GPU_TILE_W < PIX_BLOC || (GPU_TILE_W % PIX_BLOC != 0))
    exit_with_error ("Tile size (%d) must be a multiple of PIX_BLOC (%d)",
                     GPU_TILE_W, PIX_BLOC);
}
