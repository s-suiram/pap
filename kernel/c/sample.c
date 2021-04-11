
#include "easypap.h"

///////////////////////////// Simple sequential version (seq)
unsigned sample_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        cur_img (i, j) = 0xFFFF00FF;

  }

  return 0;
}

//////////////////////////////////////////////////////////////////////////
///////////////////////////// OpenCL version
// Suggested cmdlines:
// ./run -k sample -o
//
unsigned sample_invoke_ocl (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y};   // global domain size for our calculation
  size_t local[2]  = {GPU_TILE_W, GPU_TILE_H}; // local domain size for our calculation
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
