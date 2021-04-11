#include "kernel/ocl/common.cl"


__kernel void sample_ocl (__global unsigned *img)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  unsigned color = 0xFFFF00FF; // opacity

  img [y * DIM + x] = color;
}
