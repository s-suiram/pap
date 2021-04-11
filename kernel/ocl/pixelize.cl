#include "kernel/ocl/common.cl"

#ifdef PARAM
#define PIX_BLOC PARAM
#else
#define PIX_BLOC 16
#endif

// In this over-simplified kernel, all the pixels of a bloc adopt the color
// on the top-left pixel (i.e. we do not compute the average color).
__kernel void pixelize_ocl (__global unsigned *in)
{
  __local unsigned couleur [GPU_TILE_H / PIX_BLOC][GPU_TILE_W / PIX_BLOC];
  int x = get_global_id (0);
  int y = get_global_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  if (xloc % PIX_BLOC == 0 && yloc % PIX_BLOC == 0)
    couleur [yloc / PIX_BLOC][xloc / PIX_BLOC] = in [y * DIM + x];

  barrier (CLK_LOCAL_MEM_FENCE);

  in [y * DIM + x] = couleur [yloc / PIX_BLOC][xloc / PIX_BLOC];
}
