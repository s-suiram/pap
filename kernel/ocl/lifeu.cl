#include "kernel/ocl/common.cl"

// DO NOT MODIFY: this kernel updates the OpenGL texture buffer
 // This is a life-specific version (generic version is defined in common.cl)
__kernel void lifeu_update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
    int y = get_global_id (1);
    int x = get_global_id (0);

    write_imagef (tex, (int2)(x, y), color_scatter (cur [y * DIM + x] * 0xFFFF00FF));
}

__kernel void lifeu_ocl (__global unsigned *cur, __global unsigned *next, __global unsigned *change)
{
    __local unsigned changed[GPU_TILE_H][GPU_TILE_W];
    int x = get_global_id(0);
    int y = get_global_id(1);
    int lx = get_local_id (0);
    int ly = get_local_id (1);

    //changed[ly][lx] = 0;

    if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1) {
        int n = 0;
        int me = cur[y * DIM + x];

        for (int dy = y - 1; dy <= y + 1; dy++)
          for (int dx = x - 1; dx <= x + 1; dx++)
            n += cur[dy * DIM + dx];

        n = (n == 3 + me) | (n == 3);

        //if (n != me) {
        //    changed[ly][lx] |= 1;
        //}

        next[y * DIM + x] = n;
    }

    for (int d = GPU_TILE_W >> 1; d > 0; d >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (lx < d) {
        changed[ly][lx] += changed[ly][lx + d];
      }
    }


    for (int d = GPU_TILE_H >> 1; d > 0; d >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (lx == 0 && ly < d) {
        changed[ly][lx] += changed[ly + d][lx];
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    change[0] = 1;
}


//__kernel void pixelize_ocl (__global unsigned *in)
//{
//  __local int4 tile[GPU_TILE_H][GPU_TILE_W];
//  int x = get_global_id (0);
//  int y = get_global_id (1);
//  int xloc = get_local_id (0);
//  int yloc = get_local_id (1);
//
//  tile[yloc][xloc] = color_to_int4(in[y * DIM + x]);
//
//  for (int d = GPU_TILE_W >> 1; d > 0; d >>= 1) {
//    barrier(CLK_LOCAL_MEM_FENCE);
//    if (xloc < d) {
//      tile[yloc][xloc] += tile[yloc][xloc + d];
//    }
//  }
//
//
//  for (int d = GPU_TILE_H >> 1; d > 0; d >>= 1) {
//    barrier(CLK_LOCAL_MEM_FENCE);
//    if (xloc == 0 && yloc < d) {
//      tile[yloc][xloc] += tile[yloc + d][xloc];
//    }
//  }
//
//  barrier (CLK_LOCAL_MEM_FENCE);
//
//  in [y * DIM + x] = int4_to_color(tile[0][0] / (int4)(GPU_TILE_W * GPU_TILE_H));
//}
