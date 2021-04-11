
#include "easypap.h"

#include <omp.h>

// If defined, the initialization hook function is called quite early in the
// initialization process, after the size (DIM variable) of images is known.
// This function can typically spawn a team of threads, or allocated additionnal
// OpenCL buffers.
// A function named <kernel>_init_<variant> is search first. If not found, a
// function <kernel>_init is searched in turn.
void <template>_init (void)
{
  PRINT_DEBUG ('u', "Image size is %dx%d\n", DIM, DIM);
  PRINT_DEBUG ('u', "Tile size is %dx%d\n", TILE_W, TILE_H);
  PRINT_DEBUG ('u', "Press <SPACE> to pause/unpause, <ESC> to quit.\n");
}

// The image is a two-dimension array of size of DIM x DIM. Each pixel is of
// type 'unsigned' and store the color information following a RGBA layout (4
// bytes). Pixel at line 'l' and column 'c' in the current image can be accessed
// using cur_img (l, c).

static unsigned compute_color (int i, int j)
{
  return cur_img (i, j);
}

// The kernel returns 0, or the iteration step at which computation has
// completed (e.g. stabilized).

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline(s):
// ./run --size 1024 --kernel <template> --variant seq
// or
// ./run -s 1024 -k <template> -v seq
//
unsigned <template>_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        next_img (i, j) = compute_color (i, j);

    swap_images ();
  }

  return 0;
}


///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline(s):
// ./run -k <template> -v tiled -g 16 -m
// or
// ./run -k <template> -v tiled -ts 64 -m
//
static void do_tile (int x, int y, int width, int height, int who)
{
  // Calling monitoring_{start|end}_tile before/after actual computation allows
  // to monitor the execution in real time (--monitoring) and/or to generate an
  // execution trace (--trace).
  // monitoring_start_tile only needs the cpu number
  monitoring_start_tile (who);

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      next_img (i, j) = compute_color (i, j);

  // In addition to the cpu number, monitoring_end_tile also needs the tile
  // coordinates
  monitoring_end_tile (x, y, width, height, who);
}

unsigned <template>_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H, 0 /* CPU id */);

    swap_images ();
  }

  return 0;
}
