#include "easypap.h"

#include <omp.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <unistd.h>

typedef unsigned long TYPE;

static TYPE *TABLE = NULL;

static volatile int changement;

static TYPE max_grains;

static inline TYPE *table_cell (TYPE *restrict i, int y, int x)
{
  return i + y * DIM + x;
}

#define table(y, x) (*table_cell (TABLE, (y), (x)))

#define RGB(r, g, b) rgba (r, g, b, 0xFF)

void sable_init ()
{
  if (TABLE == NULL) {
    const unsigned size = DIM * DIM * sizeof (TYPE);

    PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes\n", size);

    TABLE = mmap (NULL, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }
}

void sable_finalize ()
{
  const unsigned size = DIM * DIM * sizeof (TYPE);

  munmap (TABLE, size);
}

///////////////////////////// Production d'une image
void sable_refresh_img ()
{
  unsigned long int max = 0;
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++) {
      int g = table (i, j);
      int r, v, b;
      r = v = b = 0;
      if (g == 1)
        v = 255;
      else if (g == 2)
        b = 255;
      else if (g == 3)
        r = 255;
      else if (g == 4)
        r = v = b = 255;
      else if (g > 4)
        r = b = 255 - (240 * ((double)g) / (double)max_grains);

      cur_img (i, j) = RGB (r, v, b);
      if (g > max)
        max = g;
    }
  max_grains = max;
}

///////////////////////////// Version séquentielle simple (seq)

static inline int compute_new_state (int y, int x)
{
  if (table (y, x) >= 4) {
    unsigned long int div4 = table (y, x) / 4;
    table (y, x - 1) += div4;
    table (y, x + 1) += div4;
    table (y - 1, x) += div4;
    table (y + 1, x) += div4;
    table (y, x) %= 4;
    return 1;
  }
  return 0;
}

static int do_tile (int x, int y, int width, int height, int who)
{
  int chgt = 0;
  PRINT_DEBUG ('c', "tuile [%d-%d][%d-%d] traitée\n", x, x + width - 1, y,
               y + height - 1);

  monitoring_start_tile (who);

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) {
      chgt |= compute_new_state (i, j);
    }

  monitoring_end_tile (x, y, width, height, who);
  return chgt;
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned sable_compute_seq (unsigned nb_iter)
{

  for (unsigned it = 1; it <= nb_iter; it++) {
    changement = 0;
    // On traite toute l'image en un coup (oui, c'est une grosse tuile)
    changement |= do_tile (1, 1, DIM - 2, DIM - 2, 0);
    if (changement == 0)
      return it;
  }
  return 0;
}

///////////////////////////// Version séquentielle tuilée (tiled)

unsigned sable_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
    changement = 0;

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        changement |= do_tile (x + (x == 0), y + (y == 0),
                               TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                               TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                               0 /* CPU id */);
    if (changement == 0)
      return it;
  }

  return 0;
}



///////////////////////////// Configurations initiales

static void sable_draw_4partout (void);

void sable_draw (char *param)
{
  // Call function ${kernel}_draw_${param}, or default function (second
  // parameter) if symbol not found
  hooks_draw_helper (param, sable_draw_4partout);
}

void sable_draw_4partout (void)
{
  max_grains = 8;
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
      cur_img (i, j) = table (i, j) = 4;
}

void sable_draw_DIM (void)
{
  max_grains = DIM;
  for (int i = DIM / 4; i < DIM - 1; i += DIM / 4)
    for (int j = DIM / 4; j < DIM - 1; j += DIM / 4)
      cur_img (i, j) = table (i, j) = i * j / 4;
}

void sable_draw_alea (void)
{
  max_grains = 5000;
  for (int i = 0; i< DIM>>3; i++) {
    int i = 1 + random () % (DIM - 2);
    int j = 1 + random () % (DIM - 2);
    int grains = 1000 + (random () % (4000));
    cur_img (i, j) = table (i, j) = grains;
  }
}
