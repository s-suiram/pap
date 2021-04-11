#ifndef IMG_DATA_IS_DEF
#define IMG_DATA_IS_DEF

#include "global.h"

#include <stdint.h>

extern uint32_t *restrict image, *restrict alt_image;

static inline uint32_t *img_cell (uint32_t *restrict i, int l, int c)
{
  return i + l * DIM + c;
}

#define cur_img(y, x) (*img_cell (image, (y), (x)))
#define next_img(y, x) (*img_cell (alt_image, (y), (x)))

static inline void swap_images (void)
{
  uint32_t *tmp = image;

  image     = alt_image;
  alt_image = tmp;
}

void img_data_alloc (void);
void img_data_free (void);
void img_data_replicate (void);

// Useful color functions

static inline int extract_red (uint32_t c)
{
  return c >> 24;
}

static inline int extract_green (uint32_t c)
{
  return (c >> 16) & 255;
}

static inline int extract_blue (uint32_t c)
{
  return (c >> 8) & 255;
}

static inline int extract_alpha (uint32_t c)
{
  return c & 255;
}

static inline uint32_t rgba (int r, int g, int b, int a)
{
  return (r << 24) | (g << 16) | (b << 8) | a;
}

unsigned hsv_to_rgb (float h, float s, float v);
unsigned heat_to_rgb (float h); // 0.0 = cold, 1.0 = hot
unsigned heat_to_3gauss_rgb (double v); // 0.0 = cold, 1.0 = hot

#endif
