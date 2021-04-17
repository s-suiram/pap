

#include "easypap.h"
#include "rle_lexer.h"

#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef ENABLE_VECTO
#include <immintrin.h>

#endif

static unsigned color = 0xFFFF00FF; // Living cells have the yellow color

typedef unsigned cell_t;

static cell_t *restrict _table = NULL, *restrict _alternate_table = NULL;

static inline cell_t *table_cell(cell_t *restrict i, int y, int x) {
  return i + y * DIM + x;
}

// This kernel does not directly work on cur_img/next_img.
// Instead, we use 2D arrays of boolean values, not colors
#define cur_table(y, x) (*table_cell (_table, (y), (x)))
#define next_table(y, x) (*table_cell (_alternate_table, (y), (x)))

void life_ocl_init(void) {
  // life_ocl_init may be (indirectly) called several times so we check if data were
  // already allocated
  if (_table == NULL) {
    const unsigned size = DIM * DIM * sizeof(cell_t);
    
    PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes\n", size);
    
    _table = mmap(NULL, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    _alternate_table = mmap(NULL, size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }
}

void life_ocl_finalize(void) {
  const unsigned size = DIM * DIM * sizeof(cell_t);
  
  munmap(_table, size);
  munmap(_alternate_table, size);
}

// This function is called whenever the graphical window needs to be refreshed
void life_ocl_refresh_img(void) {
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++)
      cur_img (i, j) = cur_table (i, j) * color;
}

void life_ocl_refresh_img_ocl(void) {
  cl_int err;
  err = clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0, sizeof(unsigned) * DIM * DIM, _table, 0, NULL, NULL);
  check (err, "Failed to read buffer from GPU");
  life_ocl_refresh_img();
}

static inline void swap_tables(void) {
  cell_t *tmp = _table;
  
  _table = _alternate_table;
  _alternate_table = tmp;
}

///////////////////////////// Sequential version (seq)

static int compute_new_state(int y, int x) {
  cell_t n = 0;
  cell_t me = cur_table (y, x) != 0;
  unsigned change = 0;
  
  for (int i = y - 1; i <= y + 1; i++)
    for (int j = x - 1; j <= x + 1; j++)
      n += cur_table (i, j);
  
  n = (n == 3 + me) | (n == 3);
  change |= n != me;
  
  next_table (y, x) = n;
  
  return change;
}

unsigned life_ocl_compute_seq(unsigned nb_iter) {
  for (unsigned it = 1; it <= nb_iter; it++) {
    int change = 0;
    
    monitoring_start_tile(0);
    
    for (int i = 1; i < DIM - 1; i++)
      for (int j = 1; j < DIM - 1; j++)
        change |= compute_new_state(i, j);
    
    monitoring_end_tile(0, 0, DIM, DIM, 0);
    
    swap_tables();
    
    if (!change)
      return it;
  }
  
  return 0;
}

cl_mem change;

void life_ocl_init_ocl() {
  life_ocl_init();
  change = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned), 0, 0);
}

unsigned life_ocl_invoke_ocl(unsigned nb_iter) {
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y};
  size_t local[2] = {GPU_TILE_W, GPU_TILE_H};
  cl_int err;
  
  monitoring_start_tile(easypap_gpu_lane(TASK_TYPE_COMPUTE));
  for (unsigned it = 1; it <= nb_iter; it++) {
    
    err = 0;
    err |= clSetKernelArg(compute_kernel, 0, sizeof(cl_mem), &cur_buffer);
    err |= clSetKernelArg(compute_kernel, 1, sizeof(cl_mem), &next_buffer);
    err |= clSetKernelArg(compute_kernel, 2, sizeof(cl_mem), &change);
    check (err, "Failed to set kernel arguments");
    
    err = clEnqueueNDRangeKernel(queue, compute_kernel, 2, NULL, global, local, 0, NULL, NULL);
    check (err, "Failed to execute kernel");
    
    unsigned changed;
    clEnqueueReadBuffer(queue, change, CL_TRUE, 0, sizeof(unsigned), &changed, 0, 0, 0);
    if (!changed)
      return it;
    
    cl_mem tmp = cur_buffer;
    cur_buffer = next_buffer;
    next_buffer = tmp;
    
  }
  
  clFinish(queue);
  
  monitoring_end_tile(0, 0, DIM, DIM, easypap_gpu_lane(TASK_TYPE_COMPUTE));
  
  return 0;
}

///////////////////////////// Initial configs

void life_ocl_draw_guns(void);

static inline void set_cell(int y, int x) {
  cur_table (y, x) = 1;
  if (opencl_used)
    cur_img (y, x) = 1;
}

static inline int get_cell(int y, int x) {
  return cur_table (y, x);
}

static void inline life_ocl_rle_parse(char *filename, int x, int y,
                                   int orientation) {
  rle_lexer_parse(filename, x, y, set_cell, orientation);
}

static void inline life_ocl_rle_generate(char *filename, int x, int y, int width,
                                      int height) {
  rle_generate(x, y, width, height, get_cell, filename);
}

void life_ocl_draw(char *param) {
  if (param && (access(param, R_OK) != -1)) {
    // The parameter is a filename, so we guess it's a RLE-encoded file
    life_ocl_rle_parse(param, 1, 1, RLE_ORIENTATION_NORMAL);
  } else
    // Call function ${kernel}_draw_${param}, or default function (second
    // parameter) if symbol not found
    hooks_draw_helper(param, life_ocl_draw_guns);
}

static void otca_autoswitch(char *name, int x, int y) {
  life_ocl_rle_parse(name, x, y, RLE_ORIENTATION_NORMAL);
  life_ocl_rle_parse("data/rle/autoswitch-ctrl.rle", x + 123, y + 1396,
                  RLE_ORIENTATION_NORMAL);
}

static void otca_life_ocl(char *name, int x, int y) {
  life_ocl_rle_parse(name, x, y, RLE_ORIENTATION_NORMAL);
  life_ocl_rle_parse("data/rle/b3-s23-ctrl.rle", x + 123, y + 1396,
                  RLE_ORIENTATION_NORMAL);
}

static void at_the_four_corners(char *filename, int distance) {
  life_ocl_rle_parse(filename, distance, distance, RLE_ORIENTATION_NORMAL);
  life_ocl_rle_parse(filename, distance, distance, RLE_ORIENTATION_HINVERT);
  life_ocl_rle_parse(filename, distance, distance, RLE_ORIENTATION_VINVERT);
  life_ocl_rle_parse(filename, distance, distance,
                  RLE_ORIENTATION_HINVERT | RLE_ORIENTATION_VINVERT);
}

// Suggested cmdline: ./run -k life_ocl -s 2176 -a otca_off -ts 64 -r 10 -si
void life_ocl_draw_otca_off(void) {
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);
  
  otca_autoswitch("data/rle/otca-off.rle", 1, 1);
}

// Suggested cmdline: ./run -k life_ocl -s 2176 -a otca_on -ts 64 -r 10 -si
void life_ocl_draw_otca_on(void) {
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);
  
  otca_autoswitch("data/rle/otca-on.rle", 1, 1);
}

// Suggested cmdline: ./run -k life_ocl -s 6208 -a meta3x3 -ts 64 -r 50 -si
void life_ocl_draw_meta3x3(void) {
  if (DIM < 6208)
    exit_with_error ("DIM should be at least %d", 6208);
  
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      otca_life_ocl(j == 1 ? "data/rle/otca-on.rle" : "data/rle/otca-off.rle",
                 1 + j * (2058 - 10), 1 + i * (2058 - 10));
}

// Suggested cmdline: ./run -k life_ocl -a bugs -ts 64
void life_ocl_draw_bugs(void) {
  for (int y = 16; y < DIM / 2; y += 32) {
    life_ocl_rle_parse("data/rle/tagalong.rle", y + 1, y + 8,
                    RLE_ORIENTATION_NORMAL);
    life_ocl_rle_parse("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                    RLE_ORIENTATION_NORMAL);
  }
}

// Suggested cmdline: ./run -k life_ocl -v omp -a ship -s 512 -m -ts 16
void life_ocl_draw_ship(void) {
  for (int y = 16; y < DIM / 2; y += 32) {
    life_ocl_rle_parse("data/rle/tagalong.rle", y + 1, y + 8,
                    RLE_ORIENTATION_NORMAL);
    life_ocl_rle_parse("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                    RLE_ORIENTATION_NORMAL);
  }
  
  for (int y = 43; y < DIM - 134; y += 148) {
    life_ocl_rle_parse("data/rle/greyship.rle", DIM - 100, y,
                    RLE_ORIENTATION_NORMAL);
  }
}

void life_ocl_draw_stable(void) {
  for (int i = 1; i < DIM - 2; i += 4)
    for (int j = 1; j < DIM - 2; j += 4) {
      set_cell(i, j);
      set_cell(i, j + 1);
      set_cell(i + 1, j);
      set_cell(i + 1, j + 1);
    }
}

void life_ocl_draw_guns(void) {
  at_the_four_corners("data/rle/gun.rle", 1);
}

void life_ocl_draw_random(void) {
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
      if (random() & 1)
        set_cell(i, j);
}

// Suggested cmdline: ./run -k life_ocl -a clown -s 256 -i 110
void life_ocl_draw_clown(void) {
  life_ocl_rle_parse("data/rle/clown-seed.rle", DIM / 2, DIM / 2,
                  RLE_ORIENTATION_NORMAL);
}

void life_ocl_draw_diehard(void) {
  life_ocl_rle_parse("data/rle/diehard.rle", DIM / 2, DIM / 2,
                  RLE_ORIENTATION_NORMAL);
}
