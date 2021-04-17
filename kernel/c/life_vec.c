
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

typedef uint8_t cell_t;

static cell_t *restrict _table = NULL, *restrict _alternate_table = NULL;

#define CELL_PER_VEC 32
unsigned NB_VEC_PER_LINE_SEQ = 0;
unsigned NOT_JUST_SEQ = 0;

static inline cell_t *table_cell(cell_t *restrict i, int y, int x) {
  return i + y * DIM + x;
}

// This kernel does not directly work on cur_img/next_img.
// Instead, we use 2D arrays of boolean values, not colors
#define cur_table(y, x) (*table_cell (_table, (y), (x)))
#define next_table(y, x) (*table_cell (_alternate_table, (y), (x)))

// Tiles state array and enumeration
enum TileState { AWAKE = 0, ASLEEP = -1, INSOMNIA = 1 };
enum TileState *active_tiles = 0;

void display_vec(__m256i v) {
  char tmp[32];
  
  _mm256_storeu_si256((__m256i *) tmp, v);
  
  for (int i = 0; i < 32; i++)
    printf("%X, ", tmp[i]);
  printf("\n");
  fflush(stdout);
}

void life_vec_init(void) {
  // life_vec_init may be (indirectly) called several times so we check if data were
  // already allocated
  if (_table == NULL) {
    const unsigned size = DIM * DIM * sizeof(cell_t) + CELL_PER_VEC
        * sizeof(cell_t); // To prevent segfault at the bottom right tile when loading 32 bottom right neighbors in a vector
    
    PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes\n", size);
    
    _table = mmap(NULL, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    _alternate_table = mmap(NULL, size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    NB_VEC_PER_LINE_SEQ = (DIM - 2) / CELL_PER_VEC;
    NOT_JUST_SEQ = NB_VEC_PER_LINE_SEQ * CELL_PER_VEC != DIM - 2;
    
    if (active_tiles == 0) {
      active_tiles = calloc(NB_TILES_X * NB_TILES_Y, sizeof(enum TileState));
      // All tiles start awake
      memset(active_tiles, AWAKE,
             NB_TILES_X * NB_TILES_Y * sizeof(enum TileState));
    }
  }
}

void life_vec_finalize(void) {
  const unsigned size = DIM * DIM * sizeof(cell_t);
  
  munmap(_table, size);
  munmap(_alternate_table, size);
  free(active_tiles);
}

// This function is called whenever the graphical window needs to be refreshed
void life_vec_refresh_img(void) {
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++)
      cur_img (i, j) = cur_table (i, j) * color;
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

unsigned life_vec_compute_seq(unsigned nb_iter) {
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
////
// Sequential Vectorization
////

static unsigned compute_new_state_vec(int x, int y, int size) {
  unsigned change = 0;
  
  char mask_arr[CELL_PER_VEC] = {0};
  char anti_mask_arr[CELL_PER_VEC] = {0};
  memset(mask_arr, 0xFF, size);
  memset(anti_mask_arr + size, 0xFF, CELL_PER_VEC - size);
  
  __m256i n = _mm256_setzero_si256();
  __m256i mask = _mm256_lddqu_si256((const __m256i *) mask_arr);
  __m256i anti_mask = _mm256_lddqu_si256((const __m256i *) anti_mask_arr);
  
  for (int dy = y - 1; dy <= y + 1; dy++) {
    for (int dx = x - 1; dx <= x + 1; dx++) {
      __m256i n_line = _mm256_lddqu_si256((const __m256i *) &cur_table(dy, dx));
      n = _mm256_add_epi8(n, n_line);
    }
  }
  
  __m256i cells = _mm256_lddqu_si256((const __m256i *) &cur_table(y, x));
  __m256i three = _mm256_set1_epi8(3);
  __m256i ones = _mm256_set1_epi8(1);
  
  __m256i three_plus_cell_val = _mm256_add_epi8(cells, three);
  __m256i or_first_part = _mm256_cmpeq_epi8(n, three_plus_cell_val); // (n == 3 + me)
  __m256i or_sec_part = _mm256_cmpeq_epi8(n, three); // (n == 3)
  n = _mm256_or_si256(or_first_part, or_sec_part); // n = (n == 3 + me) | (n == 3)
  // A ce moment la, une cellule qui valide la condition du dessus vaut 255, grace a cette ligne on transforme les 255 en 1
  n = _mm256_and_si256(n, ones);
  
  __m256i n_eq_cells = _mm256_cmpeq_epi8(cells, n);
  change = UINT32_MAX - _mm256_movemask_epi8(n_eq_cells);
  
  __m256i next_cells = _mm256_lddqu_si256((const __m256i *) &next_table(y, x));
  next_cells = _mm256_and_si256(next_cells, anti_mask);
  
  n = _mm256_and_si256(n, mask);
  n = _mm256_add_epi8(next_cells, n);
  
  __m256i gt_one = _mm256_cmpgt_epi8(n, ones);
  __m256i eq_one = _mm256_cmpeq_epi8(n, ones);
  __m256i alive = _mm256_or_si256(gt_one, eq_one);
  
  n = _mm256_and_si256(alive, ones);
  
  _mm256_storeu_si256((__m256i_u *) &next_table(y, x), n);
  
  return change;
}

unsigned life_vec_compute_vec_seq(unsigned nb_iter) {
  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;
    
    monitoring_start_tile(0);
    
    for (int y = 1; y < DIM - 1; y++) {
      int remaining = DIM - 2;
      int x = 1;
      do {
        int advance = min(CELL_PER_VEC, remaining);
        change |= compute_new_state_vec(x, y, advance);
        x += advance;
        remaining -= advance;
      } while (remaining);
    }
    
    monitoring_end_tile(0, 0, DIM, DIM, 0);
    
    swap_tables();
    
    if (!change)
      return it;
  }
  
  return 0;
}

///////////////////////////// Tiled vec version

static int do_tile_reg(int x, int y, int width, int height) {
  int change = 0;
  
  for (int dy = y; dy < y + height; dy++) {
    int remaining = width;
    int dx = x;
    do {
      int advance = min(CELL_PER_VEC, remaining);
      change |= compute_new_state_vec(dx, dy, advance);
      dx += advance;
      remaining -= advance;
    } while (remaining);
  }
  
  return change;
}

static int do_tile(int x, int y, int width, int height, int who) {
  int r;
  
  monitoring_start_tile(who);
  
  width -= (x == 0);
  height -= (y == 0);
  
  width -= x == DIM - TILE_W;
  height -= y == DIM - TILE_H;
  
  x += x == 0;
  y += y == 0;
  
  r = do_tile_reg(x, y, width, height);
  
  monitoring_end_tile(x, y, width, height, who);
  
  return r;
}

unsigned life_vec_compute_vec_tiled(unsigned nb_iter) {
  unsigned res = 0;
  
  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        change |= do_tile(x, y, TILE_W, TILE_H, omp_get_thread_num());
    
    swap_tables();
    
    if (!change) { // we stop when all cells are stable
      res = it;
      break;
    }
  }
  
  return res;
}

///////////////////////////// Lazy tiled vec omp version (tiled)

void life_vec_ft_vec_omp_tiled_lazy(void) {
#pragma omp parallel for collapse(2) schedule(runtime)
  for (int y = 0; y < DIM; y += TILE_H)
    for (int x = 0; x < DIM; x += TILE_W)
      next_table(y, x) = cur_table(y, x) = 0;
}

static enum TileState get_tile_from_pixel(int x, int y) {
  return active_tiles[(y / TILE_H) * NB_TILES_X + (x / TILE_W)];
}

static enum TileState get_tile(int x, int y) {
  return active_tiles[y * NB_TILES_X + x];
}

static void set_tile(int x, int y, enum TileState t) {
  active_tiles[y * NB_TILES_X + x] = t;
}

static void set_tile_from_pixel(int x, int y, enum TileState t) {
  set_tile(x / TILE_W, y / TILE_H, t);
}

// Wake up 8 neighbours
static void wakeup_around(int x, int y) {
  int tx = x / TILE_W;
  int ty = y / TILE_H;
  if (tx == 0 && ty == 0) { // top left
    set_tile(tx + 1, ty, INSOMNIA);
    set_tile(tx, ty + 1, INSOMNIA);
    set_tile(tx + 1, ty + 1, INSOMNIA);
  } else if (tx + 1 == NB_TILES_X && ty == 0) { // top right
    set_tile(tx - 1, ty, INSOMNIA);
    set_tile(tx, ty + 1, INSOMNIA);
    set_tile(tx - 1, ty + 1, INSOMNIA);
  } else if (tx + 1 == NB_TILES_X && ty + 1 == NB_TILES_Y) { // bottom right
    set_tile(tx - 1, ty, INSOMNIA);
    set_tile(tx, ty - 1, INSOMNIA);
    set_tile(tx - 1, ty - 1, INSOMNIA);
  } else if (tx == 0 && ty + 1 == NB_TILES_Y) { // bottom left
    set_tile(tx + 1, ty, INSOMNIA);
    set_tile(tx, ty - 1, INSOMNIA);
    set_tile(tx + 1, ty - 1, INSOMNIA);
  } else if (ty == 0) { // top
    set_tile(tx - 1, ty, INSOMNIA);
    set_tile(tx + 1, ty, INSOMNIA);
    set_tile(tx, ty + 1, INSOMNIA);
    set_tile(tx + 1, ty + 1, INSOMNIA);
    set_tile(tx - 1, ty + 1, INSOMNIA);
  } else if (tx + 1 == NB_TILES_X) { // right
    set_tile(tx - 1, ty, INSOMNIA);
    set_tile(tx, ty + 1, INSOMNIA);
    set_tile(tx, ty - 1, INSOMNIA);
    set_tile(tx - 1, ty - 1, INSOMNIA);
    set_tile(tx - 1, ty + 1, INSOMNIA);
  } else if (ty + 1 == NB_TILES_Y) { // bottom
    set_tile(tx - 1, ty, INSOMNIA);
    set_tile(tx + 1, ty, INSOMNIA);
    set_tile(tx, ty - 1, INSOMNIA);
    set_tile(tx + 1, ty - 1, INSOMNIA);
    set_tile(tx - 1, ty - 1, INSOMNIA);
  } else if (tx == 0) { // left
    set_tile(tx + 1, ty, INSOMNIA);
    set_tile(tx, ty + 1, INSOMNIA);
    set_tile(tx + 1, ty - 1, INSOMNIA);
    set_tile(tx + 1, ty + 1, INSOMNIA);
  } else { // inner
    set_tile(tx + 1, ty, INSOMNIA);
    set_tile(tx - 1, ty, INSOMNIA);
    set_tile(tx, ty + 1, INSOMNIA);
    set_tile(tx, ty - 1, INSOMNIA);
    set_tile(tx + 1, ty + 1, INSOMNIA);
    set_tile(tx + 1, ty - 1, INSOMNIA);
    set_tile(tx - 1, ty + 1, INSOMNIA);
    set_tile(tx - 1, ty - 1, INSOMNIA);
  }
}

static int compute_new_state_vec_omp(int x, int y) {
  int change = 0;
  
  __m256i n = _mm256_setzero_si256();
  
  for (int dy = y - 1; dy <= y + 1; dy++) {
    for (int dx = x - 1; dx <= x + 1; dx++) {
      __m256i n_line = _mm256_lddqu_si256((const __m256i *) &cur_table(dy, dx));
      n = _mm256_add_epi8(n, n_line);
    }
  }
  
  __m256i cells = _mm256_lddqu_si256((const __m256i *) &cur_table(y, x));
  __m256i three = _mm256_set1_epi8(3);
  __m256i ones = _mm256_set1_epi8(1);
  
  __m256i three_plus_cell_val = _mm256_add_epi8(cells, three);
  __m256i or_first_part = _mm256_cmpeq_epi8(n, three_plus_cell_val); // (n == 3 + me)
  __m256i or_sec_part = _mm256_cmpeq_epi8(n, three); // (n == 3)
  n = _mm256_or_si256(or_first_part, or_sec_part); // n = (n == 3 + me) | (n == 3)
  // A ce moment la, une cellule qui valide la condition du dessus vaut 255, grace a cette ligne on transforme les 255 en 1
  n = _mm256_and_si256(n, ones);
  
  __m256i n_eq_cells = _mm256_cmpeq_epi8(cells, n);
  change = UINT32_MAX - _mm256_movemask_epi8(n_eq_cells);
  
  _mm256_storeu_si256((__m256i_u *) &next_table(y, x), n);
  return change;
}

static int do_tile_reg_vec(int x, int y, int width, int height) {
  int change = 0;
  
  x += x == 0;
  y += y == 0;
  x -= x == DIM - TILE_W;
  y -= y == DIM - TILE_H;
  
  for (int dy = y; dy < y + height; dy++) {
    int remaining = width;
    int dx = x;
    do {
      int advance = min(CELL_PER_VEC, remaining);
      change |= compute_new_state_vec_omp(dx, dy);
      dx += advance;
      remaining -= advance;
    } while (remaining);
  }
  return change;
}

static int do_tile_vec(int x, int y, int width, int height, int who) {
  int r = 0;
  if (get_tile_from_pixel(x, y) >= AWAKE) {
    monitoring_start_tile(who);
    
    r = do_tile_reg_vec(x, y, width, height);
    
    enum TileState newState = r == 0 ? ASLEEP : AWAKE;
#pragma omp critical(insomnia_lock)
    {
      if (newState == AWAKE) {
        wakeup_around(x, y);
      }
      if (get_tile_from_pixel(x, y) == INSOMNIA) {
        newState = AWAKE;
      }
      set_tile_from_pixel(x, y, newState);
    }
    
    monitoring_end_tile(x, y, width, height, who);
  }
  
  return r;
}

unsigned life_vec_compute_vec_omp_tiled_lazy(unsigned nb_iter) {
  unsigned res = 0;
  
  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;

#pragma omp parallel for collapse(2) reduction(|:change) schedule(runtime)
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        change |= do_tile_vec(x, y, TILE_W, TILE_H, omp_get_thread_num());
    
    swap_tables();
    
    if (!change) {
      res = it;
      break;
    }
  }
  
  return res;
}

///////////////////////////// Initial configs

void life_vec_draw_guns(void);

static inline void set_cell(int y, int x) {
  cur_table (y, x) = 1;
  if (opencl_used)
    cur_img (y, x) = 1;
}

static inline int get_cell(int y, int x) {
  return cur_table (y, x);
}

static void inline life_vec_rle_parse(char *filename, int x, int y,
                                      int orientation) {
  rle_lexer_parse(filename, x, y, set_cell, orientation);
}

static void inline life_vec_rle_generate(char *filename, int x, int y, int width,
                                         int height) {
  rle_generate(x, y, width, height, get_cell, filename);
}

void life_vec_draw(char *param) {
  if (param && (access(param, R_OK) != -1)) {
    // The parameter is a filename, so we guess it's a RLE-encoded file
    life_vec_rle_parse(param, 1, 1, RLE_ORIENTATION_NORMAL);
  } else
    // Call function ${kernel}_draw_${param}, or default function (second
    // parameter) if symbol not found
    hooks_draw_helper(param, life_vec_draw_guns);
}

static void otca_autoswitch(char *name, int x, int y) {
  life_vec_rle_parse(name, x, y, RLE_ORIENTATION_NORMAL);
  life_vec_rle_parse("data/rle/autoswitch-ctrl.rle", x + 123, y + 1396,
                     RLE_ORIENTATION_NORMAL);
}

static void otca_life_vec(char *name, int x, int y) {
  life_vec_rle_parse(name, x, y, RLE_ORIENTATION_NORMAL);
  life_vec_rle_parse("data/rle/b3-s23-ctrl.rle", x + 123, y + 1396,
                     RLE_ORIENTATION_NORMAL);
}

static void at_the_four_corners(char *filename, int distance) {
  life_vec_rle_parse(filename, distance, distance, RLE_ORIENTATION_NORMAL);
  life_vec_rle_parse(filename, distance, distance, RLE_ORIENTATION_HINVERT);
  life_vec_rle_parse(filename, distance, distance, RLE_ORIENTATION_VINVERT);
  life_vec_rle_parse(filename, distance, distance,
                     RLE_ORIENTATION_HINVERT | RLE_ORIENTATION_VINVERT);
}

// Suggested cmdline: ./run -k life_vec -s 2176 -a otca_off -ts 64 -r 10 -si
void life_vec_draw_otca_off(void) {
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);
  
  otca_autoswitch("data/rle/otca-off.rle", 1, 1);
}

// Suggested cmdline: ./run -k life_vec -s 2176 -a otca_on -ts 64 -r 10 -si
void life_vec_draw_otca_on(void) {
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);
  
  otca_autoswitch("data/rle/otca-on.rle", 1, 1);
}

// Suggested cmdline: ./run -k life_vec -s 6208 -a meta3x3 -ts 64 -r 50 -si
void life_vec_draw_meta3x3(void) {
  if (DIM < 6208)
    exit_with_error ("DIM should be at least %d", 6208);
  
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      otca_life_vec(j == 1 ? "data/rle/otca-on.rle" : "data/rle/otca-off.rle",
                    1 + j * (2058 - 10), 1 + i * (2058 - 10));
}

// Suggested cmdline: ./run -k life_vec -a bugs -ts 64
void life_vec_draw_bugs(void) {
  for (int y = 16; y < DIM / 2; y += 32) {
    life_vec_rle_parse("data/rle/tagalong.rle", y + 1, y + 8,
                       RLE_ORIENTATION_NORMAL);
    life_vec_rle_parse("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                       RLE_ORIENTATION_NORMAL);
  }
}

// Suggested cmdline: ./run -k life_vec -v omp -a ship -s 512 -m -ts 16
void life_vec_draw_ship(void) {
  for (int y = 16; y < DIM / 2; y += 32) {
    life_vec_rle_parse("data/rle/tagalong.rle", y + 1, y + 8,
                       RLE_ORIENTATION_NORMAL);
    life_vec_rle_parse("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                       RLE_ORIENTATION_NORMAL);
  }
  
  for (int y = 43; y < DIM - 134; y += 148) {
    life_vec_rle_parse("data/rle/greyship.rle", DIM - 100, y,
                       RLE_ORIENTATION_NORMAL);
  }
}

void life_vec_draw_stable(void) {
  for (int i = 1; i < DIM - 2; i += 4)
    for (int j = 1; j < DIM - 2; j += 4) {
      set_cell(i, j);
      set_cell(i, j + 1);
      set_cell(i + 1, j);
      set_cell(i + 1, j + 1);
    }
}

void life_vec_draw_guns(void) {
  at_the_four_corners("data/rle/gun.rle", 1);
}

void life_vec_draw_random(void) {
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
      if (random() & 1)
        set_cell(i, j);
}

// Suggested cmdline: ./run -k life_vec -a clown -s 256 -i 110
void life_vec_draw_clown(void) {
  life_vec_rle_parse("data/rle/clown-seed.rle", DIM / 2, DIM / 2,
                     RLE_ORIENTATION_NORMAL);
}

void life_vec_draw_diehard(void) {
  life_vec_rle_parse("data/rle/diehard.rle", DIM / 2, DIM / 2,
                     RLE_ORIENTATION_NORMAL);
}
