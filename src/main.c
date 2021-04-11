#include <fcntl.h>
#include <hwloc.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/utsname.h>

#ifdef ENABLE_SDL
#include <SDL.h>
#endif

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#include "constants.h"
#include "cpustat.h"
#include "easypap.h"
#include "graphics.h"
#include "hooks.h"
#include "ocl.h"
#include "trace_record.h"

int max_iter            = 0;
unsigned refresh_rate   = -1;
unsigned do_display     = 1;
unsigned vsync          = 1;
unsigned soft_rendering = 0;

static char *progname    = NULL;
char *variant_name       = NULL;
char *kernel_name        = NULL;
char *draw_param         = NULL;
char *easypap_image_file = NULL;

static char *output_file = "./plots/data/perf_data.csv";
#define MAX_LABEL 64
static char trace_label[MAX_LABEL] = {0};

unsigned opencl_used                                       = 0;
unsigned easypap_mpirun                                    = 0;
static int _easypap_mpi_rank                               = 0;
static int _easypap_mpi_size                               = 1;
static unsigned master_do_display __attribute__ ((unused)) = 1;
static unsigned do_pause                                   = 0;
static unsigned quit_when_done                             = 0;
static unsigned nb_cores                                   = 1;
unsigned do_first_touch                                    = 0;
static unsigned do_dump __attribute__ ((unused))           = 0;
static unsigned do_thumbs __attribute__ ((unused))         = 0;
static unsigned show_ocl_config                            = 0;
static unsigned list_ocl_variants                          = 0;

static hwloc_topology_t topology;

unsigned easypap_requested_number_of_threads (void)
{
  char *str = getenv ("OMP_NUM_THREADS");

  if (str == NULL)
    return easypap_number_of_cores ();
  else
    return atoi (str);
}

unsigned easypap_gpu_lane (task_type_t task_type)
{
  return easypap_requested_number_of_threads () +
         (task_type == TASK_TYPE_COMPUTE ? 0 : 1);
}

char *easypap_omp_schedule (void)
{
  char *str = getenv ("OMP_SCHEDULE");
  return (str == NULL) ? "" : str;
}

char *easypap_omp_places (void)
{
  char *str = getenv ("OMP_PLACES");
  return (str == NULL) ? "" : str;
}

unsigned easypap_number_of_cores (void)
{
  return nb_cores;
}

int easypap_mpi_rank (void)
{
  return _easypap_mpi_rank;
}

int easypap_mpi_size (void)
{
  return _easypap_mpi_size;
}

int easypap_proc_is_master (void)
{
  // easypap_mpi_rank == 0 even if !easypap_mpirun
  return easypap_mpi_rank () == 0;
}

void easypap_check_mpi (void)
{
#ifndef ENABLE_MPI
  exit_with_error ("Program was not compiled with -DENABLE_MPI");
#else
  if (!easypap_mpirun)
    exit_with_error ("\n**************************************************\n"
                     "**** MPI variant was not launched using mpirun!\n"
                     "****     Please use --mpi <mpi_run_args>\n"
                     "**************************************************");
#endif
}

void easypap_check_vectorization (vec_type_t vec_type, direction_t dir)
{
#ifdef ENABLE_VECTO
  // Order of types must be consistent with that defined in vec_type enum (see
  // api_funcs.h)
  int bytes[] = {VEC_SIZE_CHAR, VEC_SIZE_INT, VEC_SIZE_FLOAT, VEC_SIZE_DOUBLE};
  int n       = (dir == DIR_HORIZONTAL ? TILE_W : TILE_H);

  if (n < bytes[vec_type] || n % bytes[vec_type])
    exit_with_error ("Tile %s (%d) is too small with respect to vectorization "
                     "requirements and should be a multiple of %d",
                     (dir == DIR_HORIZONTAL ? "width" : "height"), n,
                     bytes[vec_type]);

#endif
}

static void update_refresh_rate (int p)
{
  static int tab_refresh_rate[] = {1, 2, 5, 10, 100, 1000};
  static int i_refresh_rate     = 0;

  if (easypap_mpirun || (i_refresh_rate == 0 && p < 0) ||
      (i_refresh_rate == 5 && p > 0))
    return;

  i_refresh_rate += p;
  refresh_rate = tab_refresh_rate[i_refresh_rate];
  printf ("< Refresh rate set to: %d >\n", refresh_rate);
}

static void output_perf_numbers (long time_in_us, unsigned nb_iter)
{
  FILE *f = fopen (output_file, "a");
  struct utsname s;

  if (f == NULL)
    exit_with_error ("Cannot open \"%s\" file (%s)", output_file,
                     strerror (errno));

  if (ftell (f) == 0) {
    fprintf (f, "%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n", "machine", "size",
             "tilew", "tileh", "threads", "kernel", "variant", "iterations",
             "schedule", "places", "label", "arg", "time");
  }

  if (uname (&s) < 0)
    exit_with_error ("uname failed (%s)", strerror (errno));

  fprintf (f, "%s;%u;%u;%u;%u;%s;%s;%u;%s;%s;%s;%s;%ld\n", s.nodename, DIM,
           TILE_W, TILE_H, easypap_requested_number_of_threads (), kernel_name,
           variant_name, nb_iter, easypap_omp_schedule (),
           easypap_omp_places (), trace_label, (draw_param ?: "none"),
           time_in_us);

  fclose (f);
}

static void set_default_trace_label (void)
{
  if (trace_label[0] == '\0') {
    char *str = getenv ("OMP_SCHEDULE");

    if (str != NULL)
      snprintf (trace_label, MAX_LABEL,
                "%s_%s (%s) %d/%dx%d", kernel_name,
                variant_name, str, DIM, TILE_W, TILE_H);
    else
      snprintf (trace_label, MAX_LABEL, "%s_%s %d/%dx%d",
                kernel_name, variant_name, DIM, TILE_W, TILE_H);
  }
}

static void usage (int val);

static void filter_args (int *argc, char *argv[]);

static void check_tile_size (void)
{
  if (TILE_W == 0) {
    if (NB_TILES_X == 0) {
      TILE_W     = TILE_H ?: DEFAULT_CPU_TILE_SIZE;
      NB_TILES_X = DIM / TILE_W;
    } else {
      TILE_W = DIM / NB_TILES_X;
    }
  } else if (NB_TILES_X == 0) {
    NB_TILES_X = DIM / TILE_W;
  } else if (NB_TILES_X * TILE_W != DIM) {
    exit_with_error (
        "Inconsistency detected: NB_TILES_X (%d) x TILE_W (%d) != DIM (%d).",
        NB_TILES_X, TILE_W, DIM);
  }

  if (DIM % TILE_W)
    exit_with_error ("DIM (%d) is not a multiple of TILE_W (%d)!", DIM, TILE_W);

  if (DIM % NB_TILES_X)
    exit_with_error ("DIM (%d) is not a multiple of NB_TILES_X (%d)!", DIM,
                     NB_TILES_X);

  if (TILE_H == 0) {
    if (NB_TILES_Y == 0) {
      TILE_H     = TILE_W;
      NB_TILES_Y = DIM / TILE_H;
    } else {
      TILE_H = DIM / NB_TILES_Y;
    }
  } else if (NB_TILES_Y == 0) {
    NB_TILES_Y = DIM / TILE_H;
  } else if (NB_TILES_Y * TILE_H != DIM) {
    exit_with_error (
        "Inconsistency detected: NB_TILES_Y (%d) x TILE_H (%d) != DIM (%d).",
        NB_TILES_Y, TILE_H, DIM);
  }

  if (DIM % TILE_H)
    exit_with_error ("DIM (%d) is not a multiple of TILE_H (%d)!", DIM, TILE_H);

  if (DIM % NB_TILES_Y)
    exit_with_error ("Warning: DIM (%d) is not a multiple of NB_TILES_Y (%d)!",
                     DIM, NB_TILES_Y);
}

static void init_phases (void)
{
#ifdef ENABLE_MPI
  if (easypap_mpirun) {
    int required = MPI_THREAD_FUNNELED;
    int provided;

    MPI_Init_thread (NULL, NULL, required, &provided);

    if (provided != required)
      PRINT_DEBUG ('M', "Note: MPI thread support level = %d\n", provided);

    MPI_Comm_rank (MPI_COMM_WORLD, &_easypap_mpi_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &_easypap_mpi_size);
    PRINT_DEBUG ('i', "Init phase -1: MPI_Init_thread called (%d/%d)\n",
                 _easypap_mpi_rank, _easypap_mpi_size);
  } else
    PRINT_DEBUG ('i', "Init phase -1: [Process not launched by mpirun]\n");
#endif

  /* Allocate and initialize topology object. */
  hwloc_topology_init (&topology);

  /* Perform the topology detection. */
  hwloc_topology_load (topology);

  nb_cores = hwloc_get_nbobjs_by_type (topology, HWLOC_OBJ_PU);
  PRINT_DEBUG ('t', "%d-core machine detected\n", nb_cores);

  // Set kernel and variant
  {
    if (kernel_name == NULL)
      kernel_name = DEFAULT_KERNEL;

    if (variant_name == NULL)
      variant_name = opencl_used ? DEFAULT_OCL_VARIANT : DEFAULT_VARIANT;
  }

  hooks_establish_bindings (show_ocl_config | list_ocl_variants);

#ifdef ENABLE_SDL
  master_do_display = do_display;

  if (!(debug_enabled ('M') || easypap_proc_is_master ()))
    do_display = 0;

  if (!do_display)
    do_gmonitor = 0;

  // Create window, initialize rendering, preload image if appropriate
  graphics_init ();
#else
  if (!DIM)
    DIM = DEFAULT_DIM;
  PRINT_DEBUG ('i', "Init phase 0: DIM = %d\n", DIM);
#endif

  // At this point, we know the value of DIM
  check_tile_size ();

#ifdef ENABLE_MONITORING
#ifdef ENABLE_TRACE
  if (do_trace) {
    char filename[1024];

    if (easypap_mpirun)
      sprintf (filename, "%s/%s.%d%s", DEFAULT_EZV_TRACE_DIR,
               DEFAULT_EZV_TRACE_BASE, easypap_mpi_rank (),
               DEFAULT_EZV_TRACE_EXT);
    else
      strcpy (filename, DEFAULT_EASYVIEW_FILE);

    set_default_trace_label ();

    trace_record_init (filename, easypap_requested_number_of_threads (),
                       easypap_number_of_gpus (), DIM, trace_label);
  }
#endif
#endif

  if (the_config != NULL) {
    the_config (draw_param);
    PRINT_DEBUG ('i', "Init phase 1: config() hook called\n");
  } else {
    PRINT_DEBUG ('i', "Init phase 1: [no config() hook defined]\n");
  }

  if (opencl_used) {
    ocl_init (show_ocl_config, list_ocl_variants);
    ocl_build_program (list_ocl_variants);
    ocl_alloc_buffers ();
    PRINT_DEBUG ('i', "Init phase 2: OpenCL initialized\n");
  } else
    PRINT_DEBUG ('i', "Init phase 2: [OpenCL init not required]\n");

  // OpenCL context is initialized, so we can safely call kernel dependent
  // init() func which may allocate additional buffers.
  if (the_init != NULL) {
    the_init ();
    PRINT_DEBUG ('i', "Init phase 3: init() hook called\n");
  } else {
    PRINT_DEBUG ('i', "Init phase 3: [no init() hook defined]\n");
  }

  // Make sure at leat one task id (0 = anonymous) is stored in the trace
#ifdef ENABLE_TRACE
  if (do_trace)
    trace_record_commit_task_ids ();
#endif

  // Allocate memory for cur_img and next_img images
  img_data_alloc ();

  if (do_first_touch) {
    if (the_first_touch != NULL) {
      the_first_touch ();
      PRINT_DEBUG ('i', "Init phase 5: first-touch() hook called\n");
    } else
      PRINT_DEBUG ('i', "Init phase 5: [no first-touch() hook defined]\n");
  } else
    PRINT_DEBUG ('i', "Init phase 5: [first-touch policy not activated]\n");

#ifdef ENABLE_SDL
  // Allocate surfaces and textures
  graphics_alloc_images ();
#endif

  // Appel de la fonction de dessin spécifique, si elle existe
  if (the_draw != NULL) {
    the_draw (draw_param);
    PRINT_DEBUG ('i', "Init phase 6: kernel-specific draw() hook called\n");
  } else {
#ifndef ENABLE_SDL
    if (!do_first_touch || (the_first_touch == NULL))
      img_data_replicate (); // touch the data
#endif
    PRINT_DEBUG ('i',
                 "Init phase 6: [no kernel-specific draw() hook defined]\n");
  }

  if (opencl_used) {
    ocl_send_data ();
  } else
    PRINT_DEBUG ('i', "Init phase 7: [no OpenCL data transfer involved]\n");
}

int main (int argc, char **argv)
{
  int stable     = 0;
  int iterations = 0;

  filter_args (&argc, argv);

  if (list_ocl_variants) {
    // bypass complete initialization

    if (kernel_name == NULL)
      kernel_name = DEFAULT_KERNEL;

    ocl_init (0, list_ocl_variants);
    ocl_build_program (list_ocl_variants);

    // Never reached
    assert (0);
  }

  arch_flags_print ();

  init_phases ();

#ifdef ENABLE_SDL
  // version graphique
  if (master_do_display) {
    unsigned step = 0;

    if (opencl_used)
      graphics_share_texture_buffers ();

    if (the_refresh_img)
      the_refresh_img ();

    if (do_display)
      graphics_refresh (iterations);

    if (refresh_rate == -1)
      refresh_rate = 1;

    for (int quit = 0; !quit;) {

      int r = 0;

      if (do_pause && easypap_proc_is_master ()) {
        printf ("=== iteration %d ===\n", iterations);
        step = 1;
      }

      // Récupération éventuelle des événements clavier, souris, etc.
      if (do_display)
        do {
          SDL_Event evt;

          r = graphics_get_event (&evt, step | stable);

          if (r > 0) {
            switch (evt.type) {

            case SDL_QUIT:
              quit = 1;
              break;

            case SDL_KEYDOWN:
              // Si l'utilisateur appuie sur une touche
              switch (evt.key.keysym.sym) {
              case SDLK_ESCAPE:
              case SDLK_q:
                quit = 1;
                break;
              case SDLK_SPACE:
                step ^= 1;
                break;
              case SDLK_DOWN:
                update_refresh_rate (-1);
                break;
              case SDLK_UP:
                update_refresh_rate (1);
                break;
              case SDLK_h:
                gmonitor_toggle_heat_mode ();
                break;
              case SDLK_i:
                graphics_toggle_display_iteration_number ();
                break;
              default:;
              }
              break;

            case SDL_WINDOWEVENT:
              switch (evt.window.event) {
              case SDL_WINDOWEVENT_CLOSE:
                quit = 1;
                break;
              default:;
              }
              break;

            default:;
            }
          }

        } while ((r || step) && !quit);

#ifdef ENABLE_MPI
      if (easypap_mpirun)
        MPI_Allreduce (MPI_IN_PLACE, &quit, 1, MPI_INT, MPI_LOR,
                       MPI_COMM_WORLD);
#endif

      if (!stable) {
        if (quit) {
          PRINT_MASTER ("Computation interrupted at iteration %d\n",
                        iterations);
        } else {
          if (max_iter && iterations >= max_iter) {
            PRINT_MASTER ("Computation stopped after %d iterations\n",
                          iterations);
            stable = 1;
          } else {
            int n;

            if (max_iter && iterations + refresh_rate > max_iter)
              refresh_rate = max_iter - iterations;

            monitoring_start_iteration ();

            n = the_compute (refresh_rate);

            monitoring_end_iteration ();

            if (n > 0) {
              iterations += n;
              stable = 1;
              PRINT_MASTER ("Computation completed after %d itérations\n",
                            iterations);
            } else
              iterations += refresh_rate;

            if (!opencl_used && the_refresh_img)
              the_refresh_img ();

            if (do_thumbs && easypap_proc_is_master ()) {
              static unsigned iter_no = 0;

              if (opencl_used) {
                if (the_refresh_img)
                  the_refresh_img ();
                else
                  ocl_retrieve_data ();
              }

              graphics_save_thumbnail (++iter_no);
            }
          }

          if (do_display)
            graphics_refresh (iterations);
        }
      }
      if (stable && quit_when_done)
        quit = 1;
    }
  } else
#endif // ENABLE_SDL
  {
    // Version non graphique
    long temps;
    struct timeval t1, t2;
    int n;

    if (do_trace | do_thumbs)
      refresh_rate = 1;

    if (refresh_rate == -1) {
      if (max_iter)
        refresh_rate = max_iter;
      else
        refresh_rate = 1;
    }

    gettimeofday (&t1, NULL);

    while (!stable) {
      if (max_iter && iterations >= max_iter) {
        iterations = max_iter;
        stable     = 1;
      } else {

        if (max_iter && iterations + refresh_rate > max_iter)
          refresh_rate = max_iter - iterations;

        monitoring_start_iteration ();

        n = the_compute (refresh_rate);

        monitoring_end_iteration ();

#ifdef ENABLE_SDL
        if (do_thumbs && easypap_proc_is_master ()) {
          static unsigned iter_no = 0;

          if (the_refresh_img)
            the_refresh_img ();
          else if (opencl_used)
            ocl_retrieve_data ();

          graphics_save_thumbnail (++iter_no);
        }
#endif

        if (n > 0) {
          iterations += n;
          stable = 1;
        } else
          iterations += refresh_rate;
      }
    }

    gettimeofday (&t2, NULL);

    PRINT_MASTER ("Computation completed after %d iterations\n", iterations);

    temps = TIME_DIFF (t1, t2);

    if (easypap_proc_is_master ())
      output_perf_numbers (temps, iterations);

    PRINT_MASTER ("%ld.%03ld \n", temps / 1000, temps % 1000);
  }

#ifdef ENABLE_SDL
  // Check if final image should be dumped on disk
  if (do_dump && easypap_proc_is_master ()) {

    char filename[1024];

    if (the_refresh_img)
      the_refresh_img ();
    else if (opencl_used)
      ocl_retrieve_data ();

    sprintf (filename, "dump-%s-%s-dim-%d-iter-%d.png", kernel_name,
             variant_name, DIM, iterations);

    graphics_dump_image_to_file (filename);
  }
#endif

#ifdef ENABLE_MONITORING
#ifdef ENABLE_TRACE
  if (do_trace)
    trace_record_finalize ();
#endif
#endif

  if (the_finalize != NULL)
    the_finalize ();

#ifdef ENABLE_SDL
  graphics_clean ();
#endif

  img_data_free ();

#ifdef ENABLE_MPI
  if (easypap_mpirun)
    MPI_Finalize ();
#endif

  return 0;
}

static void usage (int val)
{
  fprintf (stderr, "Usage: %s [options]\n", progname);
  fprintf (stderr, "options can be:\n");
  fprintf (
      stderr,
      "\t-a\t| --arg <string>\t: pass argument <string> to draw function\n");
  fprintf (stderr, "\t-d\t| --debug-flags <flags>\t: enable debug messages "
                   "(see debug.h)\n");
  fprintf (stderr, "\t-du\t| --dump\t\t: dump final image to disk\n");
  fprintf (stderr,
           "\t-ft\t| --first-touch\t\t: touch memory on different cores\n");
  fprintf (stderr, "\t-h\t| --help\t\t: display help\n");
  fprintf (stderr, "\t-i\t| --iterations <n>\t: stop after n iterations\n");
  fprintf (stderr,
           "\t-k\t| --kernel <name>\t: override KERNEL environment variable\n");
  fprintf (stderr,
           "\t-lb\t| --label <name>\t: assign name <label> to current run\n");
  fprintf (stderr, "\t-lov\t| --list-ocl-variants\t: list OpenCL variants\n");
  fprintf (stderr, "\t-l\t| --load-image <file>\t: use PNG image <file>\n");
  fprintf (stderr,
           "\t-m \t| --monitoring\t\t: enable graphical thread monitoring\n");
  fprintf (stderr, "\t-mpi\t| --mpirun <args>\t: pass <args> to the mpirun MPI "
                   "process launcher\n");
  fprintf (stderr,
           "\t-n\t| --no-display\t\t: avoid graphical display overhead\n");
  fprintf (stderr, "\t-nt\t| --nb-tiles <N>\t: use N x N tiles\n");
  fprintf (stderr, "\t-nvs\t| --no-vsync\t\t: disable vertical sync\n");
  fprintf (stderr, "\t-o\t| --ocl\t\t\t: use OpenCL version\n");
  fprintf (stderr, "\t-of\t| --output-file <nfike>\t: output performance "
                   "numbers in <file>\n");
  fprintf (stderr, "\t-p\t| --pause\t\t: pause between iterations (press space "
                   "to continue)\n");
  fprintf (stderr, "\t-q\t| --quit\t\t: exit once iterations are done\n");
  fprintf (stderr,
           "\t-r\t| --refresh-rate <N>\t: display only 1/Nth of images\n");
  fprintf (stderr, "\t-s\t| --size <DIM>\t\t: use image of size DIM x DIM\n");
  fprintf (stderr,
           "\t-sr\t| --soft-rendering\t: disable hardware acceleration\n");
  fprintf (
      stderr,
      "\t-si\t| --show-iterations\t: display iterations in main window \n");
  fprintf (stderr,
           "\t-so\t| --show-ocl\t\t: display OpenCL platform and devices\n");
  fprintf (stderr, "\t-tn\t| --thumbnails\t\t: generate thumbnails\n");
  fprintf (stderr, "\t-tw\t| --tile-width <W>\t: use tiles of width W\n");
  fprintf (stderr, "\t-th\t| --tile-height <H>\t: use tiles of height H\n");
  fprintf (stderr, "\t-ts\t| --tile-size <TS>\t: use tiles of size TS x TS\n");
  fprintf (stderr, "\t-t\t| --trace\t\t: enable trace\n");
  fprintf (stderr,
           "\t-v\t| --variant <name>\t: select variant <name> of kernel\n");

  exit (val);
}

static void filter_args (int *argc, char *argv[])
{
  progname = argv[0];

  // Filter args
  //
  argv++;
  (*argc)--;
  while (*argc > 0) {
    if (!strcmp (*argv, "--no-vsync") || !strcmp (*argv, "-nvs")) {
      vsync = 0;
    } else if (!strcmp (*argv, "--no-display") || !strcmp (*argv, "-n")) {
      do_display = 0;
    } else if (!strcmp (*argv, "--pause") || !strcmp (*argv, "-p")) {
      do_pause = 1;
    } else if (!strcmp (*argv, "--quit") || !strcmp (*argv, "-q")) {
      quit_when_done = 1;
    } else if (!strcmp (*argv, "--help") || !strcmp (*argv, "-h")) {
      usage (0);
    } else if (!strcmp (*argv, "--soft-rendering") || !strcmp (*argv, "-sr")) {
      soft_rendering = 1;
    } else if (!strcmp (*argv, "--show-ocl") || !strcmp (*argv, "-so")) {
      show_ocl_config = 1;
      opencl_used     = 1;
      // do_display      = 0;
#ifdef ENABLE_SDL
    } else if (!strcmp (*argv, "--show-iterations") || !strcmp (*argv, "-si")) {
      graphics_toggle_display_iteration_number ();
#endif
    } else if (!strcmp (*argv, "--list-ocl-variants") ||
               !strcmp (*argv, "-lov")) {
      list_ocl_variants = 1;
      opencl_used       = 1;
      do_display        = 0;
    } else if (!strcmp (*argv, "--first-touch") || !strcmp (*argv, "-ft")) {
      do_first_touch = 1;
    } else if (!strcmp (*argv, "--monitoring") || !strcmp (*argv, "-m")) {
#ifndef ENABLE_SDL
      fprintf (stderr, "Warning: cannot monitor execution when ENABLE_SDL is "
                       "not defined\n");
#else
      do_gmonitor = 1;
#endif
    } else if (!strcmp (*argv, "--trace") || !strcmp (*argv, "-t")) {
#ifndef ENABLE_TRACE
      fprintf (
          stderr,
          "Warning: cannot generate trace if ENABLE_TRACE is not defined\n");
#else
      do_trace    = 1;
#endif
    } else if (!strcmp (*argv, "--thumbnails") || !strcmp (*argv, "-tn")) {
#ifndef ENABLE_SDL
      fprintf (stderr, "Warning: cannot generate thumbnails when ENABLE_SDL is "
                       "not defined\n");
#else
      do_thumbs   = 1;
#endif
    } else if (!strcmp (*argv, "--dump") || !strcmp (*argv, "-du")) {
#ifndef ENABLE_SDL
      fprintf (stderr, "Warning: cannot dump image to disk when ENABLE_SDL is "
                       "not defined\n");
#else
      do_dump     = 1;
#endif
    } else if (!strcmp (*argv, "--arg") || !strcmp (*argv, "-a")) {
      if (*argc == 1) {
        fprintf (stderr, "Error: parameter string is missing\n");
        usage (1);
      }
      (*argc)--;
      argv++;
      draw_param = *argv;
    } else if (!strcmp (*argv, "--label") || !strcmp (*argv, "-lb")) {
      if (*argc == 1) {
        fprintf (stderr, "Error: parameter string is missing\n");
        usage (1);
      }
      (*argc)--;
      argv++;
      snprintf (trace_label, MAX_LABEL, "%s", *argv);
    } else if (!strcmp (*argv, "--mpirun") || !strcmp (*argv, "-mpi")) {
#ifndef ENABLE_MPI
      fprintf (stderr, "Warning: --mpi has no effect when ENABLE_MPI "
                       "is not defined\n");
      (*argc)--;
      argv++;
#else
      if (*argc == 1) {
        fprintf (stderr, "Error: parameter string is missing\n");
        usage (1);
      }
      (*argc)--;
      argv++;
      easypap_mpirun = 1;
#endif
    } else if (!strcmp (*argv, "--ocl") || !strcmp (*argv, "-o")) {
      opencl_used = 1;
    } else if (!strcmp (*argv, "--kernel") || !strcmp (*argv, "-k")) {
      if (*argc == 1) {
        fprintf (stderr, "Error: kernel name is missing\n");
        usage (1);
      }
      (*argc)--;
      argv++;
      kernel_name = *argv;
    } else if (!strcmp (*argv, "--load-image") || !strcmp (*argv, "-l")) {
#ifndef ENABLE_SDL
      fprintf (stderr,
               "Warning: Cannot load image when ENABLE_SDL is not defined\n");
      (*argc)--;
      argv++;
#else
      if (*argc == 1) {
        fprintf (stderr, "Error: filename is missing\n");
        usage (1);
      }
      (*argc)--;
      argv++;
      easypap_image_file = *argv;
#endif
    } else if (!strcmp (*argv, "--size") || !strcmp (*argv, "-s")) {
      if (*argc == 1) {
        fprintf (stderr, "Error: DIM is missing\n");
        usage (1);
      }
      (*argc)--;
      argv++;
      DIM = atoi (*argv);
    } else if (!strcmp (*argv, "--nb-tiles") || !strcmp (*argv, "-nt")) {
      if (*argc == 1) {
        fprintf (stderr, "Error: number of tiles is missing\n");
        usage (1);
      }
      (*argc)--;
      argv++;
      NB_TILES_X = atoi (*argv);
      NB_TILES_Y = NB_TILES_X;
    } else if (!strcmp (*argv, "--tile-width") || !strcmp (*argv, "-tw")) {
      if (*argc == 1) {
        fprintf (stderr, "Error: tile width is missing\n");
        usage (1);
      }
      (*argc)--;
      argv++;
      TILE_W = atoi (*argv);
    } else if (!strcmp (*argv, "--tile-height") || !strcmp (*argv, "-th")) {
      if (*argc == 1) {
        fprintf (stderr, "Error: tile height is missing\n");
        usage (1);
      }
      (*argc)--;
      argv++;
      TILE_H = atoi (*argv);
    } else if (!strcmp (*argv, "--tile-size") || !strcmp (*argv, "-ts")) {
      if (*argc == 1) {
        fprintf (stderr, "Error: tile size is missing\n");
        usage (1);
      }
      (*argc)--;
      argv++;
      TILE_W = atoi (*argv);
      TILE_H = TILE_W;
    } else if (!strcmp (*argv, "--variant") || !strcmp (*argv, "-v")) {

      if (*argc == 1) {
        fprintf (stderr, "Error: variant name is missing\n");
        usage (1);
      }
      (*argc)--;
      argv++;
      variant_name = *argv;
    } else if (!strcmp (*argv, "--iterations") || !strcmp (*argv, "-i")) {
      if (*argc == 1) {
        fprintf (stderr, "Error: number of iterations is missing\n");
        usage (1);
      }
      (*argc)--;
      argv++;
      max_iter = atoi (*argv);
    } else if (!strcmp (*argv, "--refresh-rate") || !strcmp (*argv, "-r")) {
#ifndef ENABLE_SDL
      fprintf (stderr, "Warning: --refresh rate has no effect when ENABLE_SDL "
                       "is not defined\n");
      (*argc)--;
      argv++;
#else
      if (*argc == 1) {
        fprintf (stderr, "Error: refresh rate is missing\n");
        usage (1);
      }
      (*argc)--;
      argv++;
      refresh_rate = atoi (*argv);
#endif
    } else if (!strcmp (*argv, "--debug-flags") || !strcmp (*argv, "-d")) {
      if (*argc == 1) {
        fprintf (stderr, "Error: debug flags list is missing\n");
        usage (1);
      }
      (*argc)--;
      argv++;

      debug_init (*argv);
    } else if (!strcmp (*argv, "--output-file") || !strcmp (*argv, "-of")) {
      if (*argc == 1) {
        fprintf (stderr, "Error: filename is missing\n");
        usage (1);
      }
      (*argc)--;
      argv++;
      output_file = *argv;
    } else {
      fprintf (stderr, "Error: unknown option %s\n", *argv);
      usage (1);
    }

    (*argc)--;
    argv++;
  }
}
