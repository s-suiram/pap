
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "constants.h"
#include "cpustat.h"
#include "debug.h"
#include "error.h"
#include "global.h"
#include "graphics.h"
#include "img_data.h"
#include "minmax.h"
#include "ocl.h"
#include "time_macros.h"

#define MAX_PLATFORMS 3
#define MAX_DEVICES 5
#define MAX_KERNELS 32

unsigned GPU_TILE_W = 0;
unsigned GPU_TILE_H = 0;
unsigned GPU_SIZE_X = 0;
unsigned GPU_SIZE_Y = 0;

static size_t max_workgroup_size = 0;

static cl_platform_id chosen_platform = NULL;
cl_device_id chosen_device            = NULL;
static cl_program program; // compute program

cl_context context;
cl_kernel update_kernel;
cl_kernel compute_kernel;
static cl_kernel bench_kernel; // bench null kernel
cl_command_queue queue;
cl_mem tex_buffer, cur_buffer, next_buffer;

static size_t file_size (const char *filename)
{
  struct stat sb;

  if (stat (filename, &sb) < 0)
    exit_with_error ("Cannot access \"%s\" kernel file (%s)", filename,
                     strerror (errno));

  return sb.st_size;
}

static char *file_load (const char *filename)
{
  FILE *f;
  char *b;
  size_t s;
  size_t r;

  s = file_size (filename);
  b = malloc (s + 1);
  if (!b)
    exit_with_error ("Malloc failed (%s)", strerror (errno));

  f = fopen (filename, "r");
  if (f == NULL)
    exit_with_error ("Cannot open \"%s\" file (%s)", filename,
                     strerror (errno));

  r = fread (b, s, 1, f);
  if (r != 1)
    exit_with_error ("fread failed (%s)", strerror (errno));

  b[s] = '\0';

  return b;
}

unsigned easypap_number_of_gpus (void)
{
  return (opencl_used ? 1 : 0);
}

static void ocl_acquire (void)
{
  cl_int err;

  err = clEnqueueAcquireGLObjects (queue, 1, &tex_buffer, 0, NULL, NULL);
  check (err, "Failed to acquire lock");
}

static void ocl_release (void)
{
  cl_int err;

  err = clEnqueueReleaseGLObjects (queue, 1, &tex_buffer, 0, NULL, NULL);
  check (err, "Failed to release lock");
}

static void ocl_show_config (int quit, int verbose)
{
  cl_platform_id pf[MAX_PLATFORMS];
  cl_int nbp      = 0;
  cl_int chosen_p = -1, chosen_d = -1;
  char *glRenderer = NULL;
  char *str        = NULL;
  cl_int err;

#ifdef ENABLE_SDL
  if (do_display)
    glRenderer = (char *)glGetString (GL_RENDERER);
#endif

  // Get list of platforms
  err = clGetPlatformIDs (MAX_PLATFORMS, pf, (cl_uint *)&nbp);
  check (err, "Failed to get platform IDs");

  if (verbose == 2)
    printf ("%d OpenCL platforms detected\n", nbp);

  str = getenv ("PLATFORM");
  if (str != NULL)
    chosen_p = atoi (str);

  if (chosen_p >= nbp)
    exit_with_error (
        "Requested platform number (%d) should be in [0..%d] range", chosen_p,
        nbp - 1);

  str = getenv ("DEVICE");
  if (str != NULL)
    chosen_d = atoi (str);

  if (chosen_p == -1 && chosen_d != -1)
    chosen_p = 0;

  // Go through the list of platforms
  for (cl_uint p = 0; p < nbp; p++) {
    char name[1024], vendor[1024];
    cl_device_id devices[MAX_DEVICES];
    cl_int nbd = 0;
    cl_device_type dtype;

    if (chosen_p == p)
      chosen_platform = pf[p];

    err = clGetPlatformInfo (pf[p], CL_PLATFORM_NAME, 1024, name, NULL);
    check (err, "Failed to get Platform Info");

    err = clGetPlatformInfo (pf[p], CL_PLATFORM_VENDOR, 1024, vendor, NULL);
    check (err, "Failed to get Platform Info");

    if (verbose == 2)
      printf ("Platform %d: %s (%s)\n", p, name, vendor);

    err = clGetDeviceIDs (pf[p], CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices,
                          (cl_uint *)&nbd);

    if (chosen_p == p && chosen_d >= nbd)
      exit_with_error (
          "Requested device number (%d) should be in [0..%d] range", chosen_d,
          nbd - 1);

    // The chosen platform provides only one device, so we take device[0]
    if (chosen_p == p && chosen_d == -1 && nbd == 1) {
      chosen_d      = 0;
      chosen_device = devices[0];
    }

    // Go through the list of devices for platform p
    for (cl_uint d = 0; d < nbd; d++) {
      err = clGetDeviceInfo (devices[d], CL_DEVICE_NAME, 1024, name, NULL);
      check (err, "Cannot get type of device");

      err = clGetDeviceInfo (devices[d], CL_DEVICE_TYPE,
                             sizeof (cl_device_type), &dtype, NULL);
      check (err, "Cannot get type of device");

      // If user specified no PLATFORM/DEVICE, just take the first GPU found
      if (dtype == CL_DEVICE_TYPE_GPU && (chosen_p == -1 || chosen_p == p) &&
          (chosen_d == -1)) {
        chosen_p        = p;
        chosen_platform = pf[p];
        chosen_d        = d;
        chosen_device   = devices[d];
      }

      if (chosen_p == p) {
        if (chosen_d == d)
          chosen_device = devices[d];
        else if (chosen_d == -1 &&
                 d == nbd - 1) { // Last chance to select device
          chosen_d      = 0;
          chosen_device = devices[0];
        }
      }

      if (verbose == 2)
        printf ("%s Device %d: %s [%s]\n",
                (chosen_p == p && chosen_d == d) ? "+++" : "---", d,
                (dtype == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU", name);
      else if (verbose == 1 && chosen_p == p && chosen_d == d)
        printf ("Using OpenCL Device: %s [%s]\n",
                (dtype == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU", name);
    }
  }

  if (verbose == 2 && glRenderer != NULL)
    printf ("Note: OpenGL renderer uses [%s]\n", glRenderer);

  if (quit)
    exit (0);
}

void ocl_init (int show_config, int silent)
{
  cl_int err;
  int verbose = 0;

  if (!silent) {
    if (show_config || (debug_flags != NULL && debug_enabled ('o')))
      verbose = 2;
    else
      verbose = 1;
  }

  ocl_show_config (show_config, verbose);

  if (chosen_device == NULL)
    exit_with_error ("Device could not be automatically chosen: please use "
                     "PLATFORM and DEVICE to specify target");

  err = clGetDeviceInfo (chosen_device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                         sizeof (size_t), &max_workgroup_size, NULL);
  check (err, "Cannot get max workgroup size");

#ifdef ENABLE_SDL
  if (do_display) {
#ifdef __APPLE__
    CGLContextObj cgl_context          = CGLGetCurrentContext ();
    CGLShareGroupObj sharegroup        = CGLGetShareGroup (cgl_context);
    cl_context_properties properties[] = {
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
        (cl_context_properties)sharegroup, 0};
#else
    cl_context_properties properties[] = {
        CL_GL_CONTEXT_KHR,
        (cl_context_properties)glXGetCurrentContext (),
        CL_GLX_DISPLAY_KHR,
        (cl_context_properties)glXGetCurrentDisplay (),
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)chosen_platform,
        0};
#endif

    context = clCreateContext (properties, 1, &chosen_device, NULL, NULL, &err);
  } else
#endif // ENABLE_SDL
    context = clCreateContext (NULL, 1, &chosen_device, NULL, NULL, &err);

  check (err, "Failed to create compute context. Please make sure OpenCL and "
              "OpenGL both use the same device (--show-ocl).");

  // Create a command queue
  //
  queue = clCreateCommandQueue (context, chosen_device,
                                CL_QUEUE_PROFILING_ENABLE, &err);
  check (err, "Failed to create command queue. Please make sure OpenCL and "
              "OpenGL both use the same device (--show-ocl).");
}

void ocl_alloc_buffers (void)
{
  // Allocate buffers inside device memory
  //
  cur_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE,
                               sizeof (unsigned) * DIM * DIM, NULL, NULL);
  if (!cur_buffer)
    exit_with_error ("Failed to allocate input buffer");

  next_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE,
                                sizeof (unsigned) * DIM * DIM, NULL, NULL);
  if (!next_buffer)
    exit_with_error ("Failed to allocate output buffer");
}

void ocl_map_textures (GLuint texid)
{
  cl_int err;
  // Shared texture buffer with OpenGL
  //
#ifdef ENABLE_SDL
  tex_buffer = clCreateFromGLTexture (context, CL_MEM_READ_WRITE, GL_TEXTURE_2D,
                                      0, texid, &err);
#else
  err = 1;
#endif
  check (err, "Failed to map texture buffer\n");
}

static void ocl_list_variants (void)
{
  cl_kernel kernels[MAX_KERNELS];
  char buffer[1024];
  cl_uint kernels_found = 0;
  cl_int err;

  err =
      clCreateKernelsInProgram (program, MAX_KERNELS, kernels, &kernels_found);
  check (err, "Failed to get list of kernels from program\n");

  for (int k = 0; k < kernels_found; k++) {
    size_t len;
    err = clGetKernelInfo (kernels[k], CL_KERNEL_FUNCTION_NAME, 1024, buffer,
                           &len);
    check (err, "Failed to get name of kernel\n");

    printf ("%s\n", buffer);
  }

  exit (0);
}

#define CALIBRATION_BURST 4
#define CALIBRATION_ITER  64
long _calibration_delta = 0;

static void calibrate (void)
{
  size_t global[1] = {4096 * 64}; // global domain size for our calculation
  size_t local[1]  = {16};  // local domain size for our calculation
  cl_event events[CALIBRATION_BURST];
  long t;
  cl_int err;

  bench_kernel = clCreateKernel (program, "bench_kernel", &err);
  check (err, "Failed to create bench kernel");

  // Warmup
  for (unsigned it = 0; it < 10; it++) {

    err = clEnqueueNDRangeKernel (queue, bench_kernel, 1, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute bench kernel");
  }
  clFinish (queue);

  for (int i = 0; i < CALIBRATION_ITER; i++) {
    for (unsigned it = 0; it < CALIBRATION_BURST; it++)
      err = clEnqueueNDRangeKernel (queue, bench_kernel, 1, NULL, global, local,
                                    0, NULL, &events[it]);
    clFinish (queue);
    t = what_time_is_it ();

    cl_ulong end;

    clGetEventProfilingInfo (events[CALIBRATION_BURST - 1],
                             CL_PROFILING_COMMAND_END, sizeof (cl_ulong), &end,
                             NULL);

    if (i == 0) {
      _calibration_delta = t - (end / 1000);
    } else {
      _calibration_delta = min (_calibration_delta, t - (end / 1000));
    }

    for (unsigned it = 0; it < CALIBRATION_BURST; it++)
      clReleaseEvent (events[it]);
  }
}

void ocl_build_program (int list_variants)
{
  cl_int err;
  char *str = NULL;
  char buffer[1024];

  if (!GPU_SIZE_X) {
    str = getenv ("SIZE");
    if (str != NULL)
      GPU_SIZE_X = atoi (str);
    else
      GPU_SIZE_X = DIM;

    if (GPU_SIZE_X > DIM)
      exit_with_error ("GPU_SIZE_X (%d) cannot exceed DIM (%d)", GPU_SIZE_X, DIM);
  }

  if (!GPU_SIZE_Y)
    GPU_SIZE_Y = GPU_SIZE_X;

  if (!GPU_TILE_W) {
    str = getenv ("TILEX");
    if (str != NULL)
      GPU_TILE_W = atoi (str);
    else
      GPU_TILE_W = DEFAULT_GPU_TILE_SIZE;
  }

  if (!GPU_TILE_H) {
    str = getenv ("TILEY");
    if (str != NULL)
      GPU_TILE_H = atoi (str);
    else
      GPU_TILE_H = GPU_TILE_W;
  }

  if (GPU_SIZE_X % GPU_TILE_W)
    fprintf (stderr, "Warning: GPU_SIZE_X (%d) is not a multiple of GPU_TILE_W (%d)!\n",
             GPU_SIZE_X, GPU_TILE_W);

  if (GPU_SIZE_Y % GPU_TILE_H)
    fprintf (stderr, "Warning: GPU_SIZE_Y (%d) is not a multiple of GPU_TILE_H (%d)!\n",
             GPU_SIZE_Y, GPU_TILE_H);

  // Load program source into memory
  //
  sprintf (buffer, "kernel/ocl/%s.cl", kernel_name);
  const char *opencl_prog = file_load (buffer);

  // Attach program source to context
  //
  program = clCreateProgramWithSource (context, 1, &opencl_prog, NULL, &err);
  check (err, "Failed to create program");

  // Compile program
  //
  if (draw_param)
    sprintf (buffer,
             "-cl-mad-enable -cl-fast-relaxed-math"
             " -DDIM=%d -DGPU_SIZE_X=%d -DGPU_SIZE_Y=%d -DGPU_TILE_W=%d -DGPU_TILE_H=%d -DKERNEL_%s"
             " -DPARAM=%s",
             DIM, GPU_SIZE_X, GPU_SIZE_Y, GPU_TILE_W, GPU_TILE_H, kernel_name, draw_param);
  else
    sprintf (
        buffer,
        "-cl-mad-enable -cl-fast-relaxed-math"
        " -DDIM=%d -DGPU_SIZE_X=%d -DGPU_SIZE_Y=%d -DGPU_TILE_W=%d -DGPU_TILE_H=%d -DKERNEL_%s",
        DIM, GPU_SIZE_X, GPU_SIZE_Y, GPU_TILE_W, GPU_TILE_H, kernel_name);

  err = clBuildProgram (program, 0, NULL, buffer, NULL, NULL);

  // Display compiler log
  //
  {
    size_t len;

    clGetProgramBuildInfo (program, chosen_device, CL_PROGRAM_BUILD_LOG, 0,
                           NULL, &len);

    if (len > 2 && len <= 2048) {
      char buffer[len];

      fprintf (stderr, "--- OpenCL Compiler log ---\n");
      clGetProgramBuildInfo (program, chosen_device, CL_PROGRAM_BUILD_LOG,
                             sizeof (buffer), buffer, NULL);
      fprintf (stderr, "%s\n", buffer);
      fprintf (stderr, "---------------------------\n");
    }
  }

  if (err != CL_SUCCESS)
    exit_with_error ("Failed to build program");

  if (list_variants)
    ocl_list_variants ();

  // Create the compute kernel in the program we wish to run
  //
  sprintf (buffer, "%s_%s", kernel_name, variant_name);
  compute_kernel = clCreateKernel (program, buffer, &err);
  check (err, "Failed to create compute kernel <%s>", buffer);

  PRINT_DEBUG ('o', "Using OpenCL kernel: %s_%s\n", kernel_name, variant_name);

  sprintf (buffer, "%s_update_texture", kernel_name);

  // First look for kernel-specific version of update_texture
  update_kernel = clCreateKernel (program, buffer, &err);
  if (err != CL_SUCCESS) {
    // Fall back to generic version
    update_kernel = clCreateKernel (program, "update_texture", &err);
    check (err, "Failed to create update kernel <update_texture>");
  }

  calibrate ();

  printf ("Using %dx%d workitems grouped in %dx%d tiles \n", GPU_SIZE_X, GPU_SIZE_Y,
          GPU_TILE_W, GPU_TILE_H);
}

void ocl_send_data (void)
{
  cl_int err;
  cl_event event;

  err = clEnqueueWriteBuffer (queue, cur_buffer, CL_TRUE, 0,
                              sizeof (unsigned) * DIM * DIM, image, 0, NULL,
                              &event);
  check (err, "Failed to write to cur_buffer");

  err = clEnqueueWriteBuffer (queue, next_buffer, CL_TRUE, 0,
                              sizeof (unsigned) * DIM * DIM, alt_image, 0, NULL,
                              &event);
  check (err, "Failed to write to next_buffer");

  PRINT_DEBUG (
      'i', "Init phase 7 : Initial image data transferred to OpenCL device\n");
}

void ocl_retrieve_data (void)
{
  cl_int err;

  err =
      clEnqueueReadBuffer (queue, cur_buffer, CL_TRUE, 0,
                           sizeof (unsigned) * DIM * DIM, image, 0, NULL, NULL);
  check (err, "Failed to read from cur_buffer");

  PRINT_DEBUG ('o', "Image retrieved from device.\n");
}

unsigned ocl_invoke_kernel_generic (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y}; // global domain size for our calculation
  size_t local[2]  = {GPU_TILE_W, GPU_TILE_H}; // local domain size for our calculation
  cl_int err;

  monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {

    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
    err |= clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &next_buffer);
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");

    // Swap buffers
    {
      cl_mem tmp  = cur_buffer;
      cur_buffer  = next_buffer;
      next_buffer = tmp;
    }
  }

  clFinish (queue);

  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  return 0;
}

void ocl_update_texture (void)
{
  size_t global[2] = {DIM, DIM}; // global domain size for our calculation
  size_t local[2]  = {16, 16};   // local domain size for our calculation
  cl_int err;

  ocl_acquire ();

  // Set kernel arguments
  //
  err = 0;
  err |= clSetKernelArg (update_kernel, 0, sizeof (cl_mem), &cur_buffer);
  err |= clSetKernelArg (update_kernel, 1, sizeof (cl_mem), &tex_buffer);
  check (err, "Failed to set kernel arguments");

  err = clEnqueueNDRangeKernel (queue, update_kernel, 2, NULL, global, local, 0,
                                NULL, NULL);
  check (err, "Failed to execute kernel");

  ocl_release ();

  clFinish (queue);
}

size_t ocl_get_max_workgroup_size (void)
{
  return max_workgroup_size;
}

static inline long ocl_start_time (cl_event evt)
{
  cl_ulong t_start;

  clGetEventProfilingInfo (evt, CL_PROFILING_COMMAND_START, sizeof (cl_ulong),
                           &t_start, NULL);

  return (long)(t_start / 1000) + _calibration_delta;
}

static inline long ocl_end_time (cl_event evt)
{
  cl_ulong t_end;

  clGetEventProfilingInfo (evt, CL_PROFILING_COMMAND_END, sizeof (cl_ulong),
                           &t_end, NULL);

  return (long)(t_end / 1000) + _calibration_delta;
}

long ocl_monitor (cl_event evt, int x, int y, int width, int height,
                  task_type_t task_type)
{
  long start, end;
  unsigned gpu_lane = easypap_gpu_lane (TASK_TYPE_COMPUTE);

  start = ocl_start_time (evt);
  end   = ocl_end_time (evt);

  long now = what_time_is_it ();
  if (end > now) {
    fprintf (stderr, "Warning: end of kernel (%s) ahead of current time by %ld µs\n", task_type == TASK_TYPE_COMPUTE ? "TASK_TYPE_COMPUTE" : "TASK_TYPE_TRANSFER", end - now);
  }

  PRINT_DEBUG ('m', "[%s] start: %ld, end: %ld\n", "kernel", start, end);

  monitoring_gpu_tile (x, y, width, height, gpu_lane, start, end, task_type);

  return end - start;
}
