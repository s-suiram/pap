#ifndef OCL_IS_DEF
#define OCL_IS_DEF

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <OpenGL/CGLContext.h>
#include <OpenGL/CGLCurrent.h>
#else
#include <CL/opencl.h>
#include <GL/glx.h>
#endif

#ifdef ENABLE_SDL
#include <SDL_opengl.h>
#endif

#include "error.h"
#include "monitoring.h"

void ocl_init (int show_config, int silent);
void ocl_build_program (int list_variants);
void ocl_alloc_buffers (void);
void ocl_map_textures (GLuint texid);
void ocl_send_data (void);
void ocl_retrieve_data (void);
unsigned ocl_invoke_kernel_generic (unsigned nb_iter);
void ocl_update_texture (void);
unsigned easypap_number_of_gpus (void);
size_t ocl_get_max_workgroup_size (void);

#define check(err, format, ...)                                                \
  do {                                                                         \
    if (err != CL_SUCCESS)                                                     \
      exit_with_error (format " [OCL err %d]", ##__VA_ARGS__, err);            \
  } while (0)

// Kernels get executed by GPU_SIZE_Y * GPU_SIZE_X threads
// Tiles have a size of GPU_TILE_H * GPU_TILE_W
extern unsigned GPU_SIZE_X, GPU_SIZE_Y, GPU_TILE_W, GPU_TILE_H, GPU_TILE_W;

extern cl_context context;
extern cl_kernel compute_kernel;
extern cl_command_queue queue;
extern cl_mem cur_buffer, next_buffer;
extern long _calibration_delta;

long ocl_monitor (cl_event evt, int x, int y, int width, int height,
                  task_type_t task_type);

#endif
