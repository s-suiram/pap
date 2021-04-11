#include "trace_common.h"

unsigned cpu_colors[MAX_COLORS + 1] = {
    0xFFFF00FF, // Yellow
    0xFF0000FF, // Red
    0x00FF00FF, // Green
    0xAE4AFFFF, // Purple
    0x00FFFFFF, // Cyan
    0xB0B0B0FF, // Grey
    0x0033EEFF, // Royal Blue
    0xFFBFF7FF, // Pale Pink
    0xFFD591FF, // Cream
    0xCFFFBFFF, // Pale Green
    0xF08080FF, // Light Coral
    0xE000E0FF, // Magenta
    0x4B9447FF, // Dark green
    0x964B00FF, // Brown
    0xFFFFFFFF  // white
};

unsigned gpu_index[3] = {
    3, // TASK_TYPE_COMPUTE: magenta
    1, // TASK_TYPE_WRITE:   red
    0  // TASK_TYPE_READ:    yellow
};
