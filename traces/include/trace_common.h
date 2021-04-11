#ifndef TRACE_COMMON_IS_DEF
#define TRACE_COMMON_IS_DEF


#define MAX_COLORS 14

extern unsigned cpu_colors[];
extern unsigned gpu_index[];

#define TRACE_BEGIN_ITER   0x101
#define TRACE_BEGIN_TILE   0x102
#define TRACE_END_TILE     0x103
#define TRACE_NB_THREADS   0x104
#define TRACE_NB_ITER      0x105
#define TRACE_DIM          0x106
#define TRACE_END_ITER     0x107
#define TRACE_LABEL        0x108
#define TRACE_TASKID_COUNT 0x109
#define TRACE_TASKID       0x10A

#define DEFAULT_EZV_TRACE_DIR "traces/data"
#define DEFAULT_EZV_TRACE_BASE "ezv_trace_current"
#define DEFAULT_EZV_TRACE_EXT  ".evt"
#define DEFAULT_EZV_TRACE_FILE DEFAULT_EZV_TRACE_BASE DEFAULT_EZV_TRACE_EXT
#define DEFAULT_EASYVIEW_FILE DEFAULT_EZV_TRACE_DIR "/" DEFAULT_EZV_TRACE_FILE

typedef enum {
    TASK_TYPE_COMPUTE,
    TASK_TYPE_WRITE,
    TASK_TYPE_READ
} task_type_t;

#define TASK_TYPE_BITS 2U

#define TASK_COMBINE(ttype,tid) ((unsigned)(ttype) | ((unsigned)(tid) << TASK_TYPE_BITS))
#define TASK_EXTRACT_TID(v)     ((unsigned)(v) >> TASK_TYPE_BITS)
#define TASK_EXTRACT_TTYPE(v)   ((unsigned)(v) & ((1 << TASK_TYPE_BITS) - 1))


#endif
