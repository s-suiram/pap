#include <fcntl.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef ENABLE_FUT

#define CONFIG_FUT
#include <fut.h>
#define BUFFER_SIZE (16 << 20)

#endif

#include "error.h"
#include "trace_common.h"
#include "trace_data.h"
#include "trace_record.h"

unsigned do_trace = 0;

static unsigned task_ids_count = 0;

void trace_record_init (char *file, unsigned cpu, unsigned gpu, unsigned dim,
                        char *label)
{
  fut_set_filename (file);
  enable_fut_flush ();

  if (fut_setup (BUFFER_SIZE, 0xffff, 0) < 0)
    exit_with_error ("fut_setup");

  // We use 2 lanes per GPU : one for computations, the other for data transfers
  FUT_PROBE2 (0x1, TRACE_NB_THREADS, cpu, gpu * 2);
  FUT_PROBE1 (0x1, TRACE_DIM, dim);
  if (label != NULL)
    FUT_PROBESTR (0x1, TRACE_LABEL, label);
}

void trace_record_finalize (void)
{
  if (fut_endup ("temp") < 0)
    exit_with_error ("fut_endup");

  if (fut_done () < 0)
    exit_with_error ("fut_done");
}

void trace_record_declare_task_ids (char *task_ids[])
{
  if (!do_trace)
    return;

  // FIXME
  task_ids_count = 1;

  if (task_ids != NULL) {
    for (int i = 0; task_ids[i] != NULL; i++) {
      task_ids_count++;
    }
  }

  FUT_PROBE1 (0x1, TRACE_TASKID_COUNT, task_ids_count);
  FUT_PROBESTR (0x1, TRACE_TASKID, "anonymous"); // task id 0
  if (task_ids != NULL)
    for (int i = 0; task_ids[i] != NULL; i++)
      FUT_PROBESTR (0x1, TRACE_TASKID, task_ids[i]); // task id i + 1
}

void trace_record_commit_task_ids (void)
{
  if (task_ids_count == 0) // declare_task_ids not called by user
    trace_record_declare_task_ids (NULL);
}

void __trace_record_start_iteration (long time)
{
  FUT_PROBE1 (0x1, TRACE_BEGIN_ITER, time);
}

void __trace_record_end_iteration (long time)
{
  FUT_PROBE1 (0x1, TRACE_END_ITER, time);
}

void __trace_record_start_tile (long time, unsigned cpu)
{
  FUT_PROBE2 (0x1, TRACE_BEGIN_TILE, time, cpu);
}

void __trace_record_end_tile (long time, unsigned cpu, unsigned x, unsigned y,
                              unsigned w, unsigned h, int task_type,
                              int task_id)
{
  if (task_id >= task_ids_count)
    exit_with_error ("monitoring_end_tile: task id %d is too large (should < %d)%s\n",
                     task_id, task_ids_count, (task_ids_count == 1) ? ". Probable cause: monitoring_declare_task_ids not called" : "");
  FUT_PROBE7 (0x1, TRACE_END_TILE, time, cpu, x, y, w, h,
              TASK_COMBINE (task_type, task_id));
}
