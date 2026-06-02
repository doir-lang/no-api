#pragma once

// ---------------------------------------------------------------------------
// Synchronization Extension
// ---------------------------------------------------------------------------

#include "compute.hpp"

/**
 * gpuGetSubmissionSemaphoreEXT – Returns a semaphore that tracks the currently
 * finished submission.
 *
 * gpuSubmit returns a submission index and thus you can wait on this semaphore
 * to determine when a submission is complete.
 *
 * @param queue The GPU queue whose submission progress is tracked.
 */
const GpuSemaphore* gpuGetSubmissionSemaphoreEXT(GpuQueue* queue);

/**
 * gpuWaitIdleEXT – Waits until all pending work on the queue's GPU is done.
 *
 * @param queue The GPU queue to wait on.
 */
void gpuWaitIdleEXT(GpuQueue* queue);
