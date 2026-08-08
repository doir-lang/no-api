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

/**
 * @brief gpuSyncMemoryEXT - If this memory is MEMORY_DEFAULT this will copy the CPU data to the GPU.
 * If this memory is MEMORY_READBACK this will copy the GPU data to the CPU!
 * 
 * @note Only necessary on WebGPU due to the restrictiveness of its buffer model. 
 * On other backends this is a noop.
 *
 * @param cmd Command buffer to enqueue the commands on
 * @param mem The memory to synchronize
 * @return  
 */
void gpuSyncMemoryEXT(GpuCommandBuffer* cmd, gpu* mem);
