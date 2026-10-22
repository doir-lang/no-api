#pragma once

// ---------------------------------------------------------------------------
// Synchronization Extension
// ---------------------------------------------------------------------------

#include "compute.h"

NOAPI_EXTERN_C_BEGIN

/**
 * gpuGetSubmissionSemaphoreEXT – A semaphore counting the queue's finished submissions. gpuSubmit returns the index of the submission it made, so waiting on this semaphore for that index waits for that submission.
 *
 * @code
 * GpuCommandBuffer* cmd = gpuStartCommandRecording(queue);
 * GpuCommandBufferSpan buffers = {&cmd, 1};
 * uint64_t submission = gpuSubmit(queue, buffers, NULL, 0);
 * const GpuSemaphore* finished = gpuGetSubmissionSemaphoreEXT(queue);
 * assert(gpuWaitSemaphore(queue, finished, submission, UINT64_MAX) == submission);
 * @endcode
 *
 * @note The semaphore belongs to the queue; do not free it.
 *
 * @param queue The GPU queue whose submission progress is tracked.
 */
const GpuSemaphore* gpuGetSubmissionSemaphoreEXT(GpuQueue* queue);

/**
 * gpuWaitIdleEXT – Waits until all pending work on the queue's GPU is done.
 *
 * @code
 * GpuCommandBuffer* cmd = gpuStartCommandRecording(queue);
 * GpuCommandBufferSpan buffers = {&cmd, 1};
 * uint64_t submission = gpuSubmit(queue, buffers, NULL, 0);
 * gpuWaitIdleEXT(queue);
 * assert(gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), GPU_GET_VALUE, UINT64_MAX) >= submission);
 * @endcode
 *
 * @note On Vulkan a lost device ends the wait early, with errno set to the VkResult.
 *
 * @param queue The GPU queue to wait on.
 */
void gpuWaitIdleEXT(GpuQueue* queue);

/**
 * gpuSyncMemoryEXT – Record the copy that makes an allocation's two sides agree: for MEMORY_DEFAULT (and MEMORY_DESCRIPTOR_HEAP) the CPU's writes are copied to the GPU, and for MEMORY_READBACK (and MEMORY_TEXTURE_READBACK) the GPU's writes are copied back to the CPU once the submission finishes. GPU only memory has nothing to sync.
 *
 * Record it for an allocation the CPU wrote before the commands that read it, and for one the GPU wrote after the commands that wrote it.
 *
 * @code
 * uint32_t* values = (uint32_t*)gpuMalloc(queue, 4 * sizeof(uint32_t), 16, MEMORY_DEFAULT);
 * uint32_t* copy = (uint32_t*)gpuMalloc(queue, 4 * sizeof(uint32_t), 16, MEMORY_READBACK);
 * values[2] = 42;
 * gpu* values_gpu = gpuHostToDevicePointer(queue, values);
 * gpu* copy_gpu = gpuHostToDevicePointer(queue, copy);
 *
 * GpuCommandBuffer* cmd = gpuStartCommandRecording(queue);
 * gpuSyncMemoryEXT(cmd, values_gpu); // The CPU wrote it, so before the copy reads it
 * gpuMemCpy(cmd, copy_gpu, values_gpu, 4 * sizeof(uint32_t));
 * gpuSyncMemoryEXT(cmd, copy_gpu); // The GPU wrote it, so after
 * GpuCommandBufferSpan buffers = {&cmd, 1};
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, buffers, NULL, 0), UINT64_MAX); // copy[2] is now 42
 *
 * gpuFree(queue, values);
 * gpuFree(queue, copy);
 * @endcode
 *
 * @note Only does anything on WebGPU, whose buffers cannot stay mapped while the GPU uses them. On Vulkan the memory is shared and this is a no-op, so a program that skips it works there and nowhere else.
 *
 * @param cmd Command buffer to record the copy into.
 * @param mem GPU address of the start of the allocation to synchronize. An address that is not one is ignored.
 */
void gpuSyncMemoryEXT(GpuCommandBuffer* cmd, gpu* mem);

/**
 * gpuSyncMemoryImmediateEXT – gpuSyncMemoryEXT in a submission of its own, waited for before returning. Convenient for setting up memory the CPU filled in once.
 *
 * @code
 * float* weights = (float*)gpuMalloc(queue, 16 * sizeof(float), 16, MEMORY_DEFAULT);
 * for(int i = 0; i < 16; ++i) weights[i] = 1.0f / 16;
 * gpuSyncMemoryImmediateEXT(queue, gpuHostToDevicePointer(queue, weights));
 * gpuFree(queue, weights);
 * @endcode
 *
 * @note Spelled gpuSyncMemoryEXT in C++, where it is an overload of the command buffer flavour.
 * @note Only does anything on WebGPU; see gpuSyncMemoryEXT.
 *
 * @param queue Queue the memory was allocated on.
 * @param mem GPU address of the start of the allocation to synchronize. An address that is not one is ignored.
 */
void gpuSyncMemoryImmediateEXT(GpuQueue* queue, gpu* mem);

NOAPI_EXTERN_C_END

#ifdef __cplusplus
/**
 * gpuSyncMemoryEXT – C++ spelling of gpuSyncMemoryImmediateEXT.
 *
 * @code{.cpp}
 * auto weights = gpuMalloc<float>(queue, 16);
 * for(int i = 0; i < 16; ++i) weights[i] = 1.0f / 16;
 * gpuSyncMemoryEXT(queue, gpuHostToDevicePointer(queue, weights));
 * gpuFree(queue, weights);
 * @endcode
 *
 * @param queue Queue the memory was allocated on.
 * @param mem GPU address of the start of the allocation to synchronize. An address that is not one is ignored.
 */
inline void gpuSyncMemoryEXT(GpuQueue* queue, gpu* mem) { gpuSyncMemoryImmediateEXT(queue, mem); }
#endif
