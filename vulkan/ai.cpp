/**
 * gpu_barrier.cpp
 *
 * Vulkan 1.2+ implementation of the stage-based GPU barrier / split-barrier API.
 *
 * Design notes
 * ============
 *
 * The public API intentionally hides Vulkan resource states / image layouts and
 * exposes only producer-consumer stage pairs plus optional hazard flags for the
 * handful of non-coherent caches that matter in practice.
 *
 * Mapping strategy
 * ----------------
 *
 * STAGE → VkPipelineStageFlagBits2KHR
 * Straight bitmask translation (see stageToVk).
 *
 * Hazard flags → VkAccessFlagBits2KHR (src + dst sides of a memory barrier)
 * HAZARD_DRAW_ARGUMENTS → flush SHADER_WRITE, invalidate INDIRECT_COMMAND_READ
 * HAZARD_DESCRIPTORS → flush SHADER_WRITE, invalidate DESCRIPTOR_SET_READ
 * HAZARD_DEPTH_STENCIL → flush SHADER_WRITE, invalidate DEPTH_STENCIL_ATTACHMENT_*
 *
 * No-hazard barrier
 * A VkMemoryBarrier2KHR with MEMORY_WRITE → MEMORY_READ covers all writes
 * on a coherent-L2 architecture. The driver still inserts the correct
 * hardware stall/flush for the given stage pair.
 *
 * gpuSignalAfter
 * SET → vkCmdWriteBufferMarker2AMD (AMD) or vkCmdFillBuffer (non-AMD)
 * MAX → 1-thread atomic compute shader (requires VK_EXT_shader_atomic_int64)
 * OR → 1-thread atomic compute shader (requires VK_EXT_shader_atomic_int64)
 *
 * gpuWaitBefore
 * OP_GREATER_EQUAL + no mask → VkSemaphoreTypeTimeline fast path
 * everything else → polling compute shader
 *
 * GpuCommandBuffer is assumed to expose:
 * VkCommandBuffer cmd
 * VkDevice device
 * GpuDeviceCaps caps
 *
 * GpuDeviceCaps is assumed to expose:
 * bool hasAMDBufferMarker
 * bool hasShaderAtomicInt64
 * VkPipelineLayout atomicPipelineLayout (push: addr, value)
 * VkPipelineLayout waitPipelineLayout (push: addr, value, mask, op)
 * VkPipeline getAtomicPipeline(VkDevice, SIGNAL)
 * VkPipeline getWaitPipeline(VkDevice, OP)
 * VkSemaphore getTimelineSemaphore(void* ptrGpu) // null if not mapped
 * VkBuffer getBufferForAddr(VkDeviceAddress)
 * VkDeviceSize getOffsetForAddr(VkDeviceAddress)
 */

#include "noapi.hpp"

#include <vulkan/vulkan.h>
#include <cassert>
#include <cstring>
#include <vulkan/vulkan_core.h>

VkPipelineStageFlags2KHR stage2vulkan(STAGE stage) {
	VkPipelineStageFlags2KHR out = 0;

	if (stage & STAGE_TRANSFER)
		out |= VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR;

	if (stage & STAGE_COMPUTE)
		out |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;

	if (stage & STAGE_VERTEX_SHADER)
		// PRE_RASTERIZATION_SHADERS covers vertex + tessellation + geometry +
		// mesh stages in a single bit (Vulkan 1.3 / VK_KHR_synchronization2).
		out |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT_KHR;

	if (stage & STAGE_PIXEL_SHADER)
		out |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT_KHR;

	if (stage & STAGE_RASTER_COLOR_OUT)
		out |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR;

	if (stage & STAGE_RASTER_DEPTH_OUT)
		// Both early and late tests can write depth; include both.
		out |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT_KHR | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT_KHR;

	// If caller passed STAGE_ALL or nothing translated, use the nuclear option.
	if (stage == STAGE_ALL || out == 0)
		out = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR;

	return out;
}

std::pair<VkAccessFlags2KHR, VkAccessFlags2KHR> hazard2access(HAZARD_FLAGS hazards) {
	VkAccessFlags2KHR src_access = {}, dst_access = {};
	
	// HAZARD_DRAW_ARGUMENTS
	// A compute shader wrote an indirect argument buffer. The command
	// processor must not prefetch the arguments until the write is visible.
	//
	if (hazards & HAZARD_DRAW_ARGUMENTS) {
		src_access |= VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
		dst_access |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT_KHR;
	}

	// HAZARD_DESCRIPTORS
	// The global descriptor heap was updated (CPU or compute write).
	// Invalidate the sampler's internal descriptor cache so it re-fetches
	// the updated entries.
	//
	if (hazards & HAZARD_DESCRIPTORS) {
		src_access |= VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
		dst_access |= VK_ACCESS_2_DESCRIPTOR_BUFFER_READ_BIT_EXT | VK_ACCESS_2_UNIFORM_READ_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT_KHR;
	}

	// HAZARD_DEPTH_STENCIL
	// Compute wrote to memory that will be bound as a depth buffer.
	// Invalidate HiZ / stencil cache metadata.
	//
	if (hazards & HAZARD_DEPTH_STENCIL) {
		src_access |= VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
		dst_access |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT_KHR | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT_KHR;
	}

	// Always include the generic dependency so the barrier is never a no-op.
	src_access |= VK_ACCESS_2_MEMORY_WRITE_BIT_KHR;
	dst_access |= VK_ACCESS_2_MEMORY_READ_BIT_KHR;

	return {src_access, dst_access};
}

// ─────────────────────────────────────────────────────────────────────────────
// gpuBarrier
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Record a VkMemoryBarrier2KHR expressing a producer-to-consumer dependency.
 *
 * A single global memory barrier (no buffer or image barriers) is correct here
 * because:
 * 1. On coherent-L2 hardware (RDNA2+, Ampere+, Intel Xe) the L2 is shared
 * between all shader engines; only per-shader-engine L0 caches and
 * fixed-function caches (RB+, HiZ, CB metadata) need explicit flushing.
 * 2. The API has no per-resource layout state, so image layout transitions
 * are handled externally or not needed (GENERAL layout throughout).
 *
 * The driver translates the stage mask pair into exactly the right hardware
 * stall + cache-flush sequence for the target architecture.
 */
void gpuBarrier(GpuCommandBuffer& cmd, STAGE before, STAGE after, HAZARD_FLAGS hazards) {
	VkPipelineStageFlags2KHR src_stage = stage2vulkan(before);
	VkPipelineStageFlags2KHR dst_stage = stage2vulkan(after);
	auto [src_access, dst_access] = hazard2access(hazards);

	const VkMemoryBarrier2KHR barrier {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
		.pNext = nullptr,
		.srcStageMask = src_stage,
		.srcAccessMask = src_access,
		.dstStageMask = dst_stage,
		.dstAccessMask = dst_access,
	};
	const VkDependencyInfoKHR dependency {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
		.pNext = nullptr,
		.dependencyFlags = 0,
		.memoryBarrierCount = 1,
		.pMemoryBarriers = &barrier,
		.bufferMemoryBarrierCount = 0,
		.pBufferMemoryBarriers = nullptr,
		.imageMemoryBarrierCount = 0,
		.pImageMemoryBarriers = nullptr,
	};
	vkCmdPipelineBarrier2KHR(cmd.command_buffer, &dependency);
}

// ─────────────────────────────────────────────────────────────────────────────
// gpuSignalAfter
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Split-barrier producer: after `before` finishes, write/atomic-update the
 * 64-bit counter at `ptrGpu`.
 *
 * Implementation paths (in priority order):
 *
 * 1. vkCmdFillBuffer (SIGNAL_ATOMIC_SET)
 * Requires a barrier before (to sequence the producer stage) and after
 * (to make the fill visible to subsequent waits). Two 32-bit fills cover
 * the 64-bit counter. Not a true split barrier — the command processor
 * stalls for the leading barrier — but correct.
 *
 * 2. Atomic compute dispatch (SIGNAL_ATOMIC_MAX / SIGNAL_ATOMIC_OR)
 * A 1-thread compute shader executes atomicMax / atomicOr on a BDA pointer.
 * Requires VK_EXT_shader_atomic_int64. A barrier sequences the producer
 * stage before the compute dispatch.
 */
void gpuSignalAfter(GpuCommandBuffer& cmd, STAGE before, void* ptrGpu, uint64_t value, SIGNAL signal) {
	auto addr = (VkDeviceAddress)ptrGpu;

	if (signal == SIGNAL_ATOMIC_SET) {
		// Barrier: `before` → TRANSFER (sequence the producer before the fill)
		{
			const VkMemoryBarrier2KHR mb {
				.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
				.srcStageMask = stage2vulkan(before),
				.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT_KHR,
				.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR,
				.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR,
			};
			const VkDependencyInfoKHR dep {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
				.memoryBarrierCount = 1,
				.pMemoryBarriers = &mb,
			};
			vkCmdPipelineBarrier2KHR(cmd.command_buffer, &dep);
		}

		VkBuffer dstBuf = cmd.queue->allocations[addr].first;

		const uint32_t low = static_cast<uint32_t>(value & 0xFFFF'FFFFu);
		const uint32_t high = static_cast<uint32_t>(value >> 32u);

		vkCmdFillBuffer(cmd.command_buffer, dstBuf, 0, sizeof(uint32_t), low);
		vkCmdFillBuffer(cmd.command_buffer, dstBuf, 4, sizeof(uint32_t), high);

		// Barrier: TRANSFER → ALL_COMMANDS (make fills visible everywhere)
		{
			const VkMemoryBarrier2KHR mb {
				.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
				.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR,
				.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR,
				.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,
				.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT_KHR,
			};
			const VkDependencyInfoKHR dep {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
				.memoryBarrierCount = 1,
				.pMemoryBarriers = &mb,
			};
			vkCmdPipelineBarrier2KHR(cmd.command_buffer, &dep);
		}
		return;
	}


	assert((signal == SIGNAL_ATOMIC_MAX || signal == SIGNAL_ATOMIC_OR) && "Unexpected SIGNAL value");
	// assert(cmd.caps.hasShaderAtomicInt64 && "SIGNAL_ATOMIC_MAX/OR requires VK_EXT_shader_atomic_int64");

	// Barrier: `before` → COMPUTE (producer must finish before the atomic)
	{
		const VkMemoryBarrier2KHR mb {
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
			.srcStageMask = stage2vulkan(before),
			.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT_KHR,
			.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR,
			.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT_KHR
						 | VK_ACCESS_2_SHADER_WRITE_BIT_KHR,
		};
		const VkDependencyInfoKHR dep {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
			.memoryBarrierCount = 1,
			.pMemoryBarriers = &mb,
		};
		vkCmdPipelineBarrier2KHR(cmd.command_buffer, &dep);
	}

	VkPipeline pipeline = cmd.caps.getAtomicPipeline(cmd.device, signal);
	assert(pipeline != VK_NULL_HANDLE && "Atomic compute pipeline not compiled");

	vkCmdBindPipeline(cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

	// Push constants layout: { uint64_t addr; uint64_t value; }
	// Shader GLSL (atomicMax variant):
	//
	// #version 450
	// #extension GL_EXT_buffer_reference2 : require
	// #extension GL_EXT_shader_atomic_int64 : require
	// layout(buffer_reference, buffer_reference_align = 8) buffer Counter {
	// uint64_t v;
	// };
	// layout(push_constant) uniform PC { uint64_t addr; uint64_t value; } pc;
	// layout(local_size_x = 1) in;
	// void main() {
	// atomicMax(Counter(pc.addr).v, pc.value);
	// }
	//
	// The atomicOr variant replaces atomicMax with atomicOr.
	struct AtomicPC { uint64_t addr; uint64_t value; };
	const AtomicPC pc { .addr = addr, .value = value };

	vkCmdPushConstants(cmd.cmd,
					 cmd.caps.atomicPipelineLayout,
					 VK_SHADER_STAGE_COMPUTE_BIT,
					 /*offset=*/0, sizeof(pc), &pc);

	vkCmdDispatch(cmd.cmd, 1, 1, 1);

	// No trailing barrier here: gpuWaitBefore inserts its own. The caller is
	// expected to pair every gpuSignalAfter with a gpuWaitBefore; the barrier
	// there will sequence the atomic write before the consumer stage.
}

// ─────────────────────────────────────────────────────────────────────────────
// gpuWaitBefore
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Split-barrier consumer: stall `after` until (*ptrGpu & mask) satisfies the
 * comparison (`op`, `value`). Optionally invalidate non-coherent caches.
 *
 * Implementation paths:
 *
 * A. Timeline semaphore fast path
 * When op == OP_GREATER_EQUAL and mask == ~0 the engine may have registered
 * the buffer address as a VkSemaphoreTypeTimeline. vkWaitSemaphores on the
 * CPU stalls submission; same-queue dependency is handled via vkQueueSubmit2
 * wait-semaphore chains at the caller level. This is the zero-overhead path
 * for the canonical "monotone counter" pattern.
 *
 * B. Polling compute shader
 * A 1-thread shader spins on the counter until the condition is met, then
 * exits. The subsequent pipeline barrier from COMPUTE to `after` acts as
 * the memory fence: caches named in `hazards` are invalidated, and the
 * consumer stage is guaranteed to see the updated data.
 *
 * Shader GLSL (generic polling):
 *
 * #version 450
 * #extension GL_EXT_buffer_reference2 : require
 * #extension GL_EXT_shader_atomic_int64 : require
 * layout(buffer_reference, buffer_reference_align = 8) buffer Counter {
 * uint64_t v;
 * };
 * layout(push_constant) uniform PC {
 * uint64_t addr;
 * uint64_t value;
 * uint64_t mask;
 * int op;
 * int _pad;
 * } pc;
 * layout(local_size_x = 1) in;
 *
 * bool compare(uint64_t observed, uint64_t ref, int op) {
 * switch (op) {
 * case 0: return false; // OP_NEVER
 * case 1: return observed < ref; // OP_LESS
 * case 2: return observed == ref; // OP_EQUAL
 * case 3: return observed <= ref; // OP_LESS_EQUAL
 * case 4: return observed > ref; // OP_GREATER
 * case 5: return observed != ref; // OP_NOT_EQUAL
 * case 6: return observed >= ref; // OP_GREATER_EQUAL
 * default: return true; // OP_ALWAYS
 * }
 * }
 *
 * void main() {
 * Counter c = Counter(pc.addr);
 * uint64_t observed;
 * do {
 * observed = atomicAdd(c.v, uint64_t(0)); // atomic load
 * } while (!compare(observed & pc.mask, pc.value, pc.op));
 * }
 *
 * Note on GPU-side polling
 * Spinning in a shader is GPU-vendor-approved for this pattern (see AMD
 * "RDNA3 Synchronization Primitives" whitepaper). The thread occupies a
 * single lane on a single CU. The GPU will reorder independent work from
 * other waves while this wave polls; it does not "freeze" the whole queue.
 */
void gpuWaitBefore(GpuCommandBuffer& cmd, STAGE after, void* ptrGpu, uint64_t value, OP op, HAZARD_FLAGS hazards, uint64_t mask)
{
	const VkDeviceAddress addr = toDeviceAddr(ptrGpu);

	// ── Path A: timeline semaphore ────────────────────────────────────────────

	if (op == OP_GREATER_EQUAL && mask == ~uint64_t(0))
	{
		VkSemaphore timelineSem = cmd.caps.getTimelineSemaphore(ptrGpu);
		if (timelineSem != VK_NULL_HANDLE)
		{
			// CPU-side wait keeps things simple here; for GPU-side ordering
			// within a single queue, chain via VkSubmitInfo2 wait semaphores.
			const VkSemaphoreWaitInfo waitInfo {
				.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
				.pNext = nullptr,
				.flags = 0,
				.semaphoreCount = 1,
				.pSemaphores = &timelineSem,
				.pValues = &value,
			};
			vkWaitSemaphores(cmd.device, &waitInfo, /*timeout=*/UINT64_MAX);

			// After the CPU unblocks, insert a barrier if hazard invalidation
			// is needed. TOP_OF_PIPE → dstStage covers the cache flush.
			if (hazards != (HAZARD_FLAGS)0)
			{
				VkAccessFlags2KHR srcAccess = 0, dstAccess = 0;
				const VkPipelineStageFlags2KHR dstStage = stage2vulkan(after);
				hazard2access(hazards,
							 VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT_KHR,
							 dstStage, srcAccess, dstAccess);

				const VkMemoryBarrier2KHR mb {
					.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
					.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT_KHR,
					.srcAccessMask = srcAccess,
					.dstStageMask = dstStage,
					.dstAccessMask = dstAccess,
				};
				const VkDependencyInfoKHR dep {
					.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
					.memoryBarrierCount = 1,
					.pMemoryBarriers = &mb,
				};
				vkCmdPipelineBarrier2KHR(cmd.cmd, &dep);
			}
			return;
		}
	}

	// ── Path B: polling compute shader ───────────────────────────────────────

	VkPipeline waitPipeline = cmd.caps.getWaitPipeline(cmd.device, op);
	assert(waitPipeline != VK_NULL_HANDLE
		 && "Wait compute pipeline not compiled for this OP");

	vkCmdBindPipeline(cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, waitPipeline);

	// Push-constant layout: { uint64_t addr; uint64_t value; uint64_t mask;
	// int32_t op; int32_t _pad; }
	struct WaitPC {
		uint64_t addr;
		uint64_t value;
		uint64_t mask;
		int32_t op;
		int32_t _pad;
	};
	const WaitPC pc {
		.addr = addr,
		.value = value,
		.mask = mask,
		.op = static_cast<int32_t>(op),
		._pad = 0,
	};

	vkCmdPushConstants(cmd.cmd,
					 cmd.caps.waitPipelineLayout,
					 VK_SHADER_STAGE_COMPUTE_BIT,
					 /*offset=*/0, sizeof(pc), &pc);

	vkCmdDispatch(cmd.cmd, 1, 1, 1);

	// Barrier: COMPUTE → `after`
	//
	// This is the critical fence. Once the wait shader's dispatch completes,
	// the GPU knows the condition was satisfied. The barrier then:
	// 1. Stalls `after` until the compute stage is done.
	// 2. Flushes/invalidates any non-coherent caches named in `hazards`.
	{
		VkAccessFlags2KHR srcAccess = 0, dstAccess = 0;
		const VkPipelineStageFlags2KHR srcStage =
			VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
		const VkPipelineStageFlags2KHR dstStage = stage2vulkan(after);
		hazard2access(hazards, srcStage, dstStage, srcAccess, dstAccess);

		const VkMemoryBarrier2KHR mb {
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
			.srcStageMask = srcStage,
			.srcAccessMask = srcAccess,
			.dstStageMask = dstStage,
			.dstAccessMask = dstAccess,
		};
		const VkDependencyInfoKHR dep {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
			.memoryBarrierCount = 1,
			.pMemoryBarriers = &mb,
		};
		vkCmdPipelineBarrier2KHR(cmd.cmd, &dep);
	}
}