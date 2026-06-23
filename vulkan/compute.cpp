#include <volk.h>
#include <vk_mem_alloc.h>

#include "noapi.hpp"

#include <VkBootstrap.h>
#include <vulkan/vulkan_core.h> // TODO: Remove when it stops being auto added

struct ComputePipelinePushConstants {
	gpu* data;
	// gpu* textures; // TODO: how do we do descriptor heaps?
};

GpuPipeline gpuCreateComputePipeline(GpuQueue& queue, std::span<const std::byte> computeIR) {
	GpuPipeline out;

	if(queue.pipeline_layout == VK_NULL_HANDLE) {
		VkPushConstantRange push_constant {
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.offset = 0,
			.size = sizeof(ComputePipelinePushConstants)
		};
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_constant
		};
		VK_CHECK(vkCreatePipelineLayout(queue.device, &layout_info, queue.callbacks, &queue.pipeline_layout));
	}

	VkShaderModule compute_module;
	{
		VkShaderModuleCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = computeIR.size(),
			.pCode = (uint32_t*)computeIR.data(),
		};
		VK_CHECK(vkCreateShaderModule(queue.device, &info, queue.callbacks, &compute_module));
	}{
		VkComputePipelineCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_COMPUTE_BIT,
				.module = compute_module,
				.pName = "main"
			},
			.layout = queue.pipeline_layout
		};
		VK_CHECK(vkCreateComputePipelines(queue.device, nullptr, 1, &info, queue.callbacks, &out.pipeline));
	}

	vkDestroyShaderModule(queue.device, compute_module, queue.callbacks);
	return out;
}

void gpuDestroyPipeline(GpuQueue& queue, GpuPipeline& pipeline) {
	vkDestroyPipeline(queue.device, pipeline.pipeline, queue.callbacks);
}

inline std::pair<VkBuffer, VkDeviceSize> closest_buffer(GpuQueue& queue, gpu* addr, bool no_offsets) {
	auto address = (VkDeviceAddress)addr;
	VkDeviceAddress closest = 0; // TODO: There are probably edge cases around setting these to zero!
	if(no_offsets)
		closest = address;
	else for(auto [key, _]: queue.allocations) {
		if(closest - address > key - address)
			closest = key;
	}
	return {queue.allocations[closest].first, closest - address};
}

inline std::array<std::pair<VkBuffer, VkDeviceSize>, 2> closest_buffer(GpuQueue& queue, gpu* addrA, gpu* addrB, bool no_offsets) {
	auto a = (VkDeviceAddress)addrA, b = (VkDeviceAddress)addrB;
	VkDeviceAddress closestA = 0, closestB = 0; // TODO: There are probably edge cases around setting these to zero!
	if(no_offsets) {
		closestA = a;
		closestB = b;
	} else for(auto [key, _]: queue.allocations) {
		if(closestA - a > key - a)
			closestA = key;
		if(closestB - b > key - b)
			closestB = key;
	}
	return {
		std::pair<VkBuffer, size_t>{queue.allocations[closestA].first, closestA - a}, 
		std::pair<VkBuffer, size_t>{queue.allocations[closestB].first, closestB - b}
	};
}

void gpuMemCpy(GpuCommandBuffer& cmd, gpu* dest_, gpu* src_, size_t bytes, bool no_offsets /* = false*/) {
	auto& queue = *cmd.queue;
	auto [dest, src] = closest_buffer(*cmd.queue, dest_, src_, no_offsets);
	auto [dest_buffer, dest_offset] = dest; auto [src_buffer, src_offset] = src;

	VkBufferCopy region {
		.srcOffset = src_offset,
		.dstOffset = dest_offset,
		.size = bytes
	};
	vkCmdCopyBuffer(cmd.command_buffer, src_buffer, dest_buffer, 1, &region);
}

inline VkPipelineStageFlags2KHR stage2vulkan(STAGE stage) {
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

inline std::pair<VkAccessFlags2KHR, VkAccessFlags2KHR> hazard2access(HAZARD_FLAGS hazards) {
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

void gpuSignalAfter(GpuCommandBuffer& cmd, STAGE before, void* ptrGpu, uint64_t value, SIGNAL signal) {
	throw std::runtime_error("Not implemented yet!");
}

void gpuWaitBefore(GpuCommandBuffer& cmd, STAGE after, void* ptrGpu, uint64_t value, OP op, HAZARD_FLAGS hazards /* = (HAZARD_FLAGS)0 */, uint64_t mask /* = ~uint64_t(0) */) {
	throw std::runtime_error("Not implemented yet!");
}

void gpuSetPipeline(GpuCommandBuffer& cmd, GpuPipeline pipeline) {
	vkCmdBindPipeline(cmd.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
}

void gpuDispatch(GpuCommandBuffer& cmd, gpu* dataGpu, uvec3 gridDimensions) {
	ComputePipelinePushConstants data {
		.data = dataGpu
	};
	vkCmdPushConstants(cmd.command_buffer, cmd.queue->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePipelinePushConstants), &data);
	vkCmdDispatch(cmd.command_buffer, gridDimensions.x, gridDimensions.y, gridDimensions.z);
}

void gpuDispatchIndirect(GpuCommandBuffer& cmd, gpu* dataGpu, gpu* gridDimensionsGpu, bool no_offsets /* = false*/) {
	ComputePipelinePushConstants data {
		.data = dataGpu
	};
	vkCmdPushConstants(cmd.command_buffer, cmd.queue->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePipelinePushConstants), &data);

	auto [buffer, offset] = closest_buffer(*cmd.queue, gridDimensionsGpu, no_offsets);
	vkCmdDispatchIndirect(cmd.command_buffer, buffer, offset);
}