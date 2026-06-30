#include <volk.h>
#include <vk_mem_alloc.h>

#include "noapi.hpp"

#include <VkBootstrap.h>
#include <vulkan/vulkan_core.h> // TODO: Remove when it stops being auto added

const GpuSemaphore* gpuGetSubmissionTimelineSemaphore(GpuQueue* queue) {
	return (GpuSemaphore*)&queue->command_submission_timeline_semaphore;
}

void gpuWaitIdle(GpuQueue* queue) {
	vkDeviceWaitIdle(queue->device);
}

struct ComputePipelinePushConstants {
	gpu* data;
};

GpuPipeline* gpuCreateComputePipeline(GpuQueue* queue, std::span<const std::byte> computeIR) {
	auto out = (GpuPipeline*)queue->cpu_allocator(nullptr, sizeof(GpuPipeline));

	VkShaderModule compute_module;
	{
		VkShaderModuleCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = computeIR.size(),
			.pCode = (uint32_t*)computeIR.data(),
		};
		VK_CHECK(vkCreateShaderModule(queue->device, &info, queue->callbacks, &compute_module), nullptr);
	}{
		VkPipelineCreateFlags2CreateInfo create_flags {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
			.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT
		};
		VkComputePipelineCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.pNext = &create_flags,
			.stage = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_COMPUTE_BIT,
				.module = compute_module,
				.pName = "main"
			},
		};
		VK_CHECK(vkCreateComputePipelines(queue->device, nullptr, 1, &info, queue->callbacks, &out->pipeline), nullptr);
	}

	vkDestroyShaderModule(queue->device, compute_module, queue->callbacks);
	return out;
}

void gpuDestroyPipeline(GpuQueue* queue, GpuPipeline* pipeline) {
	vkDestroyPipeline(queue->device, pipeline->pipeline, queue->callbacks);
	queue->cpu_allocator(pipeline, 0);
}

inline std::tuple<VkBuffer, VkDeviceSize, VkDeviceAddress> closest_buffer(GpuQueue* queue, gpu* addr, bool no_offsets) {
	auto address = (VkDeviceAddress)addr;
	VkDeviceAddress closest = 0; // TODO: There are probably edge cases around setting these to zero!
	if(no_offsets)
		closest = address;
	else for(auto [key, _]: queue->allocations) {
		if(closest - address > key - address)
			closest = key;
	}
	return {std::get<VkBuffer>(queue->allocations[closest]), closest - address, address};
}

inline std::array<std::tuple<VkBuffer, VkDeviceSize, VkDeviceAddress>, 2> closest_buffer(GpuQueue* queue, gpu* addrA, gpu* addrB, bool no_offsets) {
	auto a = (VkDeviceAddress)addrA, b = (VkDeviceAddress)addrB;
	VkDeviceAddress closestA = 0, closestB = 0; // TODO: There are probably edge cases around setting these to zero!
	if(no_offsets) {
		closestA = a;
		closestB = b;
	} else for(auto [key, _]: queue->allocations) {
		if(closestA - a > key - a)
			closestA = key;
		if(closestB - b > key - b)
			closestB = key;
	}
	return {
		std::tuple<VkBuffer, VkDeviceSize, VkDeviceAddress>{std::get<VkBuffer>(queue->allocations[closestA]), closestA - a, a}, 
		std::tuple<VkBuffer, VkDeviceSize, VkDeviceAddress>{std::get<VkBuffer>(queue->allocations[closestB]), closestB - b, b}
	};
}

void gpuMemCpy(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, size_t bytes, bool no_offsets /* = false*/) {
	auto [dest, src] = closest_buffer(cmd->queue, dest_, src_, no_offsets);
	auto [dest_buffer, dest_offset, dest_addr] = dest; auto [src_buffer, src_offset, src_addr] = src;

	VkBufferCopy region {
		.srcOffset = src_offset,
		.dstOffset = dest_offset,
		.size = bytes
	};
	vkCmdCopyBuffer(cmd->command_buffer, src_buffer, dest_buffer, 1, &region);
}

void gpuCopyToTexture(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, GpuTexture* texture, bool no_offsets /* = false */) {
	auto [dest, src] = closest_buffer(cmd->queue, dest_, src_, no_offsets);
	auto [dest_buffer, dest_offset, dest_addr] = dest; auto [src_buffer, src_offset, src_addr] = src;

	assert(cmd->queue->gpu2image.contains(dest_addr) && cmd->queue->gpu2image[dest_addr] == texture->image);

	assert(dest_offset == 0); // TODO: Can we relax these restrictions?
	assert(src_offset == 0);

	// TODO: DO we need a barrier?

	VkBufferImageCopy copy {
		.bufferOffset = src_offset,
		.bufferRowLength = static_cast<uint32_t>(std::get<VkDeviceSize>(cmd->queue->allocations[src_addr]) / texture->descriptor.dimensions.y),
		.bufferImageHeight = texture->descriptor.dimensions.y,
		.imageSubresource = {
			.aspectMask = static_cast<VkImageAspectFlags>(gpuFormatIsDepth(texture->descriptor.format) 
				? VK_IMAGE_ASPECT_DEPTH_BIT | (gpuFormatIsStencil(texture->descriptor.format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)
				: VK_IMAGE_ASPECT_COLOR_BIT),
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
		.imageOffset = {0, 0, 0},
		.imageExtent = {texture->descriptor.dimensions.x, texture->descriptor.dimensions.y, texture->descriptor.dimensions.z}
	};
	vkCmdCopyBufferToImage(cmd->command_buffer, src_buffer, texture->image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
}

void gpuCopyFromTexture(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, const GpuTexture* texture, bool no_offsets /* = false */) {
	auto [dest, src] = closest_buffer(cmd->queue, dest_, src_, no_offsets);
	auto [dest_buffer, dest_offset, dest_addr] = dest; auto [src_buffer, src_offset, src_addr] = src;

	assert(cmd->queue->gpu2image.contains(src_addr) && cmd->queue->gpu2image[src_addr] == texture->image);

	assert(dest_offset == 0); // TODO: Can we relax these restrictions?
	assert(src_offset == 0);

	// TODO: DO we need a barrier?

	VkBufferImageCopy copy {
		.bufferOffset = src_offset,
		.bufferRowLength = static_cast<uint32_t>(std::get<VkDeviceSize>(cmd->queue->allocations[dest_addr]) / texture->descriptor.dimensions.y),
		.bufferImageHeight = texture->descriptor.dimensions.y,
		.imageSubresource = {
			.aspectMask = static_cast<VkImageAspectFlags>(gpuFormatIsDepth(texture->descriptor.format) 
				? VK_IMAGE_ASPECT_DEPTH_BIT | (gpuFormatIsStencil(texture->descriptor.format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)
				: VK_IMAGE_ASPECT_COLOR_BIT),
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
		.imageOffset = {0, 0, 0},
		.imageExtent = {texture->descriptor.dimensions.x, texture->descriptor.dimensions.y, texture->descriptor.dimensions.z}
	};
	vkCmdCopyImageToBuffer(cmd->command_buffer, texture->image, VK_IMAGE_LAYOUT_GENERAL, dest_buffer, 1, &copy);
}

void gpuSetActiveTextureHeapPtr(GpuCommandBuffer* cmd, gpu* texture_heap, bool no_offsets /* = false */) {
	auto [source_buffer, offset, source_address] = closest_buffer(cmd->queue, texture_heap, no_offsets);
	auto source_size = std::get<VkDeviceSize>(cmd->queue->allocations[source_address]);
	
	if(!cmd->queue->descriptor_heaps.contains(source_address)) {
		auto size = std::max<VkDeviceSize>(source_size, cmd->queue->minimum_descriptor_heap_size);
		VkBufferCreateInfo buffer_info {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		};
		VmaAllocationCreateInfo alloc_info {
			.usage = VMA_MEMORY_USAGE_AUTO,
			.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
		};

		auto& [buffer, allocation, heap_size, heap_address] = cmd->queue->descriptor_heaps[source_address];
		heap_size = size;
		VK_CHECK(vmaCreateBuffer(cmd->queue->gpu_allocator, &buffer_info, &alloc_info, &buffer, &allocation, nullptr), /*nothing*/);

		VkBufferDeviceAddressInfo address_info {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = buffer
		};
		heap_address = vkGetBufferDeviceAddress(cmd->queue->device, &address_info);
	}

	auto [heap_buffer, _allocation, heap_size, heap_address] = cmd->queue->descriptor_heaps[source_address];
	VkBufferCopy copy {
		.srcOffset = offset,
		.dstOffset = offset,
		.size = source_size - offset
	};
	vkCmdCopyBuffer(cmd->command_buffer, source_buffer, heap_buffer, 1, &copy);

	// TODO: Do we need a barrier here?

	VkBindHeapInfoEXT heap_info {
		.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
		.heapRange = {
			.address = heap_address + offset,
			.size = heap_size - offset
		},
		.reservedRangeSize = cmd->queue->minimum_descriptor_heap_size
	};
	vkCmdBindResourceHeapEXT(cmd->command_buffer, &heap_info);
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

void gpuBarrier(GpuCommandBuffer* cmd, STAGE before, STAGE after, HAZARD_FLAGS hazards) {
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
	vkCmdPipelineBarrier2KHR(cmd->command_buffer, &dependency);
}

void gpuSignalAfter(GpuCommandBuffer* cmd, STAGE before, void* ptrGpu, uint64_t value, SIGNAL signal) {
	throw std::runtime_error("Not implemented yet!");
}

void gpuWaitBefore(GpuCommandBuffer* cmd, STAGE after, void* ptrGpu, uint64_t value, OP op, HAZARD_FLAGS hazards /* = (HAZARD_FLAGS)0 */, uint64_t mask /* = ~uint64_t(0) */) {
	throw std::runtime_error("Not implemented yet!");
}

void gpuSetPipeline(GpuCommandBuffer* cmd, const GpuPipeline* pipeline) {
	vkCmdBindPipeline(cmd->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
}

void gpuDispatch(GpuCommandBuffer* cmd, gpu* dataGpu, uvec3 gridDimensions) {
	ComputePipelinePushConstants data {
		.data = dataGpu
	};
	VkPushDataInfoEXT info {
		.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
		.offset = 0,
		.data = {
			.address = &data,
			.size = sizeof(ComputePipelinePushConstants)
		}
	};
	vkCmdPushDataEXT(cmd->command_buffer, &info);
	vkCmdDispatch(cmd->command_buffer, gridDimensions.x, gridDimensions.y, gridDimensions.z);
}

void gpuDispatchIndirect(GpuCommandBuffer* cmd, gpu* dataGpu, gpu* gridDimensionsGpu, bool no_offsets /* = false*/) {
	ComputePipelinePushConstants data {
		.data = dataGpu
	};
	VkPushDataInfoEXT info {
		.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
		.offset = 0,
		.data = {
			.address = &data,
			.size = sizeof(ComputePipelinePushConstants)
		}
	};
	vkCmdPushDataEXT(cmd->command_buffer, &info);

	auto [buffer, offset, _address] = closest_buffer(cmd->queue, gridDimensionsGpu, no_offsets);
	vkCmdDispatchIndirect(cmd->command_buffer, buffer, offset);
}