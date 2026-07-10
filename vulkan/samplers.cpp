#include <cstdint>
#include <memory>
#include <set>
#include <sys/types.h>
#include <utility>
#include <vector>
#include <volk.h>
#include <vk_mem_alloc.h>

#include "compute.hpp"
#include "noapi.hpp"
#include "common.hpp"

#include <VkBootstrap.h>
#include <vulkan/vulkan_core.h> // TODO: Remove when it stops being auto added"

void gpuSetEnabledSamplersEXT(GpuCommandBuffer* cmd, std::span<GpuSamplerDesc> enabled_samplers_) {
	constexpr static auto address2vulkan = [](ADDRESS_MODE mode) {
		switch (mode) {
		case ADDRESS_MODE_CLAMP: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		case ADDRESS_MODE_MIRROR_REPEAT: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
		case ADDRESS_MODE_REPEAT: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
		}
		std::unreachable();
	};
	constexpr static auto filter2vulkan = [](FILTER mode) {
		switch (mode) {
		case FILTER_NEAREST: return VK_FILTER_NEAREST;
		case FILTER_LINEAR: return VK_FILTER_LINEAR;
		}
		std::unreachable();
	};
	constexpr static auto filter2vulkan_mip = [](FILTER mode) {
		switch (mode) {
		case FILTER_NEAREST: return VK_SAMPLER_MIPMAP_MODE_NEAREST;
		case FILTER_LINEAR: return VK_SAMPLER_MIPMAP_MODE_LINEAR;
		}
		std::unreachable();
	};

	std::vector<GpuSamplerDesc> enabled_samplers(enabled_samplers_.size() + 1, GpuSamplerDesc{});
	std::move(enabled_samplers_.begin(), enabled_samplers_.end(), enabled_samplers.begin() + 1); // +1 means that the default sampler is always enabled

	if(!cmd->queue->sampler_cache.contains(enabled_samplers)) {
		auto& sampler_mapping = cmd->queue->sampler_cache[enabled_samplers];

		uint32_t* sampler_mapping_cpu = gpuMalloc<uint32_t>(cmd->queue, GpuSamplerDesc::max_packed());
		std::memset(sampler_mapping_cpu, 0, GpuSamplerDesc::max_packed() * sizeof(uint32_t));
		for(size_t i = 0; i < enabled_samplers.size(); ++i)
			sampler_mapping_cpu[enabled_samplers[i].pack()] = i;
		sampler_mapping = (VkDeviceAddress)gpuHostToDevicePointer(cmd->queue, sampler_mapping_cpu);

		auto size = std::max<VkDeviceSize>(cmd->queue->sampler_size * enabled_samplers.size(), cmd->queue->minimum_descriptor_heap_size);
		auto tmp = gpuMalloc(cmd->queue, size);

		std::vector<VkSamplerCreateInfo> sampler_infos; sampler_infos.reserve(enabled_samplers.size());
		for(size_t i = 0; i < enabled_samplers.size(); ++i)
			sampler_infos.emplace_back(VkSamplerCreateInfo{
				.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
				.magFilter = filter2vulkan(enabled_samplers[i].mag_filter),
				.minFilter = filter2vulkan(enabled_samplers[i].min_filter),
				.mipmapMode = filter2vulkan_mip(enabled_samplers[i].mip_filter),
				.addressModeU = address2vulkan(enabled_samplers[i].address_mode_u),
				.addressModeV = address2vulkan(enabled_samplers[i].address_mode_v),
				.addressModeW = address2vulkan(enabled_samplers[i].address_mode_w),
				.maxLod = VK_LOD_CLAMP_NONE
			});
		VkHostAddressRangeEXT host_info {
			.address = tmp,
			.size = size
		};
		vkWriteSamplerDescriptorsEXT(cmd->queue->device, sampler_infos.size(), sampler_infos.data(), &host_info);

		VkBufferCreateInfo buffer_info {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		};
		VmaAllocationCreateInfo alloc_info {
			.usage = VMA_MEMORY_USAGE_AUTO,
			.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
		};

		auto& [buffer, allocation, heap_size, heap_address] = cmd->queue->descriptor_heaps[sampler_mapping];
		heap_size = size;
		VK_CHECK(vmaCreateBuffer(cmd->queue->gpu_allocator, &buffer_info, &alloc_info, &buffer, &allocation, nullptr), /*nothing*/);

		VkBufferDeviceAddressInfo address_info {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = buffer
		};
		heap_address = vkGetBufferDeviceAddress(cmd->queue->device, &address_info);

		auto copyCMD = gpuStartCommandRecording(cmd->queue);
		VkBufferCopy region {
			.size = size
		};
		vkCmdCopyBuffer(copyCMD->command_buffer, std::get<VkBuffer>(cmd->queue->allocations[(VkDeviceAddress)gpuHostToDevicePointer(cmd->queue, tmp)]), buffer, 1, &region);
		gpuWaitSemaphore(cmd->queue, gpuGetSubmissionSemaphoreEXT(cmd->queue), 
			gpuSubmit(cmd->queue, {&copyCMD, 1})
		);
		gpuFree(cmd->queue, tmp);
	}
		
	cmd->sampler_map = cmd->queue->sampler_cache[enabled_samplers];
	auto& [_buffer, _allocation, heap_size, heap_address] = cmd->queue->descriptor_heaps[cmd->sampler_map];
	VkBindHeapInfoEXT heap_info {
		.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
		.heapRange = {
			.address = heap_address,
			.size = heap_size
		},
		.reservedRangeSize = cmd->queue->minimum_descriptor_heap_size
	};
	vkCmdBindSamplerHeapEXT(cmd->command_buffer, &heap_info);
}