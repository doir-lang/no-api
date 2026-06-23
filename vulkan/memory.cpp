#include <volk.h>

#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "noapi.hpp"

#include <VkBootstrap.h>
#include <vulkan/vulkan_core.h> // TODO: Remove when it stops being auto added

void* gpuMalloc(GpuQueue& queue, size_t bytes, size_t align /* = 16 */, MEMORY memory /* = MEMORY_DEFAULT */) {
	constexpr static auto memory_to_allocation_usage = [](MEMORY memory) {
		switch (memory) {
		case MEMORY_GPU:
		case MEMORY_TEXTURE:
			return VMA_MEMORY_USAGE_GPU_ONLY;
		case MEMORY_READBACK:
		case MEMORY_TEXTURE_READBACK:
			return VMA_MEMORY_USAGE_GPU_TO_CPU;
		default:
			return VMA_MEMORY_USAGE_CPU_TO_GPU;
		}
	};

	VkBufferCreateInfo buffer_info {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = bytes,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE // TODO: Should be concurrent?
	};
	VmaAllocationCreateInfo alloc_info {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
	};
	if( !(memory == MEMORY_GPU || memory == MEMORY_TEXTURE) ) alloc_info.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
	if(memory == MEMORY_DEFAULT) alloc_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	if(memory == MEMORY_READBACK || memory == MEMORY_TEXTURE_READBACK) alloc_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	if(memory == MEMORY_TEXTURE || memory == MEMORY_TEXTURE_READBACK) alloc_info.flags |= VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT; // We can create a texture that is aliased with the buffer
	VkBuffer buffer;
	VmaAllocation allocation;
	VK_CHECK(vmaCreateBufferWithAlignment(queue.allocator, &buffer_info, &alloc_info, align, &buffer, &allocation, nullptr));

	VkBufferDeviceAddressInfo address_info {
		.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.buffer = buffer
	};
	auto gpu_ptr = vkGetBufferDeviceAddress(queue.device, &address_info);

	queue.allocations[gpu_ptr] = {buffer, allocation};

	if(memory == MEMORY_GPU || memory == MEMORY_TEXTURE)
		return (void*)gpu_ptr;

	void* cpu_ptr = allocation->GetMappedData();
	queue.host2gpu[cpu_ptr] = gpu_ptr;
	queue.gpu2host[gpu_ptr] = cpu_ptr;
	return cpu_ptr;
}

void gpuFree(GpuQueue& queue, void* ptr) {
	if(queue.host2gpu.contains(ptr)) {
		gpuFree(queue, (gpu*)queue.host2gpu[ptr]);
	}
}
void gpuFree(GpuQueue& queue, gpu* ptr) {
	auto gpu_ptr = (VkDeviceAddress)ptr;
	if(!queue.allocations.contains(gpu_ptr)) return;

	auto [buffer, allocation] = queue.allocations[gpu_ptr];
	queue.allocations.erase(gpu_ptr);

	if(queue.gpu2host.contains(gpu_ptr)) {
		auto host = queue.gpu2host[gpu_ptr];
		queue.gpu2host.erase(gpu_ptr);
		queue.host2gpu.erase(host);
	}

	vmaDestroyBuffer(queue.allocator, buffer, allocation);
}

gpu* gpuHostToDevicePointer(GpuQueue& queue, void* ptr) {
	if(queue.host2gpu.contains(ptr))
		return (gpu*)queue.host2gpu[ptr];
	return nullptr;
}

void* gpuDeviceToHostPointer(GpuQueue& queue, gpu* ptr) {
	auto gpu_ptr = (VkDeviceAddress)ptr;
	if(queue.gpu2host.contains(gpu_ptr))
		return queue.gpu2host[gpu_ptr];
	return nullptr;
}
