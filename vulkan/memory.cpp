#include <volk.h>

#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "noapi.hpp"

#include <VkBootstrap.h>
#include <vulkan/vulkan_core.h> // TODO: Remove when it stops being auto added

void* gpuMalloc(GpuQueue* queue, size_t bytes, size_t align /* = 16 */, MEMORY memory /* = MEMORY_DEFAULT */) {
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
	VK_CHECK(vmaCreateBufferWithAlignment(queue->gpu_allocator, &buffer_info, &alloc_info, align, &buffer, &allocation, nullptr), nullptr);

	VkBufferDeviceAddressInfo address_info {
		.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.buffer = buffer
	};
	auto gpu_ptr = vkGetBufferDeviceAddress(queue->device, &address_info);

	queue->allocations[gpu_ptr] = {buffer, allocation, bytes};

	if(memory == MEMORY_GPU || memory == MEMORY_TEXTURE)
		return (void*)gpu_ptr;

	void* cpu_ptr = allocation->GetMappedData();
	queue->host2gpu[cpu_ptr] = gpu_ptr;
	queue->gpu2host[gpu_ptr] = cpu_ptr;
	return cpu_ptr;
}

void gpuFree(GpuQueue* queue, void* ptr) {
	if(queue->host2gpu.contains(ptr)) {
		gpuFree(queue, (gpu*)queue->host2gpu[ptr]);
	}
}
void gpuFree(GpuQueue* queue, gpu* ptr) {
	auto gpu_ptr = (VkDeviceAddress)ptr;
	if(!queue->allocations.contains(gpu_ptr)) return;

	auto [buffer, allocation, _size] = queue->allocations[gpu_ptr];
	queue->allocations.erase(gpu_ptr);

	if(queue->gpu2host.contains(gpu_ptr)) {
		auto host = queue->gpu2host[gpu_ptr];
		queue->gpu2host.erase(gpu_ptr);
		queue->host2gpu.erase(host);
	}

	if(queue->gpu2image.contains(gpu_ptr)) {
		auto image = queue->gpu2image[gpu_ptr];
		vkDestroyImage(queue->device, image, queue->callbacks);
		queue->gpu2image.erase(gpu_ptr);
	}

	if(queue->descriptor_heaps.contains(gpu_ptr)) {
		auto [buffer, allocation, _size, _address] = queue->descriptor_heaps[gpu_ptr];
		vmaDestroyBuffer(queue->gpu_allocator, buffer, allocation);
	}

	vmaDestroyBuffer(queue->gpu_allocator, buffer, allocation);
}

gpu* gpuHostToDevicePointer(GpuQueue* queue, void* ptr) {
	if(queue->host2gpu.contains(ptr))
		return (gpu*)queue->host2gpu[ptr];
	return nullptr;
}

void* gpuDeviceToHostPointer(GpuQueue* queue, gpu* ptr) {
	auto gpu_ptr = (VkDeviceAddress)ptr;
	if(queue->gpu2host.contains(gpu_ptr))
		return queue->gpu2host[gpu_ptr];
	return nullptr;
}

inline VkFormat format2vulkan(FORMAT format) {
	switch(format) {
	case FORMAT_NONE:
		return VK_FORMAT_UNDEFINED;
	case FORMAT_RGBA8_UNORM:
		return VK_FORMAT_R8G8B8A8_UNORM;
	case FORMAT_RGBA8_SRGB:
		return VK_FORMAT_R8G8B8A8_SRGB;
	case FORMAT_RGBA16_FLOAT:
		return VK_FORMAT_R16G16B16A16_SFLOAT;
	case FORMAT_RGBA32_FLOAT:
		return VK_FORMAT_R32G32B32A32_SFLOAT;
	case FORMAT_RG11B10_FLOAT:
		return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
	case FORMAT_RGB10_A2_UNORM:
		return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
	case FORMAT_R8_UNORM:
		return VK_FORMAT_R8_UNORM;
	case FORMAT_R16_FLOAT:
		return VK_FORMAT_R16_SFLOAT;
	case FORMAT_R32_FLOAT:
		return VK_FORMAT_R32_SFLOAT;
	case FORMAT_D16_UNORM:
		return VK_FORMAT_D16_UNORM;
	case FORMAT_D24_UNORM_S8_UINT:
		return VK_FORMAT_D24_UNORM_S8_UINT;
	case FORMAT_D32_FLOAT:
		return VK_FORMAT_D32_SFLOAT;
	case FORMAT_D32_FLOAT_S8_UINT:
		return VK_FORMAT_D32_SFLOAT_S8_UINT;
	}
	std::unreachable();
};

VkImageCreateInfo descriptor2vulkan(const GpuTextureDesc& descriptor) {
	constexpr static auto type2vulkan = [](TEXTURE type) {
		switch(type) {
		case TEXTURE_1D:
			return VK_IMAGE_TYPE_1D;
		case TEXTURE_2D:
		case TEXTURE_2D_ARRAY:
		case TEXTURE_CUBE:
		case TEXTURE_CUBE_ARRAY:
			return VK_IMAGE_TYPE_2D;
		case TEXTURE_3D:
			return VK_IMAGE_TYPE_3D;
		}
		std::unreachable();
	};
	constexpr static auto samples2vulkan = [](size_t samples) {
		if(samples < 2)
			return VK_SAMPLE_COUNT_1_BIT;
		else if(samples < 4)
			return VK_SAMPLE_COUNT_2_BIT;
		else if(samples < 8)
			return VK_SAMPLE_COUNT_4_BIT;
		else if(samples < 16)
			return VK_SAMPLE_COUNT_8_BIT;
		else if(samples < 32)
			return VK_SAMPLE_COUNT_16_BIT;
		else if(samples < 64)
			return VK_SAMPLE_COUNT_32_BIT;
		else return VK_SAMPLE_COUNT_64_BIT;
	};
	constexpr static auto usage2vulkan = [](TEXTURE_USAGE_FLAGS usageFlags) {
		VkImageUsageFlags result = 0;
		if (usageFlags & USAGE_SAMPLED)
			result |= VK_IMAGE_USAGE_SAMPLED_BIT;
		if (usageFlags & USAGE_STORAGE)
			result |= VK_IMAGE_USAGE_STORAGE_BIT;
		if (usageFlags & USAGE_COLOR_ATTACHMENT)
			result |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		if (usageFlags & USAGE_DEPTH_STENCIL_ATTACHMENT)
			result |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		if (usageFlags & USAGE_TRANSFER_SRC)
			result |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		if (usageFlags & USAGE_TRANSFER_DST)
			result |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		return result;
	};

	return VkImageCreateInfo{
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.flags = VK_IMAGE_CREATE_ALIAS_BIT,
		.imageType = type2vulkan(descriptor.type),
		.format = format2vulkan(descriptor.format),
		.extent = VkExtent3D{descriptor.dimensions.x, descriptor.dimensions.y, descriptor.dimensions.z},
		.mipLevels = descriptor.mipCount,
		.arrayLayers = descriptor.layerCount,
		.samples = samples2vulkan(descriptor.sampleCount),
		.tiling = VK_IMAGE_TILING_LINEAR,
		.usage = usage2vulkan(descriptor.usage),
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
}

GpuTextureSizeAlign gpuTextureSizeAlign(GpuQueue* queue, const GpuTextureDesc& desc) {
	auto info = descriptor2vulkan(desc);
	VkImage temp;
	VK_CHECK(vkCreateImage(queue->device, &info, queue->callbacks, &temp), {});

	VkMemoryRequirements requirements;
	vkGetImageMemoryRequirements(queue->device, temp, &requirements);
	vkDestroyImage(queue->device, temp, queue->callbacks);

	return {requirements.size, requirements.alignment};
}

GpuTexture* gpuCreateTexture(GpuQueue* queue, const GpuTextureDesc& desc, gpu* memory) {
	auto out = (GpuTexture*)queue->cpu_allocator(nullptr, sizeof(GpuTexture));
	*out = {.descriptor = desc};

	auto info = descriptor2vulkan(desc);
	VK_CHECK(vkCreateImage(queue->device, &info, queue->callbacks, &out->image), nullptr);
	VK_CHECK(vkBindImageMemory(queue->device, out->image, std::get<VmaAllocation>(queue->allocations[(VkDeviceAddress)memory])->GetMemory(), 0), nullptr);

	queue->gpu2image[(VkDeviceAddress)memory] = out->image;
	return out;
}

inline GpuTextureDescriptor gpuTextureViewDescriptorImpl(GpuQueue* queue, const GpuTexture* texture, const GpuViewDesc& desc, bool read_only) {
	constexpr static auto type2vulkan = [](TEXTURE type) {
		switch (type) {
		case TEXTURE_1D: return VK_IMAGE_VIEW_TYPE_1D;
		case TEXTURE_2D: return VK_IMAGE_VIEW_TYPE_2D;
		case TEXTURE_3D: return VK_IMAGE_VIEW_TYPE_3D;
		case TEXTURE_2D_ARRAY: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
		case TEXTURE_CUBE: return VK_IMAGE_VIEW_TYPE_CUBE;
		case TEXTURE_CUBE_ARRAY: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
		}
		std::unreachable();
	};

	GpuTextureDescriptor out = {};
	VkHostAddressRangeEXT host_info {
		.address = &out,
		.size = sizeof(GpuTextureDescriptor)
	};

	auto format = desc.format == FORMAT_NONE ? texture->descriptor.format : desc.format;
	VkImageViewCreateInfo view_info {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = texture->image,
		.viewType = type2vulkan(texture->descriptor.type),
		.format = format2vulkan(format),
		.subresourceRange = {
			.aspectMask = static_cast<VkImageAspectFlags>(gpuFormatIsDepth(format) 
				? VK_IMAGE_ASPECT_DEPTH_BIT | (gpuFormatIsStencil(format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)
				: VK_IMAGE_ASPECT_COLOR_BIT),
			.baseMipLevel = desc.baseMip,
			.levelCount = desc.mipCount == ALL_MIPS ? texture->descriptor.mipCount - desc.baseMip : desc.mipCount,
			.baseArrayLayer = desc.baseLayer,
			.layerCount = desc.layerCount == ALL_LAYERS ? texture->descriptor.layerCount - desc.baseLayer : desc.layerCount,
		}
	};
	VkImageDescriptorInfoEXT image_info {
		.sType = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
		.pView = &view_info,
		.layout = VK_IMAGE_LAYOUT_GENERAL
	};
	VkResourceDescriptorInfoEXT descriptor {
		.sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
		.type = read_only ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.data = {
			.pImage = &image_info
		},
	};
	VK_CHECK(vkWriteResourceDescriptorsEXT(queue->device, 1, &descriptor, &host_info), {});

	return out;
}

GpuTextureDescriptor gpuTextureViewDescriptor(GpuQueue* queue, const GpuTexture* texture, const GpuViewDesc& desc) {
	return gpuTextureViewDescriptorImpl(queue, texture, desc, true);
}

GpuTextureDescriptor gpuRWTextureViewDescriptor(GpuQueue* queue, const GpuTexture* texture, const GpuViewDesc& desc) {
	return gpuTextureViewDescriptorImpl(queue, texture, desc, false);
}

