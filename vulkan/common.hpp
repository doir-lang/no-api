#pragma once

#include "noapi.hpp"
#include <vulkan/vulkan_core.h>

#include <utility>

namespace GPU::detail {
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

	inline VkImageUsageFlags usage2vulkan(TEXTURE_USAGE_FLAGS usageFlags) {
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

	inline VkSampleCountFlagBits samples2vulkan(size_t samples) {
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

	inline VkImageViewType type2vulkan(TEXTURE type) {
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
}