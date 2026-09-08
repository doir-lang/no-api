#define VOLK_IMPLEMENTATION
#include <volk.h>

#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "noapi.hpp"

#include <VkBootstrap.h>
#include <vulkan/vulkan_core.h> // TODO: Remove when it stops being auto added

#include <cassert>
#include <cerrno>
#include <cstdint>
#include <memory>
#include <set>
#include <sys/types.h>
#include <utility>
#include <vector>




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




thread_local static VkDebugUtilsMessageSeverityFlagBitsEXT severity_filter;

void GPU::default_::error_callback(void* queue, int type, std::string_view message) {
	auto mt = vkb::to_string_message_type(type);
	printf("[%s]\n%s\n", mt, message.data());
}

std::expected<GpuVulkanDefault, std::string> gpuSetupDefaultVulkanEXT(
	GPU::function_t<VkSurfaceKHR(VkInstance)> surface_loader, void(*error_callback)(void* queue, int type, std::string_view message) /* = GPU::default_::error_callback */, VkDebugUtilsMessageSeverityFlagBitsEXT severity_filter_set /* = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT */,
	std::span<const char*> instance_extensions /* = {} */, std::span<const char*> extra_layers /* = {} */, std::span<const char*> device_extensions /* = {} */, bool debug /* = true */
) {
	GpuVulkanDefault out;
	severity_filter = severity_filter_set;

	// Instance
	vkb::InstanceBuilder instance_builder;
	instance_builder.set_app_name("NoAPI")
		.set_engine_name("NoAPI")
		.enable_extensions(instance_extensions.size(), instance_extensions.data())
		.request_validation_layers(debug)
		.require_api_version(1, 4, 0);
	for(auto layer: extra_layers)
		instance_builder.enable_layer(layer);
	if(debug) instance_builder.set_debug_callback(+[](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageTypes, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) -> VkBool32 {
			if(messageSeverity < severity_filter) return VK_FALSE;

			auto error_callback = (void(*)(void* queue, int type, std::string_view message))pUserData;
			error_callback(nullptr, messageTypes, {pCallbackData->pMessage, strlen(pCallbackData->pMessage)});
			return VK_FALSE;
		}).set_debug_callback_user_data_pointer((void*)error_callback);//.use_default_debug_messenger();
	auto inst = instance_builder.build();
	if (!inst) return std::unexpected(inst.error().message());
	auto instance = inst.value();
	out.instance = instance.instance;
	out.messenger = instance.debug_messenger;

	out.surface = surface_loader(out.instance);

	auto extensions = gpuRequiredVulkanDeviceExtensionsEXT();
	extensions.insert(extensions.end(), device_extensions.begin(), device_extensions.end());

	// Physical Device
	vkb::PhysicalDeviceSelector gpu_selector{instance};
	auto phys = gpu_selector
		.set_required_features(gpuEnableRequiredVulkanFeaturesEXT({}))
		.set_required_features_12(gpuEnableRequiredVulkan12FeaturesEXT({}))
		.set_required_features_13(gpuEnableRequiredVulkan13FeaturesEXT({}))
		.set_required_features_14(gpuEnableRequiredVulkan14FeaturesEXT({}))
		.add_required_extensions(extensions)
		.set_surface(out.surface)
		.set_minimum_version(1, 4)
		.select();
	if (!phys) return std::unexpected(phys.error().message());
	auto gpu = phys.value();
	out.gpu = gpu.physical_device;

	// Logical Device
	vkb::DeviceBuilder device_builder{gpu};
	auto dev = device_builder.add_pNext(gpuRequiredVulkanDeviceCreateInfoPnextEXT()).build();
	if (!dev) return std::unexpected(dev.error().message());
	auto device = dev.value();
	out.device = device.device;

	out.graphics_queue = device.get_queue(vkb::QueueType::graphics).value();
	out.graphics_queue_family = device.get_queue_index(vkb::QueueType::graphics).value();

	return out;
}



std::optional<GpuSemaphore> gpuCreateSemaphoreImpl(GpuQueue* queue, uint64_t init_value);

GpuQueue* gpuCreateQueue(VkInstance instance, VkPhysicalDevice gpu, VkDevice device, VkQueue queue /* = VK_NULL_HANDLE */, uint32_t queue_family /* = -1 */, bool is_graphics_queue /* = true */, CpuAllocatorFunc allocator /* = default_::gpu_allocator */, VkAllocationCallbacks* callbacks /* = nullptr */) {
	auto out = (GpuQueue*)allocator(nullptr, sizeof(GpuQueue));
	new(out) GpuQueue {
		.cpu_allocator = allocator,
		.gpu = gpu,
		.device = device,
		.queue = queue,
		.queue_family = queue_family,
		.is_graphics_queue = is_graphics_queue,
		.callbacks = callbacks
	};

	VK_CHECK(volkInitialize(), nullptr);
	volkLoadInstance(instance);
	volkLoadDevice(out->device);

	// Find queue if one wasn't already provided
	if(!out->queue) {
		uint32_t count;
		vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
		std::vector<VkQueueFamilyProperties> families(count);
		vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, families.data());

		vkb::Device device;
		device.device = out->device;
		device.queue_families = std::move(families);

		if(auto r = device.get_queue_index(is_graphics_queue ? vkb::QueueType::graphics : vkb::QueueType::compute); r.has_value())
			out->queue_family = r.value();
		else return {};
		vkGetDeviceQueue(out->device, out->queue_family, 0, &out->queue);
	}

	// VMA
	VmaVulkanFunctions functions = {
		.vkGetInstanceProcAddr = vkGetInstanceProcAddr,
		.vkGetDeviceProcAddr = vkGetDeviceProcAddr,
	};

	VmaAllocatorCreateInfo vma_info = {
		.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
		.physicalDevice = out->gpu,
		.device = out->device,
		.pAllocationCallbacks = out->callbacks,
		.pVulkanFunctions = &functions,
		.instance = instance,
		.vulkanApiVersion = VK_API_VERSION_1_3,
	};
	if(vmaCreateAllocator(&vma_info, &out->gpu_allocator) != VK_SUCCESS)
		return {};

	out->command_submission_timeline_semaphore = gpuCreateSemaphoreImpl(out, 0)->semaphore;

	VkPhysicalDeviceDescriptorHeapPropertiesEXT heap_properties {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT,
	};
	VkPhysicalDeviceProperties2 properties {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &heap_properties
	};
	vkGetPhysicalDeviceProperties2(gpu, &properties);
	out->minimum_descriptor_heap_size = heap_properties.minResourceHeapReservedRange;
	out->sampler_size = heap_properties.samplerDescriptorSize;

	return out;
}

void gpuFreeQueue(GpuQueue* queue) {
	if(queue->command_pool)
		vkDestroyCommandPool(queue->device, queue->command_pool, queue->callbacks);
	if(queue->command_submission_timeline_semaphore)
		vkDestroySemaphore(queue->device, queue->command_submission_timeline_semaphore, queue->callbacks);

	for(auto [_, buffer]: queue->sampler_cache)
		gpuFree(queue, (gpu*)buffer);

	if(queue->gpu_allocator)
		vmaDestroyAllocator(queue->gpu_allocator);

	auto allocator = queue->cpu_allocator;
	queue->~GpuQueue(); // Get all of the caches to free their memory
	allocator(queue, 0);
}




GpuCommandBuffer* gpuStartCommandRecording(GpuQueue* queue) {
	if(!queue->command_pool) {
		VkCommandPoolCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
			.queueFamilyIndex = queue->queue_family
		};
		VK_CHECK(vkCreateCommandPool(queue->device, &info, queue->callbacks, &queue->command_pool), nullptr);
	}

	// Check the current value of the submission count timeline semaphore and free any buffers who's submission has finished
	if(!queue->command_buffers_pending_free.empty()) {
		uint64_t current_finished_submission;
		vkGetSemaphoreCounterValue(queue->device, queue->command_submission_timeline_semaphore, &current_finished_submission);

		// TODO: Would it be worth the effort to deduplicate?

		std::vector<VkCommandBuffer> to_free; to_free.reserve(queue->command_buffers_pending_free.size());
		if(queue->command_buffers_pending_free.size())
			for(size_t i = queue->command_buffers_pending_free.size(); i--; ) {
				auto [cmd, submit] = queue->command_buffers_pending_free[i];
				if(submit <= current_finished_submission) {
					to_free.push_back(cmd);
					queue->command_buffers_pending_free.erase(queue->command_buffers_pending_free.begin() + i);
				}
			}

		if(!to_free.empty())
			vkFreeCommandBuffers(queue->device, queue->command_pool, to_free.size(), to_free.data());
	}

	auto out = (GpuCommandBuffer*)queue->cpu_allocator(nullptr, sizeof(GpuCommandBuffer));
	new(out) GpuCommandBuffer{.queue = queue};

	VkCommandBufferAllocateInfo info {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = queue->command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1
	};
	VK_CHECK(vkAllocateCommandBuffers(queue->device, &info, &out->command_buffer), nullptr);

	VkCommandBufferBeginInfo begin {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};
	VK_CHECK(vkBeginCommandBuffer(out->command_buffer, &begin), nullptr);
	return out;
}

void gpuFreeCommandBuffer(GpuCommandBuffer& cmd) {
	vkFreeCommandBuffers(cmd.queue->device, cmd.queue->command_pool, 1, &cmd.command_buffer);
	cmd.~GpuCommandBuffer();
	cmd.queue->cpu_allocator(&cmd, 0);
}

uint64_t gpuSubmitNoFree(GpuQueue* queue, std::span<GpuCommandBuffer*> commandBuffers, GpuSemaphore* semaphore /* = nullptr */, uint64_t signalValue /* = 0 */) {
	std::vector<VkSemaphoreSubmitInfo> waits;
	std::vector<VkCommandBufferSubmitInfo> submits; submits.reserve(commandBuffers.size());
	for(auto& cmd: commandBuffers) {
		if(cmd->state != GpuCommandBuffer::Ended) {
			assert(cmd->state != GpuCommandBuffer::RecordingRenderPass && "Render pass hasn't been ended!");

			vkEndCommandBuffer(cmd->command_buffer);
			cmd->state = GpuCommandBuffer::Ended;
		}
		submits.emplace_back(VkCommandBufferSubmitInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
			.commandBuffer = cmd->command_buffer
		});

		for(auto sema: cmd->wait_semaphores)
			waits.emplace_back(VkSemaphoreSubmitInfo{
				.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
				.semaphore = sema,
				.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
			});
	}

	std::array<VkSemaphoreSubmitInfo, 2> signals {
		VkSemaphoreSubmitInfo{
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.semaphore = queue->command_submission_timeline_semaphore,
			.value = queue->command_submission_timeline_semaphore_next_value,
			.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
		}
	};
	if(semaphore) {
		signals[1] = VkSemaphoreSubmitInfo{
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.semaphore = semaphore->semaphore,
			.value = signalValue,
			.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
		};
	}
	VkSubmitInfo2 info {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
		.waitSemaphoreInfoCount = static_cast<uint32_t>(waits.size()),
		.pWaitSemaphoreInfos = waits.data(),
		.commandBufferInfoCount = static_cast<uint32_t>(submits.size()),
		.pCommandBufferInfos = submits.data(),
		.signalSemaphoreInfoCount = static_cast<uint32_t>(semaphore ? 2 : 1),
		.pSignalSemaphoreInfos = signals.data(),
	};
	VK_CHECK(vkQueueSubmit2(queue->queue, 1, &info, nullptr), queue->command_submission_timeline_semaphore_next_value);
	return queue->command_submission_timeline_semaphore_next_value++;
}

uint64_t gpuSubmit(GpuQueue* queue, std::span<GpuCommandBuffer*> commandBuffers, GpuSemaphore* semaphore /* = nullptr */, uint64_t signalValue /* = 0 */) {
	auto submission_index = gpuSubmitNoFree(queue, commandBuffers, semaphore, signalValue);

	for(auto& cmd: commandBuffers) {
		queue->command_buffers_pending_free.emplace_back(cmd->command_buffer, submission_index);

		cmd->~GpuCommandBuffer(); // Free the buffer
		queue->cpu_allocator(cmd, 0);
	}
	return submission_index;
}




std::optional<GpuSemaphore> gpuCreateSemaphoreImpl(GpuQueue* queue, uint64_t init_value) {
	GpuSemaphore out;

	VkSemaphoreTypeCreateInfo semaphore_type{};
	semaphore_type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
	semaphore_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	semaphore_type.initialValue = init_value; // Starting timeline value

	VkSemaphoreCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	info.pNext = &semaphore_type;
	VK_CHECK(vkCreateSemaphore(queue->device, &info, queue->callbacks, &out.semaphore), {});
	return out;
}
GpuSemaphore* gpuCreateSemaphore(GpuQueue* queue, uint64_t init_value) {
	if(auto out = gpuCreateSemaphoreImpl(queue, init_value); out) {
		auto ret = (GpuSemaphore*)queue->cpu_allocator(nullptr, sizeof(GpuSemaphore));
		*ret = *out;
		return ret;
	} else return nullptr;
}

uint64_t gpuWaitSemaphore(GpuQueue* queue, const GpuSemaphore* semaphore, uint64_t value, uint64_t timeout /* = UINT64_MAX */) {
	if(value == GPU_GET_VALUE) {
		uint64_t currentValue;
		VK_CHECK(vkGetSemaphoreCounterValue(queue->device, semaphore->semaphore, &currentValue), 0);
		return currentValue;
	}

	VkSemaphoreWaitInfo waitInfo{};
	waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
	waitInfo.flags = 0;
	waitInfo.semaphoreCount = 1;
	waitInfo.pSemaphores = &semaphore->semaphore;
	waitInfo.pValues = &value;

	VK_CHECK(vkWaitSemaphores(queue->device, &waitInfo, timeout), 0);
	return value;
}

void gpuFreeSemaphore(GpuQueue* queue, GpuSemaphore* semaphore) {
	vkDestroySemaphore(queue->device, semaphore->semaphore, queue->callbacks);
	queue->cpu_allocator(semaphore, 0);
}




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

	if(queue->gpu2index.contains(gpu_ptr)) {
		auto [buffer, allocation, _size] = queue->gpu2index[gpu_ptr];
		vmaDestroyBuffer(queue->gpu_allocator, buffer, allocation);
	}

	vmaDestroyBuffer(queue->gpu_allocator, buffer, allocation);
}

gpu* gpuHostToDevicePointer(GpuQueue* queue, void* ptr) {
	if(queue->host2gpu.contains(ptr))
		return (gpu*)queue->host2gpu[ptr];
	return nullptr;
}

void* gpuDeviceToHostPointerEXT(GpuQueue* queue, gpu* ptr) {
	auto gpu_ptr = (VkDeviceAddress)ptr;
	if(queue->gpu2host.contains(gpu_ptr))
		return queue->gpu2host[gpu_ptr];
	return nullptr;
}




namespace GPU::detail {

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

		return VkImageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.flags = VK_IMAGE_CREATE_ALIAS_BIT,
			.imageType = type2vulkan(descriptor.type),
			.format = GPU::detail::format2vulkan(descriptor.format),
			.extent = VkExtent3D{descriptor.dimensions.x, descriptor.dimensions.y, descriptor.dimensions.z},
			.mipLevels = descriptor.mipCount,
			.arrayLayers = descriptor.layerCount,
			.samples = GPU::detail::samples2vulkan(descriptor.sampleCount),
			.tiling = VK_IMAGE_TILING_LINEAR,
			.usage = GPU::detail::usage2vulkan(descriptor.usage),
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};
	}
}

GpuTextureSizeAlign gpuTextureSizeAlign(GpuQueue* queue, const GpuTextureDesc& desc) {
	auto info = GPU::detail::descriptor2vulkan(desc);
	VkImage temp;
	VK_CHECK(vkCreateImage(queue->device, &info, queue->callbacks, &temp), {});

	VkMemoryRequirements requirements;
	vkGetImageMemoryRequirements(queue->device, temp, &requirements);
	vkDestroyImage(queue->device, temp, queue->callbacks);

	return {requirements.size, requirements.alignment};
}

GpuTexture* gpuCreateTexture(GpuQueue* queue, const GpuTextureDesc& desc, gpu* memory) {
	if(queue->gpu2image.contains((VkDeviceAddress)memory)) {
		errno = VK_ERROR_TOO_MANY_OBJECTS;
		return nullptr;
	}

	auto out = (GpuTexture*)queue->cpu_allocator(nullptr, sizeof(GpuTexture));
	*out = {.descriptor = desc};

	auto info = GPU::detail::descriptor2vulkan(desc);
	VK_CHECK(vkCreateImage(queue->device, &info, queue->callbacks, &out->image), nullptr);
	VK_CHECK(vkBindImageMemory(queue->device, out->image, std::get<VmaAllocation>(queue->allocations[(VkDeviceAddress)memory])->GetMemory(), 0), nullptr);

	queue->gpu2image[(VkDeviceAddress)memory] = out->image;
	return out;
}

inline GpuTextureDescriptor gpuTextureViewDescriptorImpl(GpuQueue* queue, const GpuTexture* texture, const GpuViewDesc& desc, bool read_only) {
	GpuTextureDescriptor out = {};
	VkHostAddressRangeEXT host_info {
		.address = &out,
		.size = sizeof(GpuTextureDescriptor)
	};

	auto format = desc.format == FORMAT_NONE ? texture->descriptor.format : desc.format;
	VkImageViewCreateInfo view_info {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = texture->image,
		.viewType = GPU::detail::type2vulkan(texture->descriptor.type),
		.format = GPU::detail::format2vulkan(format),
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




const GpuSemaphore* gpuGetSubmissionSemaphoreEXT(GpuQueue* queue) {
	return (GpuSemaphore*)&queue->command_submission_timeline_semaphore;
}

void gpuWaitIdleEXT(GpuQueue* queue) {
	vkDeviceWaitIdle(queue->device);
}

void gpuSyncMemoryEXT(GpuCommandBuffer* cmd, gpu* mem) {
	// Do nothing!
}

void gpuSyncMemoryEXT(GpuQueue* queue, gpu* mem) {
	// Do nothing!
}




struct ComputePipelinePushConstants {
	gpu* data;
	gpu* sampler_map;
};

GpuPipeline* gpuCreateComputePipeline(GpuQueue* queue, std::span<const std::byte> computeIR) {
	auto out = (GpuPipeline*)queue->cpu_allocator(nullptr, sizeof(GpuPipeline));
	out->color_target_count = {}; // Null indicating compute pipeline

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

void gpuFreePipeline(GpuQueue* queue, GpuPipeline* pipeline) {
	vkDestroyPipeline(queue->device, pipeline->pipeline, queue->callbacks);
	queue->cpu_allocator(pipeline, 0);
}




void gpuMemCpy(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, size_t bytes, bool no_offsets /* = false*/) {
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_, no_offsets);
	auto [dest_buffer, dest_offset, dest_addr] = dest; auto [src_buffer, src_offset, src_addr] = src;

	VkBufferCopy region {
		.srcOffset = src_offset,
		.dstOffset = dest_offset,
		.size = bytes
	};
	vkCmdCopyBuffer(cmd->command_buffer, src_buffer, dest_buffer, 1, &region);
}

void gpuCopyToTexture(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, GpuTexture* texture, bool no_offsets /* = false */) {
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_, no_offsets);
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
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_, no_offsets);
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
	auto [source_buffer, offset, source_address] = GPU::detail::closest_buffer(cmd->queue, texture_heap, no_offsets);
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




namespace GPU::detail {

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
}

void gpuBarrier(GpuCommandBuffer* cmd, STAGE before, STAGE after, HAZARD_FLAGS hazards) {
	VkPipelineStageFlags2KHR src_stage = GPU::detail::stage2vulkan(before);
	VkPipelineStageFlags2KHR dst_stage = GPU::detail::stage2vulkan(after);
	auto [src_access, dst_access] = GPU::detail::hazard2access(hazards);

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
	vkCmdBindPipeline(cmd->command_buffer, pipeline->color_target_count.has_value() ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
	cmd->bound_pipeline = pipeline;
}

void gpuDispatch(GpuCommandBuffer* cmd, gpu* dataGpu, uvec3 gridDimensions) {
	ComputePipelinePushConstants data {
		.data = dataGpu,
		.sampler_map = (gpu*)cmd->sampler_map
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

	auto [buffer, offset, _address] = GPU::detail::closest_buffer(cmd->queue, gridDimensionsGpu, no_offsets);
	vkCmdDispatchIndirect(cmd->command_buffer, buffer, offset);
}




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




struct GraphicsPipelinePushConstants {
	gpu* vertex;
	gpu* fragment;
	gpu* index;
	gpu* sampler_map;
};

namespace GPU::detail {
	VkBlendOp blend2vulkan(BLEND op) {
		switch (op) {
		case BLEND_ADD: return VK_BLEND_OP_ADD;
		case BLEND_SUBTRACT: return VK_BLEND_OP_SUBTRACT;
		case BLEND_MIN: return VK_BLEND_OP_MIN;
		case BLEND_MAX: return VK_BLEND_OP_MAX;
		case BLEND_REV_SUBTRACT: return VK_BLEND_OP_REVERSE_SUBTRACT;
		}
		std::unreachable();
	};

	VkBlendFactor factor2vulkan(FACTOR f) {
		switch (f) {
			case FACTOR_ZERO: return VK_BLEND_FACTOR_ZERO;
			case FACTOR_ONE: return VK_BLEND_FACTOR_ONE;
			case FACTOR_SRC_COLOR: return VK_BLEND_FACTOR_SRC_COLOR;
			case FACTOR_ONE_MINUS_SRC_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
			case FACTOR_DST_COLOR: return VK_BLEND_FACTOR_DST_COLOR;
			case FACTOR_ONE_MINUS_DST_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
			case FACTOR_SRC_ALPHA: return VK_BLEND_FACTOR_SRC_ALPHA;
			case FACTOR_ONE_MINUS_SRC_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			case FACTOR_DST_ALPHA: return VK_BLEND_FACTOR_DST_ALPHA;
			case FACTOR_ONE_MINUS_DST_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
			case FACTOR_SRC1_COLOR: return VK_BLEND_FACTOR_SRC1_COLOR;
			case FACTOR_ONE_MINUS_SRC1_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
			case FACTOR_SRC1_ALPHA: return VK_BLEND_FACTOR_SRC1_ALPHA;
			case FACTOR_ONE_MINUS_SRC1_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
		}
		std::unreachable();
	};

	VkColorComponentFlags mask2vulkan(uint8_t mask) {
		VkColorComponentFlags flags = 0;
		if (mask & 0x1) flags |= VK_COLOR_COMPONENT_R_BIT;
		if (mask & 0x2) flags |= VK_COLOR_COMPONENT_G_BIT;
		if (mask & 0x4) flags |= VK_COLOR_COMPONENT_B_BIT;
		if (mask & 0x8) flags |= VK_COLOR_COMPONENT_A_BIT;
		return flags;
	};
}



GpuPipeline* gpuCreateGraphicsPipeline(GpuQueue* queue, std::span<const std::byte> vertexIR, std::span<const std::byte> fragmentIR, const GpuRasterDesc& desc) {
	constexpr static auto topology2vulkan = [](TOPOLOGY t) -> VkPrimitiveTopology{
		switch (t) {
			// case TOPOLOGY_POINT_LIST: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
			// case TOPOLOGY_LINE_LIST: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
			// case TOPOLOGY_LINE_STRIP: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
			case TOPOLOGY_TRIANGLE_LIST: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			case TOPOLOGY_TRIANGLE_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
		}
		std::unreachable();
	};
	constexpr static auto cull2vulkan = [](CULL c) -> VkCullModeFlags {
		// We are going to define CCW as front
		switch (c) {
			case CULL_NONE: return VK_CULL_MODE_NONE;
			case CULL_CCW: return VK_CULL_MODE_FRONT_BIT;
			case CULL_CW: return VK_CULL_MODE_BACK_BIT;
			case CULL_ALL: return VK_CULL_MODE_FRONT_AND_BACK;
		}
		std::unreachable();
	};

	auto out = (GpuPipeline*)queue->cpu_allocator(nullptr, sizeof(GpuPipeline));
	out->color_target_count = desc.colorTargets.size();

	std::array<VkShaderModule, 2> shader_modules;
	{
		VkShaderModuleCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = vertexIR.size(),
			.pCode = (uint32_t*)vertexIR.data(),
		};
		VK_CHECK(vkCreateShaderModule(queue->device, &info, queue->callbacks, &shader_modules[0]), nullptr);
	}{
		VkShaderModuleCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = fragmentIR.size(),
			.pCode = (uint32_t*)fragmentIR.data(),
		};
		VK_CHECK(vkCreateShaderModule(queue->device, &info, queue->callbacks, &shader_modules[1]), nullptr);
	}

	std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages {
		VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = shader_modules[0],
			.pName = "main",
		}, VkPipelineShaderStageCreateInfo {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = shader_modules[1],
			.pName = "main",
		}
	};

	VkPipelineVertexInputStateCreateInfo vertex_input_state = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

	VkPipelineInputAssemblyStateCreateInfo assembly_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = topology2vulkan(desc.topology),
	};

	VkPipelineViewportStateCreateInfo viewport_state {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	VkPipelineRasterizationStateCreateInfo rasterization_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = cull2vulkan(desc.cull),
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	VkPipelineMultisampleStateCreateInfo multisample_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = GPU::detail::samples2vulkan(desc.sampleCount),
		.alphaToCoverageEnable = desc.alphaToCoverage ? VK_TRUE : VK_FALSE,
	};

	VkPipelineDepthStencilStateCreateInfo depth_stencil_state{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

	std::vector<VkPipelineColorBlendAttachmentState> attachments;
	{
		attachments.reserve(desc.colorTargets.size());

		const bool blendEnabled = desc.blendstate.has_value();
		const GpuBlendDesc blend = desc.blendstate.value_or(GpuBlendDesc{});

		for (const GpuColorTarget& target : desc.colorTargets) {
			VkPipelineColorBlendAttachmentState state{};
			state.blendEnable = blendEnabled ? VK_TRUE : VK_FALSE;

			if (blendEnabled) {
				state.srcColorBlendFactor = GPU::detail::factor2vulkan(blend.srcColorFactor);
				state.dstColorBlendFactor = GPU::detail::factor2vulkan(blend.dstColorFactor);
				state.colorBlendOp = GPU::detail::blend2vulkan(blend.colorOp);
				state.srcAlphaBlendFactor = GPU::detail::factor2vulkan(blend.srcAlphaFactor);
				state.dstAlphaBlendFactor = GPU::detail::factor2vulkan(blend.dstAlphaFactor);
				state.alphaBlendOp = GPU::detail::blend2vulkan(blend.alphaOp);
			}

			const uint8_t effectiveMask = blendEnabled ? (target.writeMask & blend.colorWriteMask) : target.writeMask;
			state.colorWriteMask = GPU::detail::mask2vulkan(effectiveMask);

			attachments.push_back(state);
		}
	}

	VkPipelineColorBlendStateCreateInfo color_blend_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = static_cast<uint32_t>(attachments.size()),
		.pAttachments = attachments.empty() ? nullptr : attachments.data(),
	};

	std::array<VkDynamicState, 14> dynamic = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
		VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
		VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
		VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
		VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
		VK_DYNAMIC_STATE_STENCIL_OP,
		VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
		VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
		VK_DYNAMIC_STATE_STENCIL_REFERENCE,
		VK_DYNAMIC_STATE_DEPTH_BIAS,
		VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT,
		VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,
		VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT,
	};

	VkPipelineDynamicStateCreateInfo dynamic_state{};
	dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic_state.dynamicStateCount = 2;
	dynamic_state.pDynamicStates = dynamic.data();

	std::vector<VkFormat> color_formats; color_formats.reserve(desc.colorTargets.size());
	for(auto& target: desc.colorTargets)
		color_formats.emplace_back(GPU::detail::format2vulkan(target.format));

	VkPipelineRenderingCreateInfo dynamic_rendering_info {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = static_cast<uint32_t>(color_formats.size()),
		.pColorAttachmentFormats = color_formats.data(),
		.depthAttachmentFormat = desc.depthFormat == FORMAT_NONE ? VK_FORMAT_UNDEFINED : GPU::detail::format2vulkan(desc.depthFormat),
		.stencilAttachmentFormat = desc.stencilFormat == FORMAT_NONE ? VK_FORMAT_UNDEFINED : GPU::detail::format2vulkan(desc.stencilFormat),
	};

	VkPipelineCreateFlags2CreateInfo create_flags {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
		.pNext = &dynamic_rendering_info,
		.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT
	};
	VkGraphicsPipelineCreateInfo info {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = &create_flags,
		.stageCount = shader_stages.size(),
		.pStages = shader_stages.data(),
		.pVertexInputState = &vertex_input_state,
		.pInputAssemblyState = &assembly_state,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization_state,
		.pMultisampleState = &multisample_state,
		.pDepthStencilState = &depth_stencil_state,
		.pColorBlendState = &color_blend_state,
		.pDynamicState = &dynamic_state,
		.renderPass = VK_NULL_HANDLE,
		// .subpass = 0,
		.basePipelineIndex = -1
	};
	vkCreateGraphicsPipelines(queue->device, VK_NULL_HANDLE, 1, &info, queue->callbacks, &out->pipeline);

	for(auto module: shader_modules)
		vkDestroyShaderModule(queue->device, module, queue->callbacks);
	// vkDestroyRenderPass(queue->device, compatible_render_pass, queue->callbacks);

	return out;
}




// TODO: Untested!
GpuDepthStencilState* gpuCreateDepthStencilState(GpuQueue* queue, const GpuDepthStencilDesc& desc) {
	auto out = (GpuDepthStencilState*)queue->cpu_allocator(nullptr, sizeof(GpuDepthStencilState));
	out->descriptor = desc;
	return out;
}

// TODO: Untested!
GpuBlendState* gpuCreateBlendState(GpuQueue* queue, const GpuBlendDesc& desc) {
	auto out = (GpuBlendState*)queue->cpu_allocator(nullptr, sizeof(GpuBlendState));
	out->descriptor = desc;
	return out;
}

// TODO: Untested!
void gpuFreeDepthStencilState(GpuQueue* queue, GpuDepthStencilState* state) {
	queue->cpu_allocator(state, 0);
}

// TODO: Untested!
void gpuFreeBlendState(GpuQueue* queue, GpuBlendState* state) {
	queue->cpu_allocator(state, 0);
}

// TODO: Untested!
void gpuSetDepthStencilState(GpuCommandBuffer* cmd, const GpuDepthStencilState* state) {
	constexpr static auto op2vulkan = [](OP op) -> VkCompareOp {
		switch (op) {
			case OP_ALWAYS: return VK_COMPARE_OP_ALWAYS;
			case OP_NEVER: return VK_COMPARE_OP_NEVER;
			case OP_LESS: return VK_COMPARE_OP_LESS;
			case OP_LESS_EQUAL: return VK_COMPARE_OP_LESS_OR_EQUAL;
			case OP_GREATER: return VK_COMPARE_OP_GREATER;
			case OP_GREATER_EQUAL: return VK_COMPARE_OP_GREATER_OR_EQUAL;
			case OP_EQUAL: return VK_COMPARE_OP_EQUAL;
			case OP_NOT_EQUAL: return VK_COMPARE_OP_NOT_EQUAL;
		}
		std::unreachable();
	};
	constexpr static auto stencil2vulkan = [](STENCIL_OP op) -> VkStencilOp {
		switch (op) {
			case STENCIL_OP_KEEP: return VK_STENCIL_OP_KEEP;
			case STENCIL_OP_ZERO: return VK_STENCIL_OP_ZERO;
			case STENCIL_OP_REPLACE: return VK_STENCIL_OP_REPLACE;
			case STENCIL_OP_INCR_SAT: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
			case STENCIL_OP_DECR_SAT: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
			case STENCIL_OP_INVERT: return VK_STENCIL_OP_INVERT;
			case STENCIL_OP_INCR_WRAP: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
			case STENCIL_OP_DECR_WRAP: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
		}
		std::unreachable();
	};

	const bool depthRead = (state->descriptor.depthMode & DEPTH_READ) != 0;
	const bool depthWrite = (state->descriptor.depthMode & DEPTH_WRITE) != 0;

	// depthTestEnable gates both the compare op AND whether writes happen
	// in classic GL/D3D semantics; Vulkan separates test-enable from
	// write-enable, so: enable the test if either read or write was
	// requested (write without a passing test wouldn't write anything
	// meaningful), and gate the actual write bit off DEPTH_WRITE.
	vkCmdSetDepthTestEnable(cmd->command_buffer, (depthRead || depthWrite) ? VK_TRUE : VK_FALSE);
	vkCmdSetDepthWriteEnable(cmd->command_buffer, depthWrite ? VK_TRUE : VK_FALSE);
	vkCmdSetDepthCompareOp(cmd->command_buffer, op2vulkan(state->descriptor.depthTest));

	const bool stencilEnabled = state->descriptor.stencilFront.test != OP_ALWAYS || state->descriptor.stencilBack.test != OP_ALWAYS
		|| state->descriptor.stencilFront.failOp != STENCIL_OP_KEEP || state->descriptor.stencilFront.passOp != STENCIL_OP_KEEP
		|| state->descriptor.stencilFront.depthFailOp != STENCIL_OP_KEEP
		|| state->descriptor.stencilBack.failOp != STENCIL_OP_KEEP || state->descriptor.stencilBack.passOp != STENCIL_OP_KEEP
		|| state->descriptor.stencilBack.depthFailOp != STENCIL_OP_KEEP;
	vkCmdSetStencilTestEnable(cmd->command_buffer, stencilEnabled ? VK_TRUE : VK_FALSE);

	if (stencilEnabled) {
		vkCmdSetStencilOp(cmd->command_buffer, VK_STENCIL_FACE_FRONT_BIT, stencil2vulkan(state->descriptor.stencilFront.failOp),
			stencil2vulkan(state->descriptor.stencilFront.passOp), stencil2vulkan(state->descriptor.stencilFront.depthFailOp),
			op2vulkan(state->descriptor.stencilFront.test)
		);
		vkCmdSetStencilOp(cmd->command_buffer, VK_STENCIL_FACE_BACK_BIT, stencil2vulkan(state->descriptor.stencilBack.failOp),
			stencil2vulkan(state->descriptor.stencilBack.passOp), stencil2vulkan(state->descriptor.stencilBack.depthFailOp),
			op2vulkan(state->descriptor.stencilBack.test)
		);

		vkCmdSetStencilCompareMask(cmd->command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, state->descriptor.stencilReadMask);
		vkCmdSetStencilWriteMask(cmd->command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, state->descriptor.stencilWriteMask);

		// Front/back reference values differ in your struct
		// (Stencil::reference is per-face) but vkCmdSetStencilReference
		// takes a face mask too, so two calls if front != back.
		if (state->descriptor.stencilFront.reference == state->descriptor.stencilBack.reference) {
			vkCmdSetStencilReference(cmd->command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, state->descriptor.stencilFront.reference);
		} else {
			vkCmdSetStencilReference(cmd->command_buffer, VK_STENCIL_FACE_FRONT_BIT, state->descriptor.stencilFront.reference);
			vkCmdSetStencilReference(cmd->command_buffer, VK_STENCIL_FACE_BACK_BIT, state->descriptor.stencilBack.reference);
		}
	}

	vkCmdSetDepthBias(cmd->command_buffer, state->descriptor.depthBias, state->descriptor.depthBiasClamp, state->descriptor.depthBiasSlopeFactor);
}

// TODO: Untested!
void gpuSetBlendState(GpuCommandBuffer* cmd, const GpuBlendState* state) {
	assert(cmd->bound_pipeline);

	const uint32_t count = cmd->bound_pipeline->color_target_count.value_or(0);
	const bool blend_enable = state->descriptor.colorWriteMask > 0;

	std::vector<VkBool32> enables(count, blend_enable ? VK_TRUE : VK_FALSE);
	vkCmdSetColorBlendEnableEXT(cmd->command_buffer, 0, count, enables.data());

	if(blend_enable) {
		std::vector<VkColorBlendEquationEXT> equations(count);
		for (uint32_t i = 0; i < count; ++i) {
			equations[i].srcColorBlendFactor = GPU::detail::factor2vulkan(state->descriptor.srcColorFactor);
			equations[i].dstColorBlendFactor = GPU::detail::factor2vulkan(state->descriptor.dstColorFactor);
			equations[i].colorBlendOp = GPU::detail::blend2vulkan(state->descriptor.colorOp);
			equations[i].srcAlphaBlendFactor = GPU::detail::factor2vulkan(state->descriptor.srcAlphaFactor);
			equations[i].dstAlphaBlendFactor = GPU::detail::factor2vulkan(state->descriptor.dstAlphaFactor);
			equations[i].alphaBlendOp = GPU::detail::blend2vulkan(state->descriptor.alphaOp);
		}
		vkCmdSetColorBlendEquationEXT(cmd->command_buffer, 0, count, equations.data());
	}

	// std::vector<VkColorComponentFlags> masks(count);
	// for (uint32_t i = 0; i < count; ++i)
	// 	masks[i] = mask2vulkan(writeMasks[i] & state->descriptor.colorWriteMask);
	// vkCmdSetColorWriteMaskEXT(cmd, 0, count, masks.data());
}

void gpuSetViewportEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin /*= {0, 0} */, float depth_min /* = 0 */, float depth_max /* = 1 */) {
	VkViewport viewport {
		.x = static_cast<float>(origin.x),
		.y = static_cast<float>(origin.y),
		.width = static_cast<float>(extent.x),
		.height = static_cast<float>(extent.y),
		.minDepth = depth_min,
		.maxDepth = depth_max,
	};
	vkCmdSetViewport(cmd->command_buffer, 0, 1, &viewport);
}

void gpuSetScissorRectEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin /* = {0, 0} */) {
	VkRect2D scissor { // TODO: Should we support additional scissors?
		.offset = {origin.x, origin.y},
		.extent = {extent.x, extent.y},
	};
	vkCmdSetScissor(cmd->command_buffer, 0, 1, &scissor);
}




inline void transition_image_layout(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout, VkAccessFlags source_access_mask, VkAccessFlags destination_access_mask, VkPipelineStageFlags source_stage, VkPipelineStageFlags destination_stage, uint32_t base_mip_level, uint32_t mip_levels, uint32_t base_slice, uint32_t slices) {
	VkImageMemoryBarrier barrier{
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = source_access_mask,
		.dstAccessMask = destination_access_mask,
		.oldLayout = old_layout,
		.newLayout = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = base_mip_level == ALL_MIPS ? 0 : base_mip_level,
			.levelCount = base_mip_level == ALL_MIPS ? mip_levels : 1,
			.baseArrayLayer = base_slice == ALL_LAYERS ? 0 : base_slice,
			.layerCount = base_slice == ALL_LAYERS ? slices : 1,
		}
	};
	vkCmdPipelineBarrier(cmd, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void gpuBeginRenderPass(GpuCommandBuffer* cmd, const GpuRenderPassDesc& desc) {
	constexpr static auto view_create_info = [](const GpuTexture* texture) -> VkImageViewCreateInfo {
		return {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = texture->image,
			.viewType = GPU::detail::type2vulkan(texture->descriptor.type),
			.format = GPU::detail::format2vulkan(texture->descriptor.format),
			.subresourceRange = {
				.aspectMask = static_cast<VkImageAspectFlags>(gpuFormatIsDepth(texture->descriptor.format)
					? VK_IMAGE_ASPECT_DEPTH_BIT | (gpuFormatIsStencil(texture->descriptor.format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)
					: VK_IMAGE_ASPECT_COLOR_BIT),
				.baseMipLevel = 0,
				.levelCount = texture->descriptor.mipCount,
				.baseArrayLayer = 0,
				.layerCount = texture->descriptor.layerCount
			}
		};
	};

	constexpr static auto load2vulkan = [](LOAD_OP op) {
		switch (op) {
		case LOAD_OP_LOAD: return VK_ATTACHMENT_LOAD_OP_LOAD;
		case LOAD_OP_CLEAR: return VK_ATTACHMENT_LOAD_OP_CLEAR;
		case LOAD_OP_DONT_CARE: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		}
		std::unreachable();
	};
	constexpr static auto store2vulkan = [](STORE_OP op) {
		switch (op) {
		case STORE_OP_STORE: return VK_ATTACHMENT_STORE_OP_STORE;
		case STORE_OP_DONT_CARE: return VK_ATTACHMENT_STORE_OP_DONT_CARE;
		}
		std::unreachable();
	};

	assert(cmd->state == GpuCommandBuffer::Recording && "The command buffer must not be ended or recording another active render pass");
	cmd->state = GpuCommandBuffer::RecordingRenderPass;

	std::vector<VkRenderingAttachmentInfo> color_attachments; color_attachments.reserve(desc.colorAttachments.size());
	for(auto& color: desc.colorAttachments) {
		if(color.texture->available_semaphore)
			cmd->wait_semaphores.push_back(color.texture->available_semaphore);

		transition_image_layout(cmd->command_buffer,
			color.texture->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, color.mipLevel, color.texture->descriptor.mipCount, color.slice, color.texture->descriptor.sampleCount
		);

		if(!color.texture->full_view) {
			auto info = view_create_info(color.texture);
			vkCreateImageView(cmd->queue->device, &info, cmd->queue->callbacks, const_cast<VkImageView*>(&color.texture->full_view));
		}
		if(color.resolveTexture && !color.resolveTexture->full_view) {
			auto info = view_create_info(color.resolveTexture);
			VK_CHECK(vkCreateImageView(cmd->queue->device, &info, cmd->queue->callbacks, const_cast<VkImageView*>(&color.resolveTexture->full_view)), /*NOTHING*/);
		}

		color_attachments.emplace_back(VkRenderingAttachmentInfo{
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = color.texture->full_view,
			.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.resolveMode = color.resolveTexture ? VK_RESOLVE_MODE_SAMPLE_ZERO_BIT : VK_RESOLVE_MODE_NONE,
			.resolveImageView = color.resolveTexture ? color.resolveTexture->full_view : VK_NULL_HANDLE,
			.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.loadOp = load2vulkan(color.loadOp),
			.storeOp = store2vulkan(color.storeOp),
			.clearValue = {
				.color = {
					.float32 = {color.clearValue.r, color.clearValue.g, color.clearValue.b, color.clearValue.a}
				}
			}
		});
	}

	VkRenderingAttachmentInfo depth_attachment;
	if(desc.depthAttachment) {
		if(desc.depthAttachment->texture->available_semaphore)
			cmd->wait_semaphores.push_back(desc.depthAttachment->texture->available_semaphore);

		transition_image_layout(cmd->command_buffer,
			desc.depthAttachment->texture->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, 0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, desc.depthAttachment->mipLevel, desc.depthAttachment->texture->descriptor.mipCount,
			desc.depthAttachment->slice, desc.depthAttachment->texture->descriptor.sampleCount
		);

		if(!desc.depthAttachment->texture->full_view) {
			auto info = view_create_info(desc.depthAttachment->texture);
			VK_CHECK(vkCreateImageView(cmd->queue->device, &info, cmd->queue->callbacks, const_cast<VkImageView*>(&desc.depthAttachment->texture->full_view)), /*nothing*/);
		}

		depth_attachment = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = desc.depthAttachment->texture->full_view,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			.resolveMode = VK_RESOLVE_MODE_NONE,
			.resolveImageView = VK_NULL_HANDLE,
			.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.loadOp = load2vulkan(desc.depthAttachment->loadOp),
			.storeOp = store2vulkan(desc.depthAttachment->storeOp),
			.clearValue = {
				.depthStencil = {
					.depth = static_cast<float>(desc.depthAttachment->clearValue)
				}
			}
		};
	}

	VkRenderingAttachmentInfo stencil_attachment;
	if(desc.stencilAttachment) {
		if(desc.stencilAttachment->texture->available_semaphore)
			cmd->wait_semaphores.push_back(desc.stencilAttachment->texture->available_semaphore);

		transition_image_layout(cmd->command_buffer,
			desc.stencilAttachment->texture->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, desc.stencilAttachment->mipLevel, desc.stencilAttachment->texture->descriptor.mipCount,
			desc.stencilAttachment->slice, desc.stencilAttachment->texture->descriptor.sampleCount
		);

		if(!desc.stencilAttachment->texture->full_view) {
			auto info = view_create_info(desc.stencilAttachment->texture);
			VK_CHECK(vkCreateImageView(cmd->queue->device, &info, cmd->queue->callbacks, const_cast<VkImageView*>(&desc.stencilAttachment->texture->full_view)), /*NOTHING*/);
		}

		stencil_attachment = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = desc.stencilAttachment->texture->full_view,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			.resolveMode = VK_RESOLVE_MODE_NONE,
			.resolveImageView = VK_NULL_HANDLE,
			.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.loadOp = load2vulkan(desc.stencilAttachment->loadOp),
			.storeOp = store2vulkan(desc.stencilAttachment->storeOp),
			.clearValue = {
				.depthStencil = {
					.stencil = static_cast<uint32_t>(desc.stencilAttachment->clearValue)
				}
			}
		};
	}

	VkRenderingInfo info{
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = {
			.offset = {0, 0},
			.extent = {desc.colorAttachments[0].texture->descriptor.dimensions.x, desc.colorAttachments[0].texture->descriptor.dimensions.y},
		},
		.layerCount = 1,
		.colorAttachmentCount = static_cast<uint32_t>(color_attachments.size()),
		.pColorAttachments = color_attachments.data(),
		.pDepthAttachment = desc.depthAttachment ? &depth_attachment : nullptr,
		.pStencilAttachment = desc.stencilAttachment ? &stencil_attachment : nullptr,
	};
	vkCmdBeginRendering(cmd->command_buffer, &info);

	gpuSetViewportEXT(cmd, {info.renderArea.extent.width, info.renderArea.extent.height});
	gpuSetScissorRectEXT(cmd, {info.renderArea.extent.width, info.renderArea.extent.height});
}

void gpuEndRenderPass(GpuCommandBuffer* cmd, std::optional<const GpuRenderPassDesc> desc /*= {}*/) {
	vkCmdEndRendering(cmd->command_buffer);

	if(desc) for(auto& color: desc->colorAttachments)
		transition_image_layout(cmd->command_buffer,
			color.texture->image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, color.mipLevel, color.texture->descriptor.mipCount, color.slice, color.texture->descriptor.sampleCount
		);

	cmd->state = GpuCommandBuffer::Recording;
}




namespace GPU::detail {
	std::pair<VkBuffer, VkDeviceSize> ensureIndexBufferAvailable(GpuQueue* queue, gpu* indicesGpu, bool no_offsets, bool no_index_buffer_changes) {
		constexpr static std::pair<VkBuffer, VkDeviceSize> null_out = {VK_NULL_HANDLE, 0};

		auto [_buffer, offset, address] = GPU::detail::closest_buffer(queue, indicesGpu, no_offsets);
		auto [source_buffer, _alloc, size] = queue->allocations[address];
		if(!queue->gpu2index.contains(address)) {
			VkBufferCreateInfo buffer_info {
				.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
				.size = size,
				.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
			};
			VmaAllocationCreateInfo alloc_info {
				.usage = VMA_MEMORY_USAGE_AUTO,
				.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
			};

			auto& [index_buffer, allocation, index_size] = queue->gpu2index[address];
			index_size = size;
			VK_CHECK(vmaCreateBuffer(queue->gpu_allocator, &buffer_info, &alloc_info, &index_buffer, &allocation, nullptr), null_out);

			no_index_buffer_changes = false;
		}

		auto [index_buffer, _allocation, _size] = queue->gpu2index[address];
		if(!no_index_buffer_changes) {
			VkCommandBuffer tmp = VK_NULL_HANDLE;
			VkCommandBufferAllocateInfo info {
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
				.commandPool = queue->command_pool,
				.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				.commandBufferCount = 1
			};
			VK_CHECK(vkAllocateCommandBuffers(queue->device, &info, &tmp), null_out);

			VkCommandBufferBeginInfo begin {
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
			};
			VK_CHECK(vkBeginCommandBuffer(tmp, &begin), null_out);

			VkBufferCopy copy {
				.srcOffset = offset,
				.dstOffset = offset,
				.size = size - offset
			};
			vkCmdCopyBuffer(tmp, source_buffer, index_buffer, 1, &copy);

			VK_CHECK(vkEndCommandBuffer(tmp), null_out);
			VkSubmitInfo submit {
				.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
				.commandBufferCount = 1,
				.pCommandBuffers = &tmp,
			};
			VK_CHECK(vkQueueSubmit(queue->queue, 1, &submit, VK_NULL_HANDLE), null_out);
			queue->command_buffers_pending_free.emplace_back(tmp, queue->command_submission_timeline_semaphore_next_value);
		}

		return {index_buffer, offset};
	}

	VkIndexType index2vulkan(INDEX_TYPE_EXT index_type) {
		switch (index_type) {
		case INDEX_TYPE_UINT8: return VK_INDEX_TYPE_UINT8;
		case INDEX_TYPE_UINT16: return VK_INDEX_TYPE_UINT16;
		case INDEX_TYPE_UINT32: return VK_INDEX_TYPE_UINT32;
		}
		std::unreachable();
	}
}



void gpuDrawIndexedInstanced(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* indices, uint32_t index_count, uint32_t instance_count, INDEX_TYPE_EXT index_type /* = INDEX_TYPE_UINT32 */, bool no_offsets /* = false */, bool no_index_buffer_changes /* = false */) {
	GraphicsPipelinePushConstants data {
		.vertex = vertex_data,
		.fragment = fragment_data,
		.index = indices,
		.sampler_map = (gpu*)cmd->sampler_map
	};
	VkPushDataInfoEXT info {
		.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
		.offset = 0,
		.data = {
			.address = &data,
			.size = sizeof(GraphicsPipelinePushConstants)
		}
	};
	vkCmdPushDataEXT(cmd->command_buffer, &info);

	auto [index_buffer, offset] = GPU::detail::ensureIndexBufferAvailable(cmd->queue, indices, no_offsets, no_index_buffer_changes);
	assert(index_buffer != VK_NULL_HANDLE);

	vkCmdBindIndexBuffer(cmd->command_buffer, index_buffer, offset, GPU::detail::index2vulkan(index_type));
	vkCmdDrawIndexed(cmd->command_buffer, index_count, instance_count, 0, 0, 0);
}

// TODO: Untested!
void gpuDrawIndexedInstancedIndirect(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* indices, gpu* args, INDEX_TYPE_EXT index_type /* = INDEX_TYPE_UINT32 */, bool no_offsets /* = false */, bool no_index_buffer_changes /* = false */) {
	GraphicsPipelinePushConstants data {
		.vertex = vertex_data,
		.fragment = fragment_data,
		.index = indices,
		.sampler_map = (gpu*)cmd->sampler_map
	};
	VkPushDataInfoEXT info {
		.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
		.offset = 0,
		.data = {
			.address = &data,
			.size = sizeof(GraphicsPipelinePushConstants)
		}
	};
	vkCmdPushDataEXT(cmd->command_buffer, &info);

	{
		auto [index_buffer, offset] = GPU::detail::ensureIndexBufferAvailable(cmd->queue, indices, no_offsets, no_index_buffer_changes);
		assert(index_buffer != VK_NULL_HANDLE);
		vkCmdBindIndexBuffer(cmd->command_buffer, index_buffer, offset, GPU::detail::index2vulkan(index_type));
	}

	auto [_buffer, offset, address] = GPU::detail::closest_buffer(cmd->queue, args, no_offsets);
	auto [buffer, _allocation, size] = cmd->queue->allocations[address];
	auto count = (size - offset) / sizeof(VkDrawIndexedIndirectCommand);
	vkCmdDrawIndexedIndirect(cmd->command_buffer, buffer, offset, count, sizeof(VkDrawIndexedIndirectCommand)); // TODO: We should we check the size of the buffer and divide by sizeof(VkDrawIndexedIndirectCommand)?
}

// TODO: Untested!
void gpuDrawMeshlets(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, uvec3 dim) {
	GraphicsPipelinePushConstants data {
		.vertex = meshlet_data,
		.fragment = fragment_data,
		.index = nullptr,
		.sampler_map = (gpu*)cmd->sampler_map
	};
	VkPushDataInfoEXT info {
		.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
		.offset = 0,
		.data = {
			.address = &data,
			.size = sizeof(GraphicsPipelinePushConstants)
		}
	};
	vkCmdPushDataEXT(cmd->command_buffer, &info);

	vkCmdDrawMeshTasksEXT(cmd->command_buffer, dim.x, dim.y, dim.z);
}

// TODO: Untested!
void gpuDrawMeshletsIndirect(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, gpu* dim, bool no_offsets /* = false */) {
	GraphicsPipelinePushConstants data {
		.vertex = meshlet_data,
		.fragment = fragment_data,
		.index = nullptr,
		.sampler_map = (gpu*)cmd->sampler_map
	};
	VkPushDataInfoEXT info {
		.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
		.offset = 0,
		.data = {
			.address = &data,
			.size = sizeof(GraphicsPipelinePushConstants)
		}
	};
	vkCmdPushDataEXT(cmd->command_buffer, &info);

	auto [_buffer, offset, address] = GPU::detail::closest_buffer(cmd->queue, dim, no_offsets);
	auto [buffer, _allocation, size] = cmd->queue->allocations[address];
	auto count = (size - offset) / sizeof(VkDrawMeshTasksIndirectCommandEXT); // VkDrawMeshTasksIndirectCommandEXT == uvec3
	vkCmdDrawMeshTasksIndirectEXT(cmd->command_buffer, buffer, offset, count, sizeof(VkDrawMeshTasksIndirectCommandEXT));
}




GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, VkSurfaceKHR surface, const GpuSurfaceDescriptor& desc) {
	auto out = (GpuSurface*)queue->cpu_allocator(nullptr, sizeof(GpuSurface));
	new(out) GpuSurface {
		.surface = surface,
	};

	gpuSurfaceReconfigureEXT(queue, out, desc);
	return out;
}

void gpuFreeSurfaceNoSemaphores(GpuQueue* queue, GpuSurface* surface) { 
	for(auto view: surface->image_views)
		vkDestroyImageView(queue->device, view, queue->callbacks);
	if(surface->swapchain)
		vkb::destroy_swapchain(*surface->swapchain);
}

void gpuFreeSurfaceEXT(GpuQueue* queue, GpuSurface* surface) {
	for(auto semaphore: surface->image_available_semaphores)
		vkDestroySemaphore(queue->device, semaphore, queue->callbacks);
	for(auto semaphore: surface->render_finished_semaphores)
		vkDestroySemaphore(queue->device, semaphore, queue->callbacks);
	gpuFreeSurfaceNoSemaphores(queue, surface);
	surface->~GpuSurface();
	queue->cpu_allocator(surface, 0);
}

void gpuSurfaceReconfigureEXT(GpuQueue* queue, GpuSurface* surface, const GpuSurfaceDescriptor& desc) {
	surface->descriptor = desc;
	surface->descriptor.texture.type = TEXTURE_2D;
	surface->descriptor.texture.mipCount = 1;
	surface->descriptor.texture.sampleCount = 1;

	auto builder = vkb::SwapchainBuilder(queue->gpu, queue->device, surface->surface, queue->queue_family);
	if(surface->swapchain)
		builder.set_old_swapchain(*surface->swapchain);
	switch(desc.presentMode){
	break; case PRESENT_MODE_IMMEDIATE:
		builder.set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR);
	break; case PRESENT_MODE_FIFO:
		builder.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR);
	break; case PRESENT_MODE_FIFO_RELAXED:
		builder.set_desired_present_mode(VK_PRESENT_MODE_FIFO_RELAXED_KHR);
	break; case PRESENT_MODE_MAILBOX:
		builder.set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR);
	break; case PRESENT_MODE_BEST_AVAILABLE:
		builder.use_default_present_mode_selection();
	}
	assert(surface->descriptor.texture.dimensions.z == 1);
	auto swap = builder.set_desired_extent(surface->descriptor.texture.dimensions.x, surface->descriptor.texture.dimensions.y)
		.set_allocation_callbacks(queue->callbacks)
		.set_composite_alpha_flags(surface->descriptor.opaque ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
		.set_desired_format({
			GPU::detail::format2vulkan(surface->descriptor.texture.format),
			VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
		})
		.set_image_array_layer_count(surface->descriptor.texture.layerCount)
		.add_image_usage_flags(GPU::detail::usage2vulkan(surface->descriptor.texture.usage))
		.build();
	if(!swap) errno = swap.error().value();

	gpuFreeSurfaceNoSemaphores(queue, surface);
	surface->swapchain = std::make_shared<vkb::Swapchain>(std::move(*swap));
	std::vector<VkImage> images;
	std::tie(images, surface->image_views) = surface->swapchain->get_images_and_image_views().value();

	surface->images.resize(images.size());
	for(size_t i = 0; i < images.size(); ++i)
		surface->images[i] = GpuTexture {
			images[i],
			surface->image_views[i],
			surface->descriptor.texture
		};
}

GpuSurfaceDescriptor gpuSurfaceGetConfigurationEXT(const GpuSurface* surface) {
	return surface->descriptor;
}

const GpuTexture* gpuSurfaceNextTextureEXT(GpuQueue* queue, GpuSurface* surface) {
	if(surface->image_available_semaphores.size() != surface->images.size()) {
		for(auto semaphore: surface->image_available_semaphores)
			vkDestroySemaphore(queue->device, semaphore, queue->callbacks);

		surface->image_available_semaphores.resize(surface->images.size());
		for(auto& semaphore: surface->image_available_semaphores) {
			VkSemaphoreCreateInfo info { info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
			vkCreateSemaphore(queue->device, &info, queue->callbacks, &semaphore);
		}
		surface->semaphore_counter = -1;
	}

	surface->semaphore_counter = (surface->semaphore_counter + 1) % surface->image_available_semaphores.size();
	VK_CHECK(vkAcquireNextImageKHR(queue->device, surface->swapchain->swapchain, UINT64_MAX, surface->image_available_semaphores[surface->semaphore_counter], VK_NULL_HANDLE, &surface->current_image), nullptr);
	
	surface->images[surface->current_image].available_semaphore = surface->image_available_semaphores[surface->semaphore_counter];
	return &surface->images[surface->current_image];
}

void gpuSurfacePresentEXT(GpuQueue* queue, GpuSurface* surface, uint64_t wait_submission_index /*= NO_SUBMISSION_WAIT */) {
	if(wait_submission_index != NO_SUBMISSION_WAIT) {
		if(surface->render_finished_semaphores.size() != surface->images.size()) {
			for(auto semaphore: surface->render_finished_semaphores)
				vkDestroySemaphore(queue->device, semaphore, queue->callbacks);

			surface->render_finished_semaphores.resize(surface->images.size());
			for(auto& semaphore: surface->render_finished_semaphores) {
				VkSemaphoreCreateInfo info { info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
				vkCreateSemaphore(queue->device, &info, queue->callbacks, &semaphore);
			}
		}

		// Launch an empty/no command buffer that waits for the timeline semaphore and then signals the render finished semaphore
		VkSemaphoreSubmitInfo wait {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.semaphore = queue->command_submission_timeline_semaphore,
			.value = wait_submission_index,
			.stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
		};
		VkSemaphoreSubmitInfo signal {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.semaphore = surface->render_finished_semaphores[surface->current_image],
			.stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
		};
		VkSubmitInfo2 info = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
			.waitSemaphoreInfoCount = 1,
			.pWaitSemaphoreInfos = &wait,
			.signalSemaphoreInfoCount = 1,
			.pSignalSemaphoreInfos = &signal
		};
		VK_CHECK(vkQueueSubmit2(queue->queue, 1, &info, VK_NULL_HANDLE), /*nothing*/);
	}

	VkPresentInfoKHR info = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.swapchainCount = 1,
		.pSwapchains = &surface->swapchain->swapchain,
		.pImageIndices = &surface->current_image,
	};
	if(wait_submission_index != NO_SUBMISSION_WAIT) {
		info.waitSemaphoreCount = 1;
		info.pWaitSemaphores = &surface->render_finished_semaphores[surface->current_image];
	}
	VK_CHECK(vkQueuePresentKHR(queue->queue, &info), /*nothing*/);
}