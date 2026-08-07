#define VOLK_IMPLEMENTATION
#include <volk.h>

#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
// #define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "noapi.hpp"

#include <VkBootstrap.h>

#include <vulkan/vulkan_core.h> // TODO: Remove when it stops being auto added

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

GpuQueue* gpuCreateQueue(VkInstance instance, VkPhysicalDevice gpu, VkDevice device, VkQueue queue /* = VK_NULL_HANDLE */, uint32_t queue_family /* = -1 */, bool is_graphics_queue /* = true */, GpuAllocatorFunc allocator /* = default_::gpu_allocator */, VkAllocationCallbacks* callbacks /* = nullptr */) {
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

void gpuDestoryCommandBuffer(GpuCommandBuffer& cmd) {
	vkFreeCommandBuffers(cmd.queue->device, cmd.queue->command_pool, 1, &cmd.command_buffer);
}

uint64_t gpuSubmitNoDestroy(GpuQueue* queue, std::span<GpuCommandBuffer*> commandBuffers, GpuSemaphore* semaphore /* = nullptr */, uint64_t signalValue /* = 0 */) {
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
	auto submission_index = gpuSubmitNoDestroy(queue, commandBuffers, semaphore, signalValue);

	for(auto& cmd: commandBuffers)
		queue->command_buffers_pending_free.emplace_back(cmd->command_buffer, submission_index);
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