#define VOLK_IMPLEMENTATION
#include <volk.h>

#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
// #define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "noapi.hpp"

#include <VkBootstrap.h>

#include <vulkan/vulkan_core.h> // TODO: Remove when it stops being auto added


std::expected<GpuVulkanDefault, std::string> gpuSetupDefaultVulkan(std::function_ref<VkSurfaceKHR(VkInstance)> surface_loader, std::span<const char*> extra_extensions /* = {} */, std::span<const char*> extra_layers /* = {} */, bool debug /* = true */) {
	GpuVulkanDefault out;

	// Instance
	vkb::InstanceBuilder instance_builder;
	instance_builder.set_app_name("noapi")
		.set_engine_name("noapi")
		.enable_extensions(extra_extensions.size(), extra_extensions.data())
		.request_validation_layers(debug)
		.require_api_version(1, 3, 0);
	for(auto layer: extra_layers)
		instance_builder.enable_layer(layer);
	if(debug) instance_builder.use_default_debug_messenger();
	auto inst = instance_builder.build();
	if (!inst) return std::unexpected(inst.error().message());
	auto instance = inst.value();
	out.instance = instance.instance;
	out.messenger = instance.debug_messenger;

	out.surface = surface_loader(out.instance);

	// Physical Device
	vkb::PhysicalDeviceSelector gpu_selector{instance};
	auto phys = gpu_selector
		.set_required_features(gpuEnableRequiredVulkanFeatures({}))
		.set_required_features_12(gpuEnableRequiredVulkan12Features({}))
		.set_required_features_13(gpuEnableRequiredVulkan13Features({}))
		.set_surface(out.surface)
		.set_minimum_version(1, 3)
		.select();
	if (!phys) return std::unexpected(phys.error().message());
	auto gpu = phys.value();
	out.gpu = gpu.physical_device;

	// Logical Device
	vkb::DeviceBuilder device_builder{gpu};
	auto dev = device_builder.build();
	if (!dev) return std::unexpected(dev.error().message());
	auto device = dev.value();
	out.device = device.device;

	out.graphics_queue = device.get_queue(vkb::QueueType::graphics).value();
	out.graphics_queue_family = device.get_queue_index(vkb::QueueType::graphics).value();

	return out;
}

std::optional<GpuQueue> gpuCreateQueue(VkInstance instance, VkPhysicalDevice gpu, VkDevice device, VkQueue queue /* = VK_NULL_HANDLE */, uint32_t queue_family /* = -1 */, VkAllocationCallbacks* callbacks /* = nullptr */, bool is_graphics_queue /* = true */) {
	GpuQueue out {
		.gpu = gpu,
		.device = device,
		.queue = queue,
		.queue_family = queue_family,
		.is_graphics_queue = is_graphics_queue,
		.callbacks = callbacks
	};

	if(volkInitialize() != VK_SUCCESS)
		return {};
	volkLoadInstance(instance);
	volkLoadDevice(out.device);

	// Find queue if one wasn't already provided
	if(!out.queue) {
		uint32_t count;
		vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
		std::vector<VkQueueFamilyProperties> families(count);
		vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, families.data());

		vkb::Device device;
		device.device = out.device;
		device.queue_families = std::move(families);

		if(auto r = device.get_queue_index(is_graphics_queue ? vkb::QueueType::graphics : vkb::QueueType::compute); r.has_value())
			out.queue_family = r.value();
		else return {};
		vkGetDeviceQueue(out.device, out.queue_family, 0, &out.queue);
	}

	// VMA
	VmaVulkanFunctions functions = {
		.vkGetInstanceProcAddr = vkGetInstanceProcAddr,
		.vkGetDeviceProcAddr = vkGetDeviceProcAddr,
	};

	VmaAllocatorCreateInfo vma_info = {
		.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
		.physicalDevice = out.gpu,
		.device = out.device,
		.pAllocationCallbacks = out.callbacks,
		.pVulkanFunctions = &functions,
		.instance = instance,
		.vulkanApiVersion = VK_API_VERSION_1_3,
	};
	if(vmaCreateAllocator(&vma_info, &out.allocator) != VK_SUCCESS)
		return {};

	out.command_submission_timeline_semaphore = gpuCreateSemaphore(out, 0).semaphore;

	return out;
}

void gpuDestroyQueue(GpuQueue& queue) {
	if(queue.command_pool)
		vkDestroyCommandPool(queue.device, queue.command_pool, queue.callbacks);
	if(queue.command_submission_timeline_semaphore)
		vkDestroySemaphore(queue.device, queue.command_submission_timeline_semaphore, queue.callbacks);
	if(queue.pipeline_layout)
		vkDestroyPipelineLayout(queue.device, queue.pipeline_layout, queue.callbacks);

	if(queue.allocator)
		vmaDestroyAllocator(queue.allocator);
}

GpuCommandBuffer gpuStartCommandRecording(GpuQueue& queue) {
	if(!queue.command_pool) {
		VkCommandPoolCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
			.queueFamilyIndex = queue.queue_family
		};
		VK_CHECK(vkCreateCommandPool(queue.device, &info, queue.callbacks, &queue.command_pool));
	}

	// Check the current value of the submission count timeline semaphore and free any buffers who's submission has finished
	if(!queue.command_buffers_pending_free.empty()) {
		uint64_t current_finished_submission;
		vkGetSemaphoreCounterValue(queue.device, queue.command_submission_timeline_semaphore, &current_finished_submission);

		// TODO: Would it be worth the effort to deduplicate?

		std::vector<VkCommandBuffer> to_free; to_free.reserve(queue.command_buffers_pending_free.size());
		for(auto [cmd, submit]: queue.command_buffers_pending_free)
			if(submit <= current_finished_submission)
				to_free.push_back(cmd);

		if(!to_free.empty())
			vkFreeCommandBuffers(queue.device, queue.command_pool, to_free.size(), to_free.data());
	}

	GpuCommandBuffer out {.queue = &queue};
	VkCommandBufferAllocateInfo info {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = queue.command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1
	};
	VK_CHECK(vkAllocateCommandBuffers(queue.device, &info, &out.command_buffer));

	VkCommandBufferBeginInfo begin {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};
	VK_CHECK(vkBeginCommandBuffer(out.command_buffer, &begin));
	return out;
}

void gpuDestoryCommandBuffer(GpuCommandBuffer& cmd) {
	vkFreeCommandBuffers(cmd.queue->device, cmd.queue->command_pool, 1, &cmd.command_buffer);
}

void gpuSubmitNoDestroy(GpuQueue& queue, std::span<GpuCommandBuffer> commandBuffers, std::optional<GpuSemaphore&> semaphore /* = std::nullopt */, uint64_t signalValue /* = 0 */) {
	std::vector<VkCommandBufferSubmitInfo> submits; submits.reserve(commandBuffers.size());
	for(auto& cmd: commandBuffers) {
		if(!cmd.ended) {
			vkEndCommandBuffer(cmd.command_buffer);
			cmd.ended = true;
		}
		submits.emplace_back(VkCommandBufferSubmitInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
			.commandBuffer = cmd.command_buffer
		});
	}

	std::array<VkSemaphoreSubmitInfo, 2> signals {
		VkSemaphoreSubmitInfo{
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.semaphore = queue.command_submission_timeline_semaphore,
			.value = queue.command_submission_timeline_semaphore_next_value++,
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
		.commandBufferInfoCount = static_cast<uint32_t>(submits.size()),
		.pCommandBufferInfos = submits.data(),
		.signalSemaphoreInfoCount = static_cast<uint32_t>(semaphore ? 2 : 1),
		.pSignalSemaphoreInfos = signals.data(),
	};
	VK_CHECK(vkQueueSubmit2(queue.queue, 1, &info, nullptr));
}

void gpuSubmit(GpuQueue& queue, std::span<GpuCommandBuffer> commandBuffers, std::optional<GpuSemaphore&> semaphore /* = std::nullopt */, uint64_t signalValue /* = 0 */) {
	gpuSubmitNoDestroy(queue, commandBuffers, semaphore, signalValue);

	for(auto& cmd: commandBuffers)
		queue.command_buffers_pending_free.emplace_back(cmd.command_buffer, queue.command_submission_timeline_semaphore_next_value - 1); // -1 since it was incremented in the no destroy call
}

GpuSemaphore gpuCreateSemaphore(GpuQueue& queue, uint64_t initValue) {
	GpuSemaphore out;

	VkSemaphoreTypeCreateInfo semaphore_type{};
	semaphore_type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
	semaphore_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	semaphore_type.initialValue = initValue; // Starting timeline value

	VkSemaphoreCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	info.pNext = &semaphore_type;
	VK_CHECK(vkCreateSemaphore(queue.device, &info, queue.callbacks, &out.semaphore));
	return out;
}

uint64_t gpuWaitSemaphore(GpuQueue& queue, GpuSemaphore& semaphore, uint64_t value, uint64_t timeout /* = UINT64_MAX */) {
	if(value == GPU_GET_VALUE) {
		uint64_t currentValue;
		VK_CHECK(vkGetSemaphoreCounterValue(queue.device, semaphore.semaphore, &currentValue));
		return currentValue;
	}

	VkSemaphoreWaitInfo waitInfo{};
	waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
	waitInfo.flags = 0;
	waitInfo.semaphoreCount = 1;
	waitInfo.pSemaphores = &semaphore.semaphore;
	waitInfo.pValues = &value;

	VK_CHECK(vkWaitSemaphores(queue.device, &waitInfo, timeout));
	return value;
}

void gpuDestroySemaphore(GpuQueue& queue, GpuSemaphore& semaphore) {
	vkDestroySemaphore(queue.device, semaphore.semaphore, queue.callbacks);
}