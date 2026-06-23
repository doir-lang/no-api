#define VOLK_IMPLEMENTATION
#include <volk.h>

#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "noapi.hpp"

#include <VkBootstrap.h>

#include <vulkan/vulkan_core.h> // TODO: Remove when it stops being auto added

#ifndef NOAPI_NO_EXCEPTIONS
	#define VK_CHECK(expr) do {\
		auto res = expr;\
		if(res != VK_SUCCESS)\
			throw VkResultExecption(res);\
	} while(false)
#else
	#define VK_CHECK(expr) do {\
		auto res = expr;\
		if(res != VK_SUCCESS)\
			std::exit((int)res);\
	} while(false)
#endif

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

	{
		VkSemaphoreTypeCreateInfo semaphore_type{};
		semaphore_type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
		semaphore_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
		semaphore_type.initialValue = 0; // Starting timeline value

		VkSemaphoreCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		info.pNext = &semaphore_type;

		if(vkCreateSemaphore(out.device, &info, out.callbacks, &out.command_submission_timeline_semaphore) != VK_SUCCESS)
			return {};
	}

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
	VkSubmitInfo2 info {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
		.commandBufferInfoCount = static_cast<uint32_t>(submits.size()),
		.pCommandBufferInfos = submits.data(),
		.signalSemaphoreInfoCount = 1,
		.pSignalSemaphoreInfos = signals.data(),
	};
	VK_CHECK(vkQueueSubmit2(queue.queue, 1, &info, nullptr));
}

void gpuSubmit(GpuQueue& queue, std::span<GpuCommandBuffer> commandBuffers, std::optional<GpuSemaphore&> semaphore /* = std::nullopt */, uint64_t signalValue /* = 0 */) {
	gpuSubmitNoDestroy(queue, commandBuffers, semaphore, signalValue);

	for(auto& cmd: commandBuffers)
		queue.command_buffers_pending_free.emplace_back(cmd.command_buffer, queue.command_submission_timeline_semaphore_next_value - 1); // -1 since it was incremented in the no destroy call
}

void gpuMemCpy(GpuCommandBuffer& cmd, gpu* dest_, gpu* src_, size_t bytes, bool no_offsets /* = false*/) {
	auto& queue = *cmd.queue;
	auto dest = (VkDeviceAddress)dest_, src = (VkDeviceAddress)src_;
	VkDeviceAddress closest_dest = 0, closest_src = 0; // TODO: There are probably edge cases around setting these to zero!
	if(no_offsets) {
		closest_dest = dest;
		closest_src = src;
	} else for(auto [key, _]: queue.allocations) {
		if(closest_dest - dest > key - dest)
			closest_dest = key;
		if(closest_src - src > key - src)
			closest_src = key;
	}
	VkBuffer dest_buffer = queue.allocations[closest_dest].first;
	size_t dest_offset = closest_dest - dest;
	VkBuffer src_buffer = queue.allocations[closest_src].first;
	size_t src_offset = closest_src - src;

	VkBufferCopy region {
		.srcOffset = src_offset,
		.dstOffset = dest_offset,
		.size = bytes
	};
	vkCmdCopyBuffer(cmd.command_buffer, src_buffer, dest_buffer, 1, &region);
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