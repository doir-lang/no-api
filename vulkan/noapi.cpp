
#define VOLK_IMPLEMENTATION
#include <volk.h>

#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
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

std::optional<GpuQueue> gpuCreateQueue(VkInstance instance, VkPhysicalDevice gpu, VkDevice device, VkQueue queue /* = VK_NULL_HANDLE */, uint32_t queue_family /* = -1 */, bool is_graphics_queue /* = true */) {
	GpuQueue out {
		.gpu = gpu,
		.device = device,
		.queue = queue,
		.queue_family = queue_family,
		.is_graphics_queue = is_graphics_queue
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
		.physicalDevice = out.gpu,
		.device = out.device,
		.pVulkanFunctions = &functions,
		.instance = instance,
		.vulkanApiVersion = VK_API_VERSION_1_3,
	};
	if(vmaCreateAllocator(&vma_info, &out.allocator) != VK_SUCCESS)
		return {};

	return out;
}