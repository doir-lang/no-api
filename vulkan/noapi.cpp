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
		.set_required_features_12(gpuEnableRequiredVulkan12Features({}))
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
		.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
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
	if(vmaCreateBufferWithAlignment(queue.allocator, &buffer_info, &alloc_info, align, &buffer, &allocation, nullptr) != VK_SUCCESS)
		return nullptr;

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