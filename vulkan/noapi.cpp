#define VOLK_IMPLEMENTATION
#include <volk.h>

#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "noapi.hpp"
#include "../slang_compiler.hpp"

#include <VkBootstrap.h>
#include <vulkan/vulkan_core.h> // TODO: Remove when it stops being auto added

// gpuBlitTextureEXT compiles its (tiny, fixed) GLSL at runtime rather than shipping a SPIR-V blob

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <sys/types.h>
#include <utility>
#include <vector>

// Returns RETURN from the enclosing function, with the VkResult in errno, when a Vulkan call fails
#define VK_CHECK(expr, RETURN) do {\
	auto res = expr;\
	if(res != VK_SUCCESS) {\
		errno = res;\
		return RETURN;\
	}\
} while(false)


// What each debug messenger reports to. The messenger's destruction isn't seen here, so targets are kept for the program's life; a setup adds a few bytes.
struct DebugTarget {
	GpuErrorCallbackEXT callback;
	VkDebugUtilsMessageSeverityFlagBitsEXT severity_filter;
};

static DebugTarget* add_debug_target(GpuErrorCallbackEXT callback, VkDebugUtilsMessageSeverityFlagBitsEXT severity_filter) {
	static std::mutex mutex;
	static std::deque<DebugTarget> targets; // A deque never moves what it already holds
	std::lock_guard lock(mutex);
	return &targets.emplace_back(callback, severity_filter);
}

void gpuDefaultErrorCallbackEXT(void* queue, int type, GpuStringView message) {
	auto mt = vkb::to_string_message_type(type);
	printf("[%s]\n%.*s\n", mt, (int)message.count, message.ptr);
}

// Copies why the setup gave up into the caller's buffer (which is allowed to be absent) and reports the failure
static bool setup_failed(char* out_error, size_t out_error_capacity, const std::string& message) {
	if(out_error && out_error_capacity) {
		auto length = std::min(message.size(), out_error_capacity - 1);
		memcpy(out_error, message.data(), length);
		out_error[length] = '\0';
	}
	return false;
}

bool gpuSetupDefaultVulkanEXT(GpuVulkanSurfaceLoaderEXT surface_loader, void* surface_loader_userdata,
	GpuErrorCallbackEXT error_callback, VkDebugUtilsMessageSeverityFlagBitsEXT severity_filter_set,
	GpuCStringSpan instance_extensions, GpuCStringSpan extra_layers, GpuCStringSpan device_extensions, bool debug,
	GpuVulkanDefault* out_default, char* out_error, size_t out_error_capacity
) {
	GpuVulkanDefault out;

	// Instance
	vkb::InstanceBuilder instance_builder;
	instance_builder.set_app_name("NoAPI")
		.set_engine_name("NoAPI")
		.enable_extensions(instance_extensions.size(), instance_extensions.data())
		.request_validation_layers(debug)
		.require_api_version(1, 4, 0);
	// Optional, and checked through vkb::SystemInfo because volk has not loaded instance entry points yet; naming an absent extension fails instance creation.
	if(auto system = vkb::SystemInfo::get_system_info(); system)
		// Both, and only together: VK_KHR_surface_maintenance1 depends on the capabilities query, and naming it without that dependency is what an instance is rejected for
		if(system->is_extension_available("VK_KHR_surface_maintenance1")
				&& system->is_extension_available("VK_KHR_get_surface_capabilities2")) {
			instance_builder.enable_extension("VK_KHR_get_surface_capabilities2");
			instance_builder.enable_extension("VK_KHR_surface_maintenance1");
		}
	for(auto layer: extra_layers)
		instance_builder.enable_layer(layer);
	if(debug) instance_builder.set_debug_callback(+[](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageTypes, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) -> VkBool32 {
			auto target = (const DebugTarget*)pUserData;
			if(messageSeverity < target->severity_filter) return VK_FALSE;

			target->callback(nullptr, messageTypes, {pCallbackData->pMessage, strlen(pCallbackData->pMessage)});
			return VK_FALSE;
		}).set_debug_callback_user_data_pointer(add_debug_target(error_callback, severity_filter_set));
	auto inst = instance_builder.build();
	if (!inst) return setup_failed(out_error, out_error_capacity, inst.error().message());
	auto instance = inst.value();
	out.instance = instance.instance;
	out.messenger = instance.debug_messenger;

	out.surface = surface_loader ? surface_loader(out.instance, surface_loader_userdata) : VK_NULL_HANDLE;

	auto required = gpuRequiredVulkanDeviceExtensionsEXT();
	std::vector<const char*> extensions(required.begin(), required.end());
	extensions.insert(extensions.end(), device_extensions.begin(), device_extensions.end());

	// Physical Device
	vkb::PhysicalDeviceSelector gpu_selector{instance};
	// Without this vk-bootstrap rejects a null surface instead of skipping the presentation check
	if(!out.surface) gpu_selector.defer_surface_initialization();
	// Nothing outlives a failure, as the header promises
	auto fail = [&](const std::string& message) {
		if(out.surface) vkb::destroy_surface(instance, out.surface);
		vkb::destroy_instance(instance);
		return setup_failed(out_error, out_error_capacity, message);
	};

	auto phys = gpu_selector
		.set_required_features(gpuEnableRequiredVulkanFeaturesEXT({}))
		.set_required_features_11(gpuEnableRequiredVulkan11FeaturesEXT({}))
		.set_required_features_12(gpuEnableRequiredVulkan12FeaturesEXT({}))
		.set_required_features_13(gpuEnableRequiredVulkan13FeaturesEXT({}))
		.set_required_features_14(gpuEnableRequiredVulkan14FeaturesEXT({}))
		.add_required_extensions(extensions)
		.set_surface(out.surface)
		.set_minimum_version(1, 4)
		.select();
	if (!phys) return fail(phys.error().message());
	auto gpu = phys.value();
	out.gpu = gpu.physical_device;

	// Enabled through the feature check, so an extension whose feature bit is missing isn't turned on half way.
	VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR swapchain_maintenance {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR,
		.swapchainMaintenance1 = true
	};
	if(gpu.enable_extension_features_if_present(swapchain_maintenance))
		gpu.enable_extension_if_present("VK_KHR_swapchain_maintenance1");

	// No taskShader: there is no amplification stage, and requiring one a device lacks would lose the mesh stage too.
	VkPhysicalDeviceMeshShaderFeaturesEXT mesh_shader {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
		.meshShader = true
	};
	if(gpu.enable_extension_features_if_present(mesh_shader))
		gpu.enable_extension_if_present("VK_EXT_mesh_shader");

	// Optional too: only GpuRasterDesc::supportDualSourceBlending needs it
	(void)gpu.enable_features_if_present(VkPhysicalDeviceFeatures{.dualSrcBlend = true});


	// Logical Device
	vkb::DeviceBuilder device_builder{gpu};
	auto dev = device_builder.add_pNext(gpuRequiredVulkanDeviceCreateInfoPnextEXT()).build();
	if (!dev) return fail(dev.error().message());
	auto device = dev.value();
	out.device = device.device;

	out.graphics_queue = device.get_queue(vkb::QueueType::graphics).value();
	out.graphics_queue_family = device.get_queue_index(vkb::QueueType::graphics).value();

	*out_default = out;
	return true;
}

static std::optional<GpuSemaphore> gpuCreateSemaphoreImpl(GpuQueue* queue, uint64_t init_value) {
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

GpuQueue* gpuCreateQueue(VkInstance instance, VkPhysicalDevice gpu, VkDevice device, VkQueue queue, uint32_t queue_family, bool is_graphics_queue, CpuAllocatorFunc allocator, VkAllocationCallbacks* callbacks) {
	// Instance level only, since volkLoadDevice would tie volk's entry points to one device; once per instance, so an entry point a program swapped stays swapped
	if(volkGetLoadedInstance() != instance) {
		VK_CHECK(volkInitialize(), nullptr);
		volkLoadInstance(instance);
	}

	// Find queue if one wasn't already provided; vk-bootstrap only names a compute queue in a family of its own
	if(!queue) {
		uint32_t count;
		vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
		std::vector<VkQueueFamilyProperties> families(count);
		vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, families.data());

		vkb::Device lookup;
		lookup.device = device;
		lookup.queue_families = std::move(families);

		auto found = lookup.get_queue_index(is_graphics_queue ? vkb::QueueType::graphics : vkb::QueueType::compute);
		if(!found.has_value()) {
			errno = VK_ERROR_FEATURE_NOT_PRESENT;
			return nullptr;
		}
		queue_family = found.value();
		vkGetDeviceQueue(device, queue_family, 0, &queue);
	}

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
		.vulkanApiVersion = VK_API_VERSION_1_4,
	};
	VK_CHECK(vmaCreateAllocator(&vma_info, &out->gpu_allocator), nullptr);

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
	out->image_size = heap_properties.imageDescriptorSize;
	out->descriptor_heap_alignment = heap_properties.resourceHeapAlignment;

	// Probed, because the heap usage bit can move a buffer to other memory types (it does on Intel/Mesa).
	{
		constexpr static VkBufferUsageFlags base = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
			| VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
			| VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

		constexpr static auto requirements_for = [](VkDevice device, VkBufferUsageFlags usage) {
			VkBufferCreateInfo buffer_info {
				.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
				.size = 64 * 1024, // Any size works; the answer is about usage, not extent
				.usage = usage
			};
			VkDeviceBufferMemoryRequirements query {
				.sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS,
				.pCreateInfo = &buffer_info
			};
			VkMemoryRequirements2 out { .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
			vkGetDeviceBufferMemoryRequirements(device, &query, &out);
			return out.memoryRequirements;
		};

		auto without = requirements_for(out->device, base);
		auto with = requirements_for(out->device, base | VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT);
		out->capabilities.descriptor_heap_in_any_memory =
			with.memoryTypeBits == without.memoryTypeBits && with.alignment == without.alignment;
	}

	// Int64 atomics make SIGNAL_ATOMIC_MAX and OR. Only the physical device can be asked, so this trusts gpuEnableRequiredVulkan12FeaturesEXT was used.
	{
		VkPhysicalDeviceVulkan12Features features12 { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
		VkPhysicalDeviceFeatures2 features { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &features12 };
		vkGetPhysicalDeviceFeatures2(gpu, &features);
		out->capabilities.split_barrier_signals = features12.shaderBufferInt64Atomics;
		// Read from the physical device as well, so this trusts the VkDevice enabled dualSrcBlend, as gpuSetupDefaultVulkanEXT does where it can.
		out->capabilities.dual_source_blending = features.features.dualSrcBlend;
	}

	// Read from the physical device, so this trusts the VkDevice enabled gpuOptionalVulkanDeviceExtensionsEXT, as gpuSetupDefaultVulkanEXT does.
	{
		VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR maintenance {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR
		};
		VkPhysicalDeviceFeatures2 features { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &maintenance };
		vkGetPhysicalDeviceFeatures2(gpu, &features);
		out->present_fences_supported = maintenance.swapchainMaintenance1;
	}

	// Both halves: the feature says it could be enabled, and the device handing out the command says it actually enabled VK_EXT_mesh_shader.
	{
		VkPhysicalDeviceMeshShaderFeaturesEXT mesh {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT
		};
		VkPhysicalDeviceFeatures2 features { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &mesh };
		vkGetPhysicalDeviceFeatures2(gpu, &features);
		out->capabilities.mesh_shaders = mesh.meshShader && vkGetDeviceProcAddr(device, "vkCmdDrawMeshTasksEXT");
	}

	// Events give Vulkan a real split barrier, so a paired wait need not stall the work between the halves.
	out->capabilities.split_barriers = true;
	out->capabilities.gpu_writable_texture_heap = true;

	out->command_submission_timeline_semaphore = gpuCreateSemaphoreImpl(out, 0)->semaphore;

	// A driver's image descriptor is 64 bytes or a divisor of it; shaders multiply heap indices by the ratio.
	assert(heap_properties.imageDescriptorSize <= sizeof(GpuTextureDescriptor)
		&& "This driver's image descriptors are wider than GpuTextureDescriptor");
	assert(sizeof(GpuTextureDescriptor) % heap_properties.imageDescriptorSize == 0
		&& "This driver's image descriptor size doesn't divide GpuTextureDescriptor's");
	assert(heap_properties.imageDescriptorAlignment <= alignof(GpuTextureDescriptor)
		&& "This driver wants image descriptors aligned more strictly than GpuTextureDescriptor is");
	if(heap_properties.imageDescriptorSize == 0
		|| heap_properties.imageDescriptorSize > sizeof(GpuTextureDescriptor)
		|| sizeof(GpuTextureDescriptor) % heap_properties.imageDescriptorSize != 0) return {};

	out->heap_stride_ratio = static_cast<uint32_t>(sizeof(GpuTextureDescriptor) / heap_properties.imageDescriptorSize);
	out->capabilities.texture_descriptor_stride_ratio = out->heap_stride_ratio;

	return out;
}

GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, VkSurfaceKHR surface, const GpuSurfaceDescriptor& desc) {
	if(!surface) {
		errno = VK_ERROR_SURFACE_LOST_KHR;
		return nullptr;
	}

	auto out = (GpuSurface*)queue->cpu_allocator(nullptr, sizeof(GpuSurface));
	new(out) GpuSurface {
		.surface = surface,
	};

	gpuSurfaceReconfigureEXT(queue, out, desc);
	return out;
}

void gpuFreeQueue(GpuQueue* queue) {
	(void)vkDeviceWaitIdle(queue->device); // Fails only with the device lost, when destroying everything is all that is left

	if(queue->command_pool)
		vkDestroyCommandPool(queue->device, queue->command_pool, queue->callbacks);
	if(queue->command_submission_timeline_semaphore)
		vkDestroySemaphore(queue->device, queue->command_submission_timeline_semaphore, queue->callbacks);

	for(auto [_, buffer]: queue->sampler_cache)
		gpuFree(queue, (gpu*)buffer);

	for(auto [_view, _submit]: queue->views_in_flight)
		vkDestroyImageView(queue->device, _view, queue->callbacks);
	for(auto [_key, pipeline]: queue->blit_pipelines)
		vkDestroyPipeline(queue->device, pipeline, queue->callbacks);
	for(auto pool: queue->blit_descriptor_pools)
		vkDestroyDescriptorPool(queue->device, pool, queue->callbacks);
	for(auto sampler: queue->blit_samplers)
		if(sampler) vkDestroySampler(queue->device, sampler, queue->callbacks);
	for(auto module: queue->blit_shader_modules)
		if(module) vkDestroyShaderModule(queue->device, module, queue->callbacks);
	if(queue->blit_pipeline_layout)
		vkDestroyPipelineLayout(queue->device, queue->blit_pipeline_layout, queue->callbacks);
	if(queue->blit_descriptor_set_layout)
		vkDestroyDescriptorSetLayout(queue->device, queue->blit_descriptor_set_layout, queue->callbacks);

	if(queue->signal_pipeline)
		vkDestroyPipeline(queue->device, queue->signal_pipeline, queue->callbacks);
	if(queue->signal_pipeline_layout)
		vkDestroyPipelineLayout(queue->device, queue->signal_pipeline_layout, queue->callbacks);
	if(queue->signal_shader_module)
		vkDestroyShaderModule(queue->device, queue->signal_shader_module, queue->callbacks);
	for(auto event: queue->events_free)
		vkDestroyEvent(queue->device, event, queue->callbacks);
	for(auto [event, _submit]: queue->events_in_flight)
		vkDestroyEvent(queue->device, event, queue->callbacks);

	// Free what the program did not, before VMA asserts on live blocks; that takes the textures and heap snapshots made on each allocation with it.
	std::vector<VkDeviceAddress> leftovers;
	for(auto& [address, _allocation]: queue->allocations) leftovers.push_back(address);
	for(auto address: leftovers) gpuFreeDevicePointerEXT(queue, (gpu*)address);

	if(queue->gpu_allocator)
		vmaDestroyAllocator(queue->gpu_allocator);

	auto allocator = queue->cpu_allocator;
	queue->~GpuQueue(); // Get all of the caches to free their memory
	allocator(queue, 0);
}

GpuCapabilities gpuGetCapabilitiesEXT(const GpuQueue* queue) {
	return queue->capabilities;
}

void gpuSetDiagnosticCallbackEXT(GpuQueue* queue, GpuDiagnosticCallbackEXT callback, void* userdata) {
	queue->diagnostic_callback = callback;
	queue->diagnostic_userdata = userdata;
	// A new destination has not been told any of this yet, so let it hear each cause once too
	queue->reported_diagnostics.clear();
}

void* gpuMalloc(GpuQueue* queue, size_t bytes, size_t align, MEMORY memory) {
	// Memory has no declared purpose, so every buffer may hold indices or draw arguments; neither bit changes alignment or memory types on any driver checked.
	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		| VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

	// Descriptor heap usage can move a buffer to other memory types (see the probe in gpuCreateQueue), so only what asks gets it.
	const bool descriptor_heap = memory == MEMORY_DESCRIPTOR_HEAP || queue->capabilities.descriptor_heap_in_any_memory;
	if(descriptor_heap) usage |= VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT;

	VkBufferCreateInfo buffer_info {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = bytes,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE // TODO: Should be concurrent?
	};
	VmaAllocationCreateInfo alloc_info {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
	};
	if( !(memory == MEMORY_GPU || memory == MEMORY_TEXTURE) ) alloc_info.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
	// A descriptor heap is written by the CPU the same way MEMORY_DEFAULT is
	if(memory == MEMORY_DEFAULT || memory == MEMORY_DESCRIPTOR_HEAP) alloc_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	if(memory == MEMORY_READBACK || memory == MEMORY_TEXTURE_READBACK) alloc_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	if(memory == MEMORY_TEXTURE || memory == MEMORY_TEXTURE_READBACK) alloc_info.flags |= VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT; // We can create a texture that is aliased with the buffer
	VkBuffer buffer;
	VmaAllocation allocation;
	// A heap has to start where the hardware can address one from, which is a stricter alignment than a caller would think to ask for
	if(descriptor_heap) align = std::max<size_t>(align, queue->descriptor_heap_alignment);
	VK_CHECK(vmaCreateBufferWithAlignment(queue->gpu_allocator, &buffer_info, &alloc_info, align, &buffer, &allocation, nullptr), nullptr);

	VkBufferDeviceAddressInfo address_info {
		.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.buffer = buffer
	};
	auto gpu_ptr = vkGetBufferDeviceAddress(queue->device, &address_info);

	queue->allocations[gpu_ptr] = {buffer, allocation, bytes, descriptor_heap};

	if(memory == MEMORY_GPU || memory == MEMORY_TEXTURE)
		return (void*)gpu_ptr;

	void* cpu_ptr = allocation->GetMappedData();
	queue->host2gpu[cpu_ptr] = gpu_ptr;
	queue->gpu2host[gpu_ptr] = cpu_ptr;
	return cpu_ptr;
}

void gpuFree(GpuQueue* queue, void* ptr) {
	auto found = queue->host2gpu.find(ptr);
	gpuFreeDevicePointerEXT(queue, found != queue->host2gpu.end() ? (gpu*)found->second : (gpu*)ptr);
}

void gpuFreeDevicePointerEXT(GpuQueue* queue, gpu* ptr) {
	auto gpu_ptr = (VkDeviceAddress)ptr;
	if(!queue->allocations.contains(gpu_ptr)) return;

	auto found = queue->allocations.find(gpu_ptr);
	auto [buffer, allocation, _size, _heap] = found->second;
	queue->allocations.erase(found);

	if(queue->gpu2host.contains(gpu_ptr)) {
		auto host = queue->gpu2host[gpu_ptr];
		queue->gpu2host.erase(gpu_ptr);
		queue->host2gpu.erase(host);
	}

	if(auto texture = queue->gpu2texture.find(gpu_ptr); texture != queue->gpu2texture.end()) {
		// A texture freed before any command buffer started is still waiting for its initial transition.
		std::erase(queue->textures_pending_initial_layout, texture->second);
		vkDestroyImage(queue->device, texture->second->image, queue->callbacks);
		queue->cpu_allocator(texture->second, 0);
		queue->gpu2texture.erase(texture);
	}

	// Erased too: addresses are recycled, and a stale entry would hand the next allocation here a destroyed VkBuffer.
	if(auto found = queue->descriptor_heaps.find(gpu_ptr); found != queue->descriptor_heaps.end()) {
		auto [buffer, allocation, _size, _address] = found->second;
		vmaDestroyBuffer(queue->gpu_allocator, buffer, allocation);
		queue->descriptor_heaps.erase(found);
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
	// Vulkan packed formats name components from the high bits down, so the WebGPU names map onto reversed ones (rgb10a2unorm is A2B10G10R10).
	inline VkFormat format2vulkan(FORMAT format) {
		switch(format) {
		case FORMAT_NONE: return VK_FORMAT_UNDEFINED;

		case FORMAT_R8_UNORM: return VK_FORMAT_R8_UNORM;
		case FORMAT_R8_SNORM: return VK_FORMAT_R8_SNORM;
		case FORMAT_R8_UINT: return VK_FORMAT_R8_UINT;
		case FORMAT_R8_SINT: return VK_FORMAT_R8_SINT;

		case FORMAT_R16_UNORM: return VK_FORMAT_R16_UNORM;
		case FORMAT_R16_SNORM: return VK_FORMAT_R16_SNORM;
		case FORMAT_R16_UINT: return VK_FORMAT_R16_UINT;
		case FORMAT_R16_SINT: return VK_FORMAT_R16_SINT;
		case FORMAT_R16_FLOAT: return VK_FORMAT_R16_SFLOAT;
		case FORMAT_RG8_UNORM: return VK_FORMAT_R8G8_UNORM;
		case FORMAT_RG8_SNORM: return VK_FORMAT_R8G8_SNORM;
		case FORMAT_RG8_UINT: return VK_FORMAT_R8G8_UINT;
		case FORMAT_RG8_SINT: return VK_FORMAT_R8G8_SINT;

		case FORMAT_R32_UINT: return VK_FORMAT_R32_UINT;
		case FORMAT_R32_SINT: return VK_FORMAT_R32_SINT;
		case FORMAT_R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
		case FORMAT_RG16_UNORM: return VK_FORMAT_R16G16_UNORM;
		case FORMAT_RG16_SNORM: return VK_FORMAT_R16G16_SNORM;
		case FORMAT_RG16_UINT: return VK_FORMAT_R16G16_UINT;
		case FORMAT_RG16_SINT: return VK_FORMAT_R16G16_SINT;
		case FORMAT_RG16_FLOAT: return VK_FORMAT_R16G16_SFLOAT;
		case FORMAT_RGBA8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
		case FORMAT_RGBA8_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
		case FORMAT_RGBA8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
		case FORMAT_RGBA8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
		case FORMAT_RGBA8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
		case FORMAT_BGRA8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
		case FORMAT_BGRA8_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;

		case FORMAT_RGB10_A2_UINT: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
		case FORMAT_RGB10_A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
		case FORMAT_RG11B10_UFLOAT: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
		case FORMAT_RGB9E5_UFLOAT: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;

		case FORMAT_RG32_UINT: return VK_FORMAT_R32G32_UINT;
		case FORMAT_RG32_SINT: return VK_FORMAT_R32G32_SINT;
		case FORMAT_RG32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
		case FORMAT_RGBA16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
		case FORMAT_RGBA16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
		case FORMAT_RGBA16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
		case FORMAT_RGBA16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
		case FORMAT_RGBA16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;

		case FORMAT_RGBA32_UINT: return VK_FORMAT_R32G32B32A32_UINT;
		case FORMAT_RGBA32_SINT: return VK_FORMAT_R32G32B32A32_SINT;
		case FORMAT_RGBA32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;

		case FORMAT_S8_UINT: return VK_FORMAT_S8_UINT;
		case FORMAT_D16_UNORM: return VK_FORMAT_D16_UNORM;
		// X8_D24 makes depth24plus's promise of at least 24 bits; a driver without it reports D32_SFLOAT support instead.
		case FORMAT_D24_PLUS: return VK_FORMAT_X8_D24_UNORM_PACK32;
		case FORMAT_D24_PLUS_S8_UINT: return VK_FORMAT_D24_UNORM_S8_UINT;
		case FORMAT_D32_FLOAT: return VK_FORMAT_D32_SFLOAT;
		case FORMAT_D32_FLOAT_S8_UINT: return VK_FORMAT_D32_SFLOAT_S8_UINT;
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

	// Views and copies name exactly one aspect: the depth of a combined format, which is what sampling and readback see
	inline VkImageAspectFlags single_aspect(FORMAT format) {
		return gpuFormatIsDepthEXT(format) ? VK_IMAGE_ASPECT_DEPTH_BIT : gpuFormatIsStencilEXT(format) ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
	}

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

		const bool cube = descriptor.type == TEXTURE_CUBE || descriptor.type == TEXTURE_CUBE_ARRAY;
		return VkImageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.flags = VkImageCreateFlags(VK_IMAGE_CREATE_ALIAS_BIT | (cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0)),
			.imageType = type2vulkan(descriptor.type),
			.format = GPU::detail::format2vulkan(descriptor.format),
			.extent = VkExtent3D{descriptor.dimensions.x, descriptor.dimensions.y, descriptor.dimensions.z},
			.mipLevels = descriptor.mipCount,
			.arrayLayers = descriptor.layerCount,
			.samples = GPU::detail::samples2vulkan(descriptor.sampleCount),
			// Optimal: nothing reads these as linear memory, and linear tiling rules out depth formats and the compression MEMORY_TEXTURE promises.
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = GPU::detail::usage2vulkan(descriptor.usage),
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};
	}
}

namespace GPU::detail {
	// Asked first because creating an image the device can't make is invalid usage, not an error vkCreateImage reliably returns.
	inline VkResult image_supported(GpuQueue* queue, const GpuTextureDesc& desc, const VkImageCreateInfo& info) {
		// gpuStoreTexture writes a float4, which an integer image can't take; and asking about VK_FORMAT_UNDEFINED is itself invalid
		if(desc.format == FORMAT_NONE || ((desc.usage & USAGE_STORAGE) && !gpuFormatIsStorableEXT(desc.format))) return VK_ERROR_FORMAT_NOT_SUPPORTED;
		VkImageFormatProperties properties;
		auto result = vkGetPhysicalDeviceImageFormatProperties(queue->gpu, info.format, info.imageType, info.tiling, info.usage, info.flags, &properties);
		if(result != VK_SUCCESS) return result;
		bool fits = info.extent.width <= properties.maxExtent.width && info.extent.height <= properties.maxExtent.height
			&& info.extent.depth <= properties.maxExtent.depth && info.mipLevels <= properties.maxMipLevels
			&& info.arrayLayers <= properties.maxArrayLayers && (info.samples & properties.sampleCounts);
		return fits ? VK_SUCCESS : VK_ERROR_FORMAT_NOT_SUPPORTED;
	}
}

GpuTextureSizeAlign gpuTextureSizeAlign(GpuQueue* queue, const GpuTextureDesc& desc) {
	auto info = GPU::detail::descriptor2vulkan(desc);
	if(auto supported = GPU::detail::image_supported(queue, desc, info); supported != VK_SUCCESS) {
		errno = supported;
		return {};
	}
	VkImage temp;
	VK_CHECK(vkCreateImage(queue->device, &info, queue->callbacks, &temp), {});

	VkMemoryRequirements requirements;
	vkGetImageMemoryRequirements(queue->device, temp, &requirements);
	vkDestroyImage(queue->device, temp, queue->callbacks);

	return {requirements.size, requirements.alignment};
}

GpuTexture* gpuCreateTexture(GpuQueue* queue, const GpuTextureDesc& desc, gpu* memory) {
	if(queue->gpu2texture.contains((VkDeviceAddress)memory)) {
		errno = VK_ERROR_TOO_MANY_OBJECTS;
		return nullptr;
	}

	auto backing = queue->allocations.find((VkDeviceAddress)memory);
	if(backing == queue->allocations.end()) {
		errno = VK_ERROR_INITIALIZATION_FAILED;
		return nullptr;
	}

	auto info = GPU::detail::descriptor2vulkan(desc);
	if(auto supported = GPU::detail::image_supported(queue, desc, info); supported != VK_SUCCESS) {
		errno = supported;
		return nullptr;
	}

	VkImage image;
	VK_CHECK(vkCreateImage(queue->device, &info, queue->callbacks, &image), nullptr);
	// Through VMA: the allocation is a suballocation, and vkBindImageMemory at offset 0 would alias whatever starts the block.
	if(auto bound = vmaBindImageMemory(queue->gpu_allocator, backing->second.allocation, image); bound != VK_SUCCESS) {
		vkDestroyImage(queue->device, image, queue->callbacks);
		errno = bound;
		return nullptr;
	}

	auto out = (GpuTexture*)queue->cpu_allocator(nullptr, sizeof(GpuTexture));
	*out = {.image = image, .descriptor = desc};
	queue->gpu2texture[(VkDeviceAddress)memory] = out;
	queue->textures_pending_initial_layout.push_back(out);
	return out;
}

static GpuTextureDescriptor gpuTextureViewDescriptorImpl(GpuQueue* queue, const GpuTexture* texture, const GpuViewDesc& desc, bool read_only) {
	// The whole slot is offered: the write is rejected if the range is narrower than the driver's descriptor.
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
			.aspectMask = GPU::detail::single_aspect(format),
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

// This backend's half of the shader ABI (see slang_compiler.hpp): addresses are device addresses, roots arrive as push data, and textures index the descriptor heap.
static const char* const NOAPI_BACKEND_MODULE = R"(
module noapi_backend;

#if GPU_COMPUTE
// Matches ComputePipelinePushConstants
public struct GpuPushConstants {
	public uint64_t compute_data;
	public uint64_t sampler_map;
}
#else
// Matches GraphicsPipelinePushConstants
public struct GpuPushConstants {
	public uint64_t vertex_data;
	public uint64_t fragment_data;
	public uint64_t index_data;
	public uint64_t sampler_map;
}
#endif

[[vk::push_constant]] public ConstantBuffer<GpuPushConstants> gpu_pc;

// The portable module's 64 bit address, low word first, is a device address here, so loads and stores are plain dereferences.
static uint64_t gpuBackendAddress(uint2 address) { return (uint64_t(address.y) << 32) | uint64_t(address.x); }
static uint2 gpuBackendWords(uint64_t address) { return uint2(uint(address), uint(address >> 32)); }

public uint gpuBackendLoadU32(uint2 address) { return *((uint*)gpuBackendAddress(address)); }
public void gpuBackendStoreU32(uint2 address, uint value) { *((uint*)gpuBackendAddress(address)) = value; }

#if GPU_COMPUTE
public uint2 gpuBackendRootCompute() { return gpuBackendWords(gpu_pc.compute_data); }
#else
public uint2 gpuBackendRootVertex() { return gpuBackendWords(gpu_pc.vertex_data); }
public uint2 gpuBackendRootFragment() { return gpuBackendWords(gpu_pc.fragment_data); }
public uint2 gpuBackendRootIndex() { return gpuBackendWords(gpu_pc.index_data); }
#endif

// The table ensure_sampler_set built: each packed GpuSamplerDesc's slot, or 0 (the default sampler) when it was never enabled.
public uint gpuBackendSamplerSlot(uint packed) {
	if(gpu_pc.sampler_map == 0) return 0u;
	if(packed > 0x1ffu) return 0u;
	return *((uint*)(gpu_pc.sampler_map + uint64_t(packed) * 4u));
}

// 2D only, and `layer` is ignored: a heap entry is opaque hardware bits, so there is no dimensionality to switch on.
public float4 gpuBackendSample(uint heap_index, uint slot, float3 uv, uint layer, float mip) {
	// The heap's stride is a 64 byte GpuTextureDescriptor, but this indexes in the driver's descriptor size; see heap_stride_ratio.
	Texture2D<float4> texture = ResourceDescriptorHeap[heap_index * GPU_HEAP_STRIDE_RATIO];
	SamplerState sampler = SamplerDescriptorHeap[slot];
	return texture.SampleLevel(sampler, uv.xy, mip);
}

// 2D only, as gpuBackendSample. The unformatted image takes any format, which shaderStorageImageWriteWithoutFormat (see gpuEnableRequiredVulkanFeaturesEXT) allows.
public void gpuBackendStoreTexture(uint heap_index, uint3 texel, uint layer, float4 value) {
	RWTexture2D<float4> texture = ResourceDescriptorHeap[heap_index * GPU_HEAP_STRIDE_RATIO];
	texture[texel.xy] = value;
}
)";

static std::optional<std::string> compile_shader(GpuQueue* queue, GpuByteSpan ir, GPU::shaders::SHADER_STAGE stage) {
	// A mesh stage counts as graphics, so it gets the draw's root pointers that gpuMeshletData reads.
	auto graphics = GPU::shaders::is_graphics(stage);
	std::string error;
	auto code = GPU::shaders::compile(SLANG_SPIRV, NOAPI_BACKEND_MODULE,
		std::string_view((const char*)ir.data(), ir.size()), stage, {
			{"GPU_COMPUTE", graphics ? "0" : "1"},
			{"GPU_GRAPHICS", graphics ? "1" : "0"},
			{"GPU_STORES", "1"},
			// A storage image comes from the same descriptor heap as a sampled one, so every stage can write one.
			{"GPU_TEXTURE_STORES", "1"},
			// A macro rather than a constant because it is device dependent, and the session cache keys on macros.
			{"GPU_HEAP_STRIDE_RATIO", std::to_string(queue->heap_stride_ratio)},
		}, error);

	if(!code) {
		GPU::shaders::report_diagnostic(error);
		errno = VK_ERROR_INITIALIZATION_FAILED;
	}
	return code;
}

GpuPipeline* gpuCreateComputePipeline(GpuQueue* queue, GpuByteSpan computeIR) {
	auto spirv = compile_shader(queue, computeIR, GPU::shaders::SHADER_STAGE::COMPUTE);
	if(!spirv) return nullptr;

	VkShaderModule compute_module = VK_NULL_HANDLE;
	{
		VkShaderModuleCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = spirv->size(),
			.pCode = (const uint32_t*)spirv->data(),
		};
		VK_CHECK(vkCreateShaderModule(queue->device, &info, queue->callbacks, &compute_module), nullptr);
	}

	VkPipeline pipeline = VK_NULL_HANDLE;
	{
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
		auto created = vkCreateComputePipelines(queue->device, nullptr, 1, &info, queue->callbacks, &pipeline);
		vkDestroyShaderModule(queue->device, compute_module, queue->callbacks);
		VK_CHECK(created, nullptr);
	}

	auto out = new(queue->cpu_allocator(nullptr, sizeof(GpuPipeline))) GpuPipeline{};
	out->color_target_count = {}; // Null indicating compute pipeline
	out->pipeline = pipeline;
	return out;
}

void gpuFreePipeline(GpuQueue* queue, GpuPipeline* pipeline) {
	vkDestroyPipeline(queue->device, pipeline->pipeline, queue->callbacks);
	pipeline->~GpuPipeline(); // Its vectors own memory of their own
	queue->cpu_allocator(pipeline, 0);
}

namespace GPU::detail {
	// The last finished submission, or 0 (nothing finished) if the counter can't be read
	inline uint64_t finished_submission(GpuQueue* queue) {
		uint64_t value = 0;
		if(vkGetSemaphoreCounterValue(queue->device, queue->command_submission_timeline_semaphore, &value) != VK_SUCCESS) return 0;
		return value;
	}

	// The base is what the queue's other maps (allocations, gpu2texture, descriptor_heaps) are keyed by.
	struct BufferLocation {
		VkBuffer buffer = VK_NULL_HANDLE; // Null when the address landed outside every allocation
		VkDeviceSize offset = 0; // Byte offset of the address within `buffer`
		VkDeviceAddress base = 0; // Base address of the allocation holding it
		VkDeviceSize size = 0; // Size of that whole allocation
	};

	// upper_bound lands one past the holding allocation, so step back once. An unowned address returns a null buffer rather than inserting an entry.
	inline BufferLocation closest_buffer(GpuQueue* queue, gpu* addr, const GpuQueue::Allocation** out_allocation = nullptr) {
		auto address = (VkDeviceAddress)addr;

		auto after = queue->allocations.upper_bound(address);
		if(after == queue->allocations.begin()) return {}; // Below every allocation
		auto found = std::prev(after);

		const auto& [base, allocation] = *found;
		if(address - base >= allocation.size) return {}; // The nearest one ends before the address

		if(out_allocation) *out_allocation = &allocation;
		return {allocation.buffer, address - base, base, allocation.size};
	}

	inline std::array<BufferLocation, 2> closest_buffer(GpuQueue* queue, gpu* addrA, gpu* addrB) {
		return {closest_buffer(queue, addrA), closest_buffer(queue, addrB)};
	}

	// Slot 0 is always the default sampler, as on WebGPU, so an unenabled description still resolves to something valid.
	static VkDeviceAddress ensure_sampler_set(GpuQueue* queue, std::span<const GpuSamplerDesc> requested) {
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

		std::vector<GpuSamplerDesc> enabled_samplers(requested.size() + 1, GpuSamplerDesc{});
		std::copy(requested.begin(), requested.end(), enabled_samplers.begin() + 1); // +1 means that the default sampler is always enabled

		if(auto found = queue->sampler_cache.find(enabled_samplers); found != queue->sampler_cache.end())
			return found->second;

		// The map is indexed by packed description, so it needs one word per value up to the largest
		constexpr static auto sampler_lookup_size = gpuSamplerDescMaxPackedEXT() + 1;
		uint32_t* sampler_mapping_cpu = gpuMalloc<uint32_t>(queue, sampler_lookup_size);
		if(!sampler_mapping_cpu) return 0;
		std::memset(sampler_mapping_cpu, 0, sampler_lookup_size * sizeof(uint32_t));
		for(size_t i = 0; i < enabled_samplers.size(); ++i)
			sampler_mapping_cpu[gpuSamplerDescPackEXT(enabled_samplers[i])] = i;
		// Kept in a local until the heap below is built: gpuMalloc can rehash sampler_cache, which would leave a reference into it dangling
		auto sampler_mapping = (VkDeviceAddress)gpuHostToDevicePointer(queue, sampler_mapping_cpu);

		auto size = std::max<VkDeviceSize>(queue->sampler_size * enabled_samplers.size(), queue->minimum_descriptor_heap_size);
		auto tmp = gpuMalloc(queue, size);

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
		// One range per sampler: pDescriptors parallels pSamplers, and a single range left every slot past the first unwritten.
		std::vector<VkHostAddressRangeEXT> sampler_ranges; sampler_ranges.reserve(sampler_infos.size());
		for(size_t i = 0; i < sampler_infos.size(); ++i)
			sampler_ranges.emplace_back(VkHostAddressRangeEXT{
				.address = (char*)tmp + i * queue->sampler_size,
				.size = queue->sampler_size,
			});
		VK_CHECK(vkWriteSamplerDescriptorsEXT(queue->device, sampler_infos.size(), sampler_infos.data(), sampler_ranges.data()), 0);

		VkBufferCreateInfo buffer_info {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		};
		VmaAllocationCreateInfo alloc_info {
			.usage = VMA_MEMORY_USAGE_AUTO,
			.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
		};

		auto& [buffer, allocation, heap_size, heap_address] = queue->descriptor_heaps[sampler_mapping];
		heap_size = size;
		VK_CHECK(vmaCreateBuffer(queue->gpu_allocator, &buffer_info, &alloc_info, &buffer, &allocation, nullptr), 0);

		VkBufferDeviceAddressInfo address_info {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = buffer
		};
		heap_address = vkGetBufferDeviceAddress(queue->device, &address_info);

		// A private one shot submission rather than gpuStartCommandRecording: the default set is installed from inside gpuStartCommandRecording, which would otherwise recurse
		VkCommandBuffer copy_cmd = VK_NULL_HANDLE;
		VkCommandBufferAllocateInfo copy_alloc {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = queue->command_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1
		};
		VK_CHECK(vkAllocateCommandBuffers(queue->device, &copy_alloc, &copy_cmd), 0);
		VkCommandBufferBeginInfo copy_begin {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
		};
		VK_CHECK(vkBeginCommandBuffer(copy_cmd, &copy_begin), 0);

		VkBufferCopy region {
			.size = size
		};
		auto staging = GPU::detail::closest_buffer(queue, gpuHostToDevicePointer(queue, tmp));
		vkCmdCopyBuffer(copy_cmd, staging.buffer, buffer, 1, &region);

		VK_CHECK(vkEndCommandBuffer(copy_cmd), 0);
		VkSubmitInfo submit {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &copy_cmd,
		};
		VK_CHECK(vkQueueSubmit(queue->queue, 1, &submit, VK_NULL_HANDLE), 0);
		// Built once per distinct sampler set, so stalling is affordable, and it makes freeing the staging allocation safe.
		VK_CHECK(vkQueueWaitIdle(queue->queue), 0);
		vkFreeCommandBuffers(queue->device, queue->command_pool, 1, &copy_cmd);

		gpuFree(queue, tmp);

		queue->sampler_cache[enabled_samplers] = sampler_mapping;
		return sampler_mapping;
	}

	// Points a command buffer at a sampler set built by ensure_sampler_set
	static void bind_sampler_set(GpuCommandBuffer* cmd, VkDeviceAddress sampler_map) {
		if(!sampler_map) return;

		cmd->sampler_map = sampler_map;
		auto& [_buffer, _allocation, heap_size, heap_address] = cmd->queue->descriptor_heaps[sampler_map];
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

	// This state is dynamic, so a draw is invalid until it is set; recording starts with the default description, which gpuSetDepthStencilState later overrides.
	void apply_depth_stencil_state(GpuCommandBuffer* cmd, const GpuDepthStencilDesc& descriptor) {
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

		const bool depthRead = (descriptor.depthMode & DEPTH_READ) != 0;
		const bool depthWrite = (descriptor.depthMode & DEPTH_WRITE) != 0;

		// Vulkan separates the depth test from depth writes, so either flag enables the test and only DEPTH_WRITE enables writing.
		vkCmdSetDepthTestEnable(cmd->command_buffer, (depthRead || depthWrite) ? VK_TRUE : VK_FALSE);
		vkCmdSetDepthWriteEnable(cmd->command_buffer, depthWrite ? VK_TRUE : VK_FALSE);
		vkCmdSetDepthCompareOp(cmd->command_buffer, op2vulkan(descriptor.depthTest));

		const bool stencilEnabled = descriptor.stencilFront.test != OP_ALWAYS || descriptor.stencilBack.test != OP_ALWAYS
			|| descriptor.stencilFront.failOp != STENCIL_OP_KEEP || descriptor.stencilFront.passOp != STENCIL_OP_KEEP
			|| descriptor.stencilFront.depthFailOp != STENCIL_OP_KEEP
			|| descriptor.stencilBack.failOp != STENCIL_OP_KEEP || descriptor.stencilBack.passOp != STENCIL_OP_KEEP
			|| descriptor.stencilBack.depthFailOp != STENCIL_OP_KEEP;
		vkCmdSetStencilTestEnable(cmd->command_buffer, stencilEnabled ? VK_TRUE : VK_FALSE);

		if (stencilEnabled) {
			vkCmdSetStencilOp(cmd->command_buffer, VK_STENCIL_FACE_FRONT_BIT, stencil2vulkan(descriptor.stencilFront.failOp),
				stencil2vulkan(descriptor.stencilFront.passOp), stencil2vulkan(descriptor.stencilFront.depthFailOp),
				op2vulkan(descriptor.stencilFront.test)
			);
			vkCmdSetStencilOp(cmd->command_buffer, VK_STENCIL_FACE_BACK_BIT, stencil2vulkan(descriptor.stencilBack.failOp),
				stencil2vulkan(descriptor.stencilBack.passOp), stencil2vulkan(descriptor.stencilBack.depthFailOp),
				op2vulkan(descriptor.stencilBack.test)
			);

			vkCmdSetStencilCompareMask(cmd->command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, descriptor.stencilReadMask);
			vkCmdSetStencilWriteMask(cmd->command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, descriptor.stencilWriteMask);

			// Front/back reference values differ in your struct (Stencil::reference is per-face) but vkCmdSetStencilReference takes a face mask too, so two calls if front != back.
			if (descriptor.stencilFront.reference == descriptor.stencilBack.reference) {
				vkCmdSetStencilReference(cmd->command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, descriptor.stencilFront.reference);
			} else {
				vkCmdSetStencilReference(cmd->command_buffer, VK_STENCIL_FACE_FRONT_BIT, descriptor.stencilFront.reference);
				vkCmdSetStencilReference(cmd->command_buffer, VK_STENCIL_FACE_BACK_BIT, descriptor.stencilBack.reference);
			}
		}

		vkCmdSetDepthBias(cmd->command_buffer, descriptor.depthBias, descriptor.depthBiasClamp, descriptor.depthBiasSlopeFactor);
	}
}

inline void transition_image_layout(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout, VkAccessFlags source_access_mask, VkAccessFlags destination_access_mask, VkPipelineStageFlags source_stage, VkPipelineStageFlags destination_stage, uint32_t base_mip_level, uint32_t mip_levels, uint32_t base_slice, uint32_t slices, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
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
			.aspectMask = aspect,
			.baseMipLevel = base_mip_level == ALL_MIPS ? 0 : base_mip_level,
			.levelCount = base_mip_level == ALL_MIPS ? mip_levels : 1,
			.baseArrayLayer = base_slice == ALL_LAYERS ? 0 : base_slice,
			.layerCount = base_slice == ALL_LAYERS ? slices : 1,
		}
	};
	vkCmdPipelineBarrier(cmd, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

namespace GPU::detail {
	// A depth/stencil image described with VK_IMAGE_ASPECT_COLOR_BIT is invalid.
	inline VkImageAspectFlags texture_aspect(FORMAT format) {
		if(!gpuFormatIsDepthStencilEXT(format)) return VK_IMAGE_ASPECT_COLOR_BIT;
		return (gpuFormatIsDepthEXT(format) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0)
			| (gpuFormatIsStencilEXT(format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
	}

	// Tracks the layout on the texture; `discard` gives up the contents, as non-LOAD attachments and never written images allow.
	inline void transition_texture(VkCommandBuffer cmd, const GpuTexture* texture, VkImageLayout new_layout,
			VkAccessFlags source_access_mask, VkAccessFlags destination_access_mask,
			VkPipelineStageFlags source_stage, VkPipelineStageFlags destination_stage,
			uint32_t mip, uint32_t slice, bool discard) {
		auto tracked = const_cast<GpuTexture*>(texture);
		transition_image_layout(cmd, texture->image,
			discard ? VK_IMAGE_LAYOUT_UNDEFINED : tracked->layout, new_layout,
			source_access_mask, destination_access_mask, source_stage, destination_stage,
			mip, texture->descriptor.mipCount, slice, texture->descriptor.layerCount,
			texture_aspect(texture->descriptor.format)
		);
		tracked->layout = new_layout;
	}
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
		uint64_t current_finished_submission = GPU::detail::finished_submission(queue);

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

	// Without a sampler set, sampler_map is null and gpuGetSamplerIndex dereferences it as soon as a shader asks for an unenabled sampler.
	GPU::detail::bind_sampler_set(out, GPU::detail::ensure_sampler_set(queue, {}));

	// Depth and stencil are dynamic state, which a draw cannot proceed without; the default is no test, no write and no stencil.
	GPU::detail::apply_depth_stencil_state(out, GpuDepthStencilDesc{});

	// Shaders reach textures through the heap, which names none, so nothing else moves new textures out of UNDEFINED into the GENERAL their descriptors claim.
	for(auto texture: queue->textures_pending_initial_layout)
		GPU::detail::transition_texture(out->command_buffer, texture, VK_IMAGE_LAYOUT_GENERAL,
			0, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			0, 0, true);
	queue->textures_pending_initial_layout.clear();

	return out;
}

void gpuFreeCommandBuffer(GpuCommandBuffer* cmd) {
	auto queue = cmd->queue; // Read before the destructor runs, since the free below still needs it
	// Submitting hands these over (see gpuSubmitNoFreeEXT), so any left were never set, and can be reused straight away
	queue->events_free.insert(queue->events_free.end(), cmd->events_used.begin(), cmd->events_used.end());
	vkFreeCommandBuffers(queue->device, queue->command_pool, 1, &cmd->command_buffer);
	cmd->~GpuCommandBuffer();
	queue->cpu_allocator(cmd, 0);
}

uint64_t gpuSubmit(GpuQueue* queue, GpuCommandBufferSpan commandBuffers, GpuSemaphore* semaphore, uint64_t signalValue) {
	auto submission_index = gpuSubmitNoFreeEXT(queue, commandBuffers, semaphore, signalValue);

	for(auto& cmd: commandBuffers) {
		queue->command_buffers_pending_free.emplace_back(cmd->command_buffer, submission_index);

		cmd->~GpuCommandBuffer(); // Free the buffer
		queue->cpu_allocator(cmd, 0);
	}
	return submission_index;
}

namespace GPU::detail {
	// Once per distinct message per queue: callers sit in per draw functions, and repeats would bury the first report.
	void report(GpuQueue* queue, GPU_DIAGNOSTIC kind, std::string message) {
		if(!queue) return;
		if(!queue->reported_diagnostics.insert(message).second) return;

		if(queue->diagnostic_callback)
			queue->diagnostic_callback(queue, kind, {message.data(), message.size()}, queue->diagnostic_userdata);
		else
			fprintf(stderr, "[noapi] %s: %s\n",
				kind == GPU_DIAGNOSTIC_UNSUPPORTED ? "unsupported" : "emulated", message.c_str());
	}

	inline void report(GpuCommandBuffer* cmd, GPU_DIAGNOSTIC kind, std::string message) {
		report(cmd ? cmd->queue : nullptr, kind, std::move(message));
	}
}

uint64_t gpuSubmitNoFreeEXT(GpuQueue* queue, GpuCommandBufferSpan commandBuffers, GpuSemaphore* semaphore, uint64_t signalValue) {
	std::vector<VkSemaphoreSubmitInfo> waits;
	std::vector<VkCommandBufferSubmitInfo> submits; submits.reserve(commandBuffers.size());
	for(auto& cmd: commandBuffers) {
		if(cmd->state != GpuCommandBuffer::Ended) {
			assert(cmd->state != GpuCommandBuffer::RecordingRenderPass && "Render pass hasn't been ended!");

			VK_CHECK(vkEndCommandBuffer(cmd->command_buffer), 0);
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

		// Events this recording set become reusable once this submission has run, and not before: resetting one earlier could release a wait still queued behind it
		for(auto event: cmd->events_used)
			queue->events_in_flight.emplace_back(event, queue->command_submission_timeline_semaphore_next_value);
		cmd->events_used.clear();

		// Harmless, since the counter was written and the event is recycled, but the caller's pair was never a pair.
		if(!cmd->pending_signals.empty()) {
			GPU::detail::report(queue, GPU_DIAGNOSTIC_EMULATED,
				"gpuSignalAfter: this command buffer was submitted with a signal no gpuWaitBefore ever paired with, so the signal ordered nothing");
			cmd->pending_signals.clear();
		}
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
	VK_CHECK(vkQueueSubmit2(queue->queue, 1, &info, nullptr), 0);
	return queue->command_submission_timeline_semaphore_next_value++;
}

GpuSemaphore* gpuCreateSemaphore(GpuQueue* queue, uint64_t init_value) {
	if(auto out = gpuCreateSemaphoreImpl(queue, init_value); out) {
		auto ret = (GpuSemaphore*)queue->cpu_allocator(nullptr, sizeof(GpuSemaphore));
		*ret = *out;
		return ret;
	} else return nullptr;
}

uint64_t gpuWaitSemaphore(GpuQueue* queue, const GpuSemaphore* semaphore, uint64_t value, uint64_t timeout) {
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

void gpuMemCpy(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, size_t bytes) {
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_);
	assert(dest.buffer && src.buffer && "Neither end of a copy may be an address outside every allocation");

	VkBufferCopy region {
		.srcOffset = src.offset,
		.dstOffset = dest.offset,
		.size = bytes
	};
	vkCmdCopyBuffer(cmd->command_buffer, src.buffer, dest.buffer, 1, &region);
}

void gpuCopyToTexture(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, GpuTexture* texture) {
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_);
	assert(dest.buffer && src.buffer && "Neither end of a copy may be an address outside every allocation");

	assert(cmd->queue->gpu2texture.contains(dest.base) && cmd->queue->gpu2texture[dest.base] == texture);

	assert(dest.offset == 0); // TODO: Can we relax this restriction?

	// Into the GENERAL every other path expects; discarding, since the copy overwrites it.
	GPU::detail::transition_texture(cmd->command_buffer, texture, VK_IMAGE_LAYOUT_GENERAL,
		0, VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, ALL_LAYERS, true); // Every layer, as the copy covers them all

	VkBufferImageCopy copy {
		.bufferOffset = src.offset,
		// In texels rather than bytes; staging data is one image row per buffer row, as on WebGPU.
		.bufferRowLength = texture->descriptor.dimensions.x,
		.bufferImageHeight = texture->descriptor.dimensions.y,
		.imageSubresource = {
			.aspectMask = GPU::detail::single_aspect(texture->descriptor.format),
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = texture->descriptor.type == TEXTURE_3D ? 1 : texture->descriptor.layerCount,
		},
		.imageOffset = {0, 0, 0},
		.imageExtent = {texture->descriptor.dimensions.x, texture->descriptor.dimensions.y, texture->descriptor.dimensions.z}
	};
	vkCmdCopyBufferToImage(cmd->command_buffer, src.buffer, texture->image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
}

void gpuCopyFromTexture(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, const GpuTexture* texture) {
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_);
	assert(dest.buffer && src.buffer && "Neither end of a copy may be an address outside every allocation");

	assert(cmd->queue->gpu2texture.contains(src.base) && cmd->queue->gpu2texture[src.base] == texture);

	assert(src.offset == 0); // TODO: Can we relax this restriction?

	// Into GENERAL in case it was left elsewhere; not discarding, since the contents are the point of a readback.
	GPU::detail::transition_texture(cmd->command_buffer, texture, VK_IMAGE_LAYOUT_GENERAL,
		VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, ALL_LAYERS, false);

	VkBufferImageCopy copy {
		// The texture is the source here, so the buffer half of the copy is the destination
		.bufferOffset = dest.offset,
		// In texels, not bytes; see gpuCopyToTexture
		.bufferRowLength = texture->descriptor.dimensions.x,
		.bufferImageHeight = texture->descriptor.dimensions.y,
		.imageSubresource = {
			.aspectMask = GPU::detail::single_aspect(texture->descriptor.format),
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = texture->descriptor.type == TEXTURE_3D ? 1 : texture->descriptor.layerCount,
		},
		.imageOffset = {0, 0, 0},
		.imageExtent = {texture->descriptor.dimensions.x, texture->descriptor.dimensions.y, texture->descriptor.dimensions.z}
	};
	vkCmdCopyImageToBuffer(cmd->command_buffer, texture->image, VK_IMAGE_LAYOUT_GENERAL, dest.buffer, 1, &copy);
}

namespace GPU::detail {
	inline uvec2 mip_extent(const GpuTexture* texture, uint32_t mip) {
		return {
			std::max(texture->descriptor.dimensions.x >> mip, 1u),
			std::max(texture->descriptor.dimensions.y >> mip, 1u)
		};
	}

	// Compiled with the API's own sessions, so it imports nothing and binds its source as an ordinary descriptor set.
	constexpr static std::string_view BLIT_SHADER = R"slang(
struct BlitVaryings {
	float4 position : SV_Position;
	float2 uv : TEXCOORD0;
};

[[vk::binding(0, 0)]] Sampler2D blit_source;

[shader("vertex")]
BlitVaryings vertexMain(uint index : SV_VertexID) {
	// An oversized triangle covering the whole viewport, its uvs run 0..1 across the covered area
	float2 uv = float2((index << 1) & 2, index & 2);
	BlitVaryings varyings;
	// The session inverts Y for Vulkan, so v = 0 is written as +Y to land at the top of the viewport
	varyings.position = float4(uv.x * 2 - 1, 1 - uv.y * 2, 0, 1);
	varyings.uv = uv;
	return varyings;
}

[shader("fragment")]
float4 fragmentMain(float2 uv : TEXCOORD0) : SV_Target {
	// The view covers a single mip, so there is never another level to pick between
	return blit_source.SampleLevel(uv, 0);
}
)slang";

	inline bool create_internal_shader_module(GpuQueue* queue, std::string_view source, GPU::shaders::SHADER_STAGE stage, VkShaderModule& out_module) {
		auto spirv = compile_shader(queue, {source.data(), source.size()}, stage);
		if(!spirv) return false;

		VkShaderModuleCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = spirv->size(),
			.pCode = (const uint32_t*)spirv->data(),
		};
		VK_CHECK(vkCreateShaderModule(queue->device, &info, queue->callbacks, &out_module), false);
		return true;
	}

	// Format independent, so built once and shared by every blit on the queue.
	inline bool ensure_blit_layout(GpuQueue* queue) {
		if(queue->blit_pipeline_layout) return true;

		constexpr static std::array stages = {GPU::shaders::SHADER_STAGE::VERTEX, GPU::shaders::SHADER_STAGE::FRAGMENT};
		for(size_t i = 0; i < stages.size(); ++i)
			if(!create_internal_shader_module(queue, BLIT_SHADER, stages[i], queue->blit_shader_modules[i])) return false;

		for(size_t linear = 0; linear < queue->blit_samplers.size(); ++linear) {
			auto filter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
			VkSamplerCreateInfo info {
				.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
				.magFilter = filter,
				.minFilter = filter,
				.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
				.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
				.maxLod = VK_LOD_CLAMP_NONE,
			};
			VK_CHECK(vkCreateSampler(queue->device, &info, queue->callbacks, &queue->blit_samplers[linear]), false);
		}

		VkDescriptorSetLayoutBinding binding {
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		};
		VkDescriptorSetLayoutCreateInfo set_layout {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = 1,
			.pBindings = &binding,
		};
		VK_CHECK(vkCreateDescriptorSetLayout(queue->device, &set_layout, queue->callbacks, &queue->blit_descriptor_set_layout), false);

		VkPipelineLayoutCreateInfo layout {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &queue->blit_descriptor_set_layout,
		};
		VK_CHECK(vkCreatePipelineLayout(queue->device, &layout, queue->callbacks, &queue->blit_pipeline_layout), false);

		return true;
	}

	// Dynamic rendering bakes the destination's format into the pipeline, so one is kept per format
	inline VkPipeline blit_pipeline(GpuQueue* queue, VkFormat format) {
		auto key = static_cast<uint64_t>(format);
		if(auto found = queue->blit_pipelines.find(key); found != queue->blit_pipelines.end())
			return found->second;

		if(!ensure_blit_layout(queue)) return VK_NULL_HANDLE;

		std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages {
			VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_VERTEX_BIT,
				.module = queue->blit_shader_modules[0],
				.pName = "main",
			}, VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
				.module = queue->blit_shader_modules[1],
				.pName = "main",
			}
		};

		// The triangle is generated from SV_VertexID, so nothing is fetched and nothing is indexed
		VkPipelineVertexInputStateCreateInfo vertex_input_state {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
		VkPipelineInputAssemblyStateCreateInfo assembly_state {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};
		VkPipelineViewportStateCreateInfo viewport_state {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.scissorCount = 1,
		};
		VkPipelineRasterizationStateCreateInfo rasterization_state {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_NONE,
			.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
			.lineWidth = 1.0f,
		};
		VkPipelineMultisampleStateCreateInfo multisample_state {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		};
		VkPipelineDepthStencilStateCreateInfo depth_stencil_state {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

		VkPipelineColorBlendAttachmentState attachment {
			.blendEnable = VK_FALSE,
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
		};
		VkPipelineColorBlendStateCreateInfo color_blend_state {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &attachment,
		};

		std::array<VkDynamicState, 2> dynamic = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo dynamic_state {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = dynamic.size(),
			.pDynamicStates = dynamic.data(),
		};

		VkPipelineRenderingCreateInfo dynamic_rendering_info {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &format,
		};
		// Deliberately no VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT: the blit binds its source the classic way rather than through whatever heap the user happens to have active
		VkGraphicsPipelineCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.pNext = &dynamic_rendering_info,
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
			.layout = queue->blit_pipeline_layout,
			.renderPass = VK_NULL_HANDLE,
			.basePipelineIndex = -1,
		};

		VkPipeline pipeline = VK_NULL_HANDLE;
		VK_CHECK(vkCreateGraphicsPipelines(queue->device, VK_NULL_HANDLE, 1, &info, queue->callbacks, &pipeline), VK_NULL_HANDLE);

		queue->blit_pipelines[key] = pipeline;
		return pipeline;
	}

	// Destroys every view whose submission has finished. Each view names one subresource, so unlike a descriptor set there is nothing about one worth recycling.
	inline void reclaim_views(GpuQueue* queue) {
		if(queue->views_in_flight.empty()) return;

		uint64_t current_finished_submission = GPU::detail::finished_submission(queue);

		for(size_t i = queue->views_in_flight.size(); i--; ) {
			auto [view, submit] = queue->views_in_flight[i];
			if(submit > current_finished_submission) continue;

			vkDestroyImageView(queue->device, view, queue->callbacks);
			queue->views_in_flight.erase(queue->views_in_flight.begin() + i);
		}
	}

	// Views are shared with render passes and retired by reclaim_views; sets go back on the free list.
	inline void reclaim_blit_transients(GpuQueue* queue) {
		reclaim_views(queue);
		if(queue->blit_descriptor_sets_in_flight.empty()) return;

		uint64_t current_finished_submission = GPU::detail::finished_submission(queue);

		for(size_t i = queue->blit_descriptor_sets_in_flight.size(); i--; ) {
			auto [set, submit] = queue->blit_descriptor_sets_in_flight[i];
			if(submit > current_finished_submission) continue;

			queue->blit_descriptor_sets_free.push_back(set);
			queue->blit_descriptor_sets_in_flight.erase(queue->blit_descriptor_sets_in_flight.begin() + i);
		}
	}

	constexpr static uint32_t BLIT_DESCRIPTORS_PER_POOL = 32;

	inline VkDescriptorSet blit_descriptor_set(GpuQueue* queue) {
		if(!queue->blit_descriptor_sets_free.empty()) {
			auto set = queue->blit_descriptor_sets_free.back();
			queue->blit_descriptor_sets_free.pop_back();
			return set;
		}

		VkDescriptorSetAllocateInfo info {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorSetCount = 1,
			.pSetLayouts = &queue->blit_descriptor_set_layout,
		};

		// Sets are recycled rather than freed, so pools only grow by adding another.
		if(!queue->blit_descriptor_pools.empty()) {
			info.descriptorPool = queue->blit_descriptor_pools.back();

			VkDescriptorSet set = VK_NULL_HANDLE;
			if(vkAllocateDescriptorSets(queue->device, &info, &set) == VK_SUCCESS)
				return set;
		}

		VkDescriptorPoolSize size {
			.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = BLIT_DESCRIPTORS_PER_POOL,
		};
		VkDescriptorPoolCreateInfo pool {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = BLIT_DESCRIPTORS_PER_POOL,
			.poolSizeCount = 1,
			.pPoolSizes = &size,
		};
		VkDescriptorPool created = VK_NULL_HANDLE;
		VK_CHECK(vkCreateDescriptorPool(queue->device, &pool, queue->callbacks, &created), VK_NULL_HANDLE);
		queue->blit_descriptor_pools.push_back(created);

		info.descriptorPool = created;
		VkDescriptorSet set = VK_NULL_HANDLE;
		VK_CHECK(vkAllocateDescriptorSets(queue->device, &info, &set), VK_NULL_HANDLE);
		return set;
	}
}

void gpuBlitTextureEXT(GpuCommandBuffer* cmd, GpuTexture* destination, const GpuTexture* source,
		bool linear_filter,
		uint32_t destination_mip, uint32_t destination_slice,
		uint32_t source_mip, uint32_t source_slice) {
	assert(cmd->state == GpuCommandBuffer::Recording && "A blit opens a render pass of its own, so it can't be recorded inside another one");
	assert((source->descriptor.usage & USAGE_SAMPLED) && "The blit source must have been created with USAGE_SAMPLED");
	assert((destination->descriptor.usage & USAGE_COLOR_ATTACHMENT) && "The blit destination must have been created with USAGE_COLOR_ATTACHMENT");
	assert(!gpuFormatIsDepthStencilEXT(source->descriptor.format) && !gpuFormatIsDepthStencilEXT(destination->descriptor.format) && "Depth/stencil textures can't be blitted");
	assert(source->descriptor.type != TEXTURE_3D && "A slice of a 3D texture can't be sampled on its own, blit out of a 2D array instead");
	assert(source_mip < source->descriptor.mipCount && destination_mip < destination->descriptor.mipCount);

	auto queue = cmd->queue;
	auto pipeline = GPU::detail::blit_pipeline(queue, GPU::detail::format2vulkan(destination->descriptor.format));
	if(pipeline == VK_NULL_HANDLE) return;

	GPU::detail::reclaim_blit_transients(queue);
	auto retire_after = queue->command_submission_timeline_semaphore_next_value;

	// Both sides are one mip of one layer, so the shader is a plain sampler2D whatever the textures are.
	constexpr static auto subresource_view = [](GpuQueue* queue, const GpuTexture* texture, uint32_t mip, uint32_t slice) -> VkImageView {
		VkImageViewCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = texture->image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = GPU::detail::format2vulkan(texture->descriptor.format),
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = mip,
				.levelCount = 1,
				.baseArrayLayer = slice,
				.layerCount = 1,
			}
		};
		VkImageView view = VK_NULL_HANDLE;
		VK_CHECK(vkCreateImageView(queue->device, &info, queue->callbacks, &view), VK_NULL_HANDLE);
		return view;
	};

	auto source_view = subresource_view(queue, source, source_mip, source_slice);
	auto destination_view = subresource_view(queue, destination, destination_mip, destination_slice);
	if(source_view == VK_NULL_HANDLE || destination_view == VK_NULL_HANDLE) return;
	queue->views_in_flight.emplace_back(source_view, retire_after);
	queue->views_in_flight.emplace_back(destination_view, retire_after);

	auto set = GPU::detail::blit_descriptor_set(queue);
	if(set == VK_NULL_HANDLE) return;
	queue->blit_descriptor_sets_in_flight.emplace_back(set, retire_after);
	{
		// Textures in this backend live in VK_IMAGE_LAYOUT_GENERAL, the same layout their heap descriptors are written against
		bool filtering = linear_filter && gpuFormatIsFilterableEXT(source->descriptor.format);
		VkDescriptorImageInfo image {
			.sampler = queue->blit_samplers[filtering],
			.imageView = source_view,
			.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
		};
		VkWriteDescriptorSet write {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &image,
		};
		vkUpdateDescriptorSets(queue->device, 1, &write, 0, nullptr);
	}

	// A source just rendered into is still in an attachment layout.
	GPU::detail::transition_texture(cmd->command_buffer, source, VK_IMAGE_LAYOUT_GENERAL,
		VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		source_mip, source_slice, false
	);

	// The draw covers the whole mip, so the old contents are never worth reading back in
	GPU::detail::transition_texture(cmd->command_buffer, destination, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		destination_mip, destination_slice, true
	);

	auto extent = GPU::detail::mip_extent(destination, destination_mip);
	VkRenderingAttachmentInfo attachment {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = destination_view,
		.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.resolveMode = VK_RESOLVE_MODE_NONE,
		.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
	};
	VkRenderingInfo rendering {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = {
			.offset = {0, 0},
			.extent = {extent.x, extent.y},
		},
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &attachment,
	};
	vkCmdBeginRendering(cmd->command_buffer, &rendering);

	vkCmdBindPipeline(cmd->command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	vkCmdBindDescriptorSets(cmd->command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, queue->blit_pipeline_layout, 0, 1, &set, 0, nullptr);
	gpuSetViewportEXT(cmd, extent);
	gpuSetScissorRectEXT(cmd, extent);
	vkCmdDraw(cmd->command_buffer, 3, 1, 0, 0);

	vkCmdEndRendering(cmd->command_buffer);

	// Hand the destination back in the layout the rest of the backend expects to find it in
	GPU::detail::transition_texture(cmd->command_buffer, destination, VK_IMAGE_LAYOUT_GENERAL,
		VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		destination_mip, destination_slice, false
	);

	// The blit bound a pipeline of its own, so put back whatever the user had set before it
	if(cmd->bound_pipeline)
		vkCmdBindPipeline(cmd->command_buffer,
			cmd->bound_pipeline->color_target_count.has_value() ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE,
			cmd->bound_pipeline->pipeline);
}

void gpuSetActiveTextureHeapPtr(GpuCommandBuffer* cmd, gpu* texture_heap) {
	const GpuQueue::Allocation* holding = nullptr;
	auto heap = GPU::detail::closest_buffer(cmd->queue, texture_heap, &holding);
	assert(heap.buffer && "The texture heap pointer doesn't lie inside any allocation");
	if(!heap.buffer) return;

	// Bound in place when allocated bindable and long enough for the reserved range, so shader writes are seen after a HAZARD_DESCRIPTORS barrier.
	if(holding && holding->descriptor_heap && heap.size - heap.offset >= cmd->queue->minimum_descriptor_heap_size) {
		VkBindHeapInfoEXT heap_info {
			.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
			.heapRange = {
				.address = (VkDeviceAddress)texture_heap,
				.size = heap.size - heap.offset
			},
			.reservedRangeSize = cmd->queue->minimum_descriptor_heap_size
		};
		vkCmdBindResourceHeapEXT(cmd->command_buffer, &heap_info);
		return;
	}

	// Otherwise a snapshot is bound: copied on every call, and shader writes to the caller's memory stay invisible until the heap is set again.
	GPU::detail::report(cmd, GPU_DIAGNOSTIC_EMULATED,
		holding && !holding->descriptor_heap
			? "gpuSetActiveTextureHeapPtr: the heap was not allocated as MEMORY_DESCRIPTOR_HEAP, so it is copied into a bindable heap on every call and writes a shader makes to it are not seen"
			: "gpuSetActiveTextureHeapPtr: the heap is shorter than this device's reserved descriptor heap range, so it is copied into a padded heap on every call");

	if(!cmd->queue->descriptor_heaps.contains(heap.base)) {
		auto size = std::max<VkDeviceSize>(heap.size, cmd->queue->minimum_descriptor_heap_size);
		VkBufferCreateInfo buffer_info {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		};
		VmaAllocationCreateInfo alloc_info {
			.usage = VMA_MEMORY_USAGE_AUTO,
			.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
		};

		auto& [buffer, allocation, heap_size, heap_address] = cmd->queue->descriptor_heaps[heap.base];
		heap_size = size;
		VK_CHECK(vmaCreateBuffer(cmd->queue->gpu_allocator, &buffer_info, &alloc_info, &buffer, &allocation, nullptr), /*nothing*/);

		VkBufferDeviceAddressInfo address_info {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = buffer
		};
		heap_address = vkGetBufferDeviceAddress(cmd->queue->device, &address_info);
	}

	auto [heap_buffer, _allocation, heap_size, heap_address] = cmd->queue->descriptor_heaps[heap.base];
	VkBufferCopy copy {
		.srcOffset = heap.offset,
		.dstOffset = heap.offset,
		.size = heap.size - heap.offset
	};
	vkCmdCopyBuffer(cmd->command_buffer, heap.buffer, heap_buffer, 1, &copy);

	// Without this, the bind below and every draw after it race the copy that filled the heap.
	const VkMemoryBarrier2 barrier {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
		.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		.dstAccessMask = VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT | VK_ACCESS_2_SHADER_READ_BIT,
	};
	const VkDependencyInfo dependency {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.memoryBarrierCount = 1,
		.pMemoryBarriers = &barrier,
	};
	vkCmdPipelineBarrier2(cmd->command_buffer, &dependency);

	VkBindHeapInfoEXT heap_info {
		.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
		.heapRange = {
			.address = heap_address + heap.offset,
			.size = heap_size - heap.offset
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
			// PRE_RASTERIZATION_SHADERS covers vertex + tessellation + geometry + mesh stages in a single bit (Vulkan 1.3 / VK_KHR_synchronization2).
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

		// The command processor must not prefetch indirect arguments until a compute shader's writes to them are visible.
		if (hazards & HAZARD_DRAW_ARGUMENTS) {
			src_access |= VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
			dst_access |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT_KHR;
		}

		// Invalidate the sampler's descriptor cache so it re-fetches updated heap entries.
		if (hazards & HAZARD_DESCRIPTORS) {
			src_access |= VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
			// VK_EXT_descriptor_heap's read bits; VK_ACCESS_2_DESCRIPTOR_BUFFER_READ_BIT_EXT belongs to descriptor buffers, which are not enabled, and made this barrier invalid.
			dst_access |= VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT | VK_ACCESS_2_SAMPLER_HEAP_READ_BIT_EXT
				| VK_ACCESS_2_UNIFORM_READ_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT_KHR;
		}

		// Invalidate HiZ and stencil metadata after compute wrote what will be bound as a depth buffer.
		if (hazards & HAZARD_DEPTH_STENCIL) {
			src_access |= VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
			dst_access |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT_KHR | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT_KHR;
		}

		// Always include the generic dependency so the barrier is never a no-op.
		src_access |= VK_ACCESS_2_MEMORY_WRITE_BIT_KHR;
		dst_access |= VK_ACCESS_2_MEMORY_READ_BIT_KHR;

		return {src_access, dst_access};
	}

	// Vulkan rejects access bits a stage cannot perform; what is dropped is still covered by MEMORY_READ/WRITE, which every stage accepts.
	inline VkAccessFlags2 access_for_stages(VkAccessFlags2 access, VkPipelineStageFlags2 stages) {
		constexpr VkPipelineStageFlags2 everything = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

		constexpr VkPipelineStageFlags2 shader_stages = everything | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT
			| VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
			| VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT
			| VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
		if(!(stages & shader_stages))
			access &= ~(VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT
				| VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
				| VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
				| VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT | VK_ACCESS_2_SAMPLER_HEAP_READ_BIT_EXT);

		constexpr VkPipelineStageFlags2 indirect_stages = everything | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT
			| VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
		if(!(stages & indirect_stages))
			access &= ~VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;

		constexpr VkPipelineStageFlags2 depth_stages = everything | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT
			| VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
		if(!(stages & depth_stages))
			access &= ~(VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

		constexpr VkPipelineStageFlags2 color_stages = everything | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT
			| VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
		if(!(stages & color_stages))
			access &= ~(VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

		constexpr VkPipelineStageFlags2 transfer_stages = everything | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT
			| VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT
			| VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT;
		if(!(stages & transfer_stages))
			access &= ~(VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);

		return access;
	}
}

void gpuBarrier(GpuCommandBuffer* cmd, STAGE before, STAGE after, HAZARD_FLAGS hazards) {
	VkPipelineStageFlags2KHR src_stage = GPU::detail::stage2vulkan(before);
	VkPipelineStageFlags2KHR dst_stage = GPU::detail::stage2vulkan(after);
	auto [src_access, dst_access] = GPU::detail::hazard2access(hazards);
	// The hazard flags don't know which stages they will be paired with, and Vulkan rejects a barrier that names an access a stage cannot perform
	src_access = GPU::detail::access_for_stages(src_access, src_stage);
	dst_access = GPU::detail::access_for_stages(dst_access, dst_stage);

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
	// The core entry point: volk leaves the KHR alias null since that extension is not enabled on a 1.4 device.
	vkCmdPipelineBarrier2(cmd->command_buffer, &dependency);
}

namespace GPU::detail {
	struct SignalPushConstants {
		VkDeviceAddress counter;
		uint64_t value;
		uint32_t use_max;
	};


	// For SIGNAL_ATOMIC_MAX and OR. The address arrives as a pointer in a push constant, so no descriptor is needed.
	constexpr static std::string_view SIGNAL_SHADER = R"slang(
// Matches SignalPushConstants
struct SignalPush {
	uint64_t* counter;
	uint64_t value;
	uint use_max; // 1 for SIGNAL_ATOMIC_MAX, 0 for SIGNAL_ATOMIC_OR
};

[[vk::push_constant]] ConstantBuffer<SignalPush> push;

[shader("compute")] [numthreads(1, 1, 1)]
void main() {
	if(push.use_max != 0) InterlockedMax(*push.counter, push.value);
	else InterlockedOr(*push.counter, push.value);
}
)slang";

	// Built on first use; one pipeline covers both operations, chosen by push constant.
	inline VkPipeline signal_pipeline(GpuQueue* queue) {
		if(queue->signal_pipeline) return queue->signal_pipeline;

		{
			if(!create_internal_shader_module(queue, SIGNAL_SHADER, GPU::shaders::SHADER_STAGE::COMPUTE, queue->signal_shader_module))
				return VK_NULL_HANDLE;

			VkPushConstantRange range {
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
				.offset = 0,
				.size = sizeof(SignalPushConstants),
			};
			VkPipelineLayoutCreateInfo layout_info {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &range,
			};
			VK_CHECK(vkCreatePipelineLayout(queue->device, &layout_info, queue->callbacks, &queue->signal_pipeline_layout), VK_NULL_HANDLE);
		}

		VkComputePipelineCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_COMPUTE_BIT,
				.module = queue->signal_shader_module,
				.pName = "main",
			},
			.layout = queue->signal_pipeline_layout,
		};
		VK_CHECK(vkCreateComputePipelines(queue->device, VK_NULL_HANDLE, 1, &info, queue->callbacks, &queue->signal_pipeline), VK_NULL_HANDLE);
		return queue->signal_pipeline;
	}

	// Recycled only after their submission finishes: resetting one a command buffer in flight waits on would let the wait through early.
	inline VkEvent acquire_event(GpuQueue* queue) {
		if(!queue->events_in_flight.empty()) {
			uint64_t finished = GPU::detail::finished_submission(queue);

			for(size_t i = queue->events_in_flight.size(); i--; ) {
				auto [event, submit] = queue->events_in_flight[i];
				if(submit > finished) continue;

				if(vkResetEvent(queue->device, event) == VK_SUCCESS) queue->events_free.push_back(event);
				else vkDestroyEvent(queue->device, event, queue->callbacks); // Still set, so it can't be reused
				queue->events_in_flight.erase(queue->events_in_flight.begin() + i);
			}
		}

		if(!queue->events_free.empty()) {
			auto event = queue->events_free.back();
			queue->events_free.pop_back();
			return event;
		}

		// Not VK_EVENT_CREATE_DEVICE_ONLY_BIT: reuse above resets these from the host
		VkEventCreateInfo info { .sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO };
		VkEvent event = VK_NULL_HANDLE;
		VK_CHECK(vkCreateEvent(queue->device, &info, queue->callbacks, &event), VK_NULL_HANDLE);
		return event;
	}
}

void gpuSignalAfter(GpuCommandBuffer* cmd, STAGE before, gpu* ptr, uint64_t value, SIGNAL signal) {
	auto queue = cmd->queue;
	auto producer_stage = GPU::detail::stage2vulkan(before);

	// A render pass forbids every way of writing the counter, and the event too, so a signal inside one does nothing.
	if(cmd->state != GpuCommandBuffer::Recording) {
		GPU::detail::report(cmd, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuSignalAfter: writing the counter needs a transfer or a dispatch, and neither can be recorded inside a render pass, so nothing is signalled; split the pass or signal around it");
		return;
	}

	// The counter is ordinary memory in an ordinary allocation, so writing it needs to be ordered after the work whose completion it is reporting
	{
		const VkMemoryBarrier2 barrier {
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
			.srcStageMask = producer_stage,
			.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
			.dstStageMask = signal == SIGNAL_ATOMIC_SET ? VK_PIPELINE_STAGE_2_COPY_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.dstAccessMask = signal == SIGNAL_ATOMIC_SET ? VK_ACCESS_2_TRANSFER_WRITE_BIT : VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		};
		const VkDependencyInfo dependency {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.memoryBarrierCount = 1,
			.pMemoryBarriers = &barrier,
		};
		vkCmdPipelineBarrier2(cmd->command_buffer, &dependency);
	}

	VkPipelineStageFlags2 wrote_at = VK_PIPELINE_STAGE_2_COPY_BIT;
	if(signal == SIGNAL_ATOMIC_SET) {
		// A store needs no atomic: vkCmdUpdateBuffer writes the 8 bytes straight into the counter
		auto counter = GPU::detail::closest_buffer(queue, ptr);
		assert(counter.buffer && "The split barrier counter doesn't lie inside any allocation");
		if(!counter.buffer) return;

		// vkCmdUpdateBuffer writes at a 4 byte granularity, and a 64 bit counter wants 8 anyway
		assert(counter.offset % sizeof(value) == 0 && "The split barrier counter isn't 8 byte aligned");

		vkCmdUpdateBuffer(cmd->command_buffer, counter.buffer, counter.offset, sizeof(value), &value);
	} else if(!queue->capabilities.split_barrier_signals) {
		GPU::detail::report(cmd, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuSignalAfter: SIGNAL_ATOMIC_MAX and SIGNAL_ATOMIC_OR need 64 bit buffer atomics, which this device does not support, so the counter is left alone");
	} else if(auto pipeline = GPU::detail::signal_pipeline(queue); pipeline != VK_NULL_HANDLE) {
		GPU::detail::SignalPushConstants push {
			.counter = (VkDeviceAddress)ptr,
			.value = value,
			.use_max = signal == SIGNAL_ATOMIC_MAX,
		};
		vkCmdBindPipeline(cmd->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
		vkCmdPushConstants(cmd->command_buffer, queue->signal_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
		vkCmdDispatch(cmd->command_buffer, 1, 1, 1);
		wrote_at = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

		// Restore the caller's pipeline, as a blit does, or the next dispatch would run this shader.
		if(cmd->bound_pipeline)
			vkCmdBindPipeline(cmd->command_buffer,
				cmd->bound_pipeline->color_target_count.has_value() ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE,
				cmd->bound_pipeline->pipeline);
	}

	// The second half of a split barrier: an event set here, for a later gpuWaitBefore to wait on without stalling what sits between them.
	auto event = GPU::detail::acquire_event(queue);
	if(event == VK_NULL_HANDLE) return;

	GpuCommandBuffer::PendingSignal pending {
		.ptr = (VkDeviceAddress)ptr,
		.value = value,
		.event = event,
		.barrier = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
			.srcStageMask = producer_stage | wrote_at,
			.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
			// The consumer is unknown until the wait is recorded, and Vulkan wants both halves to name the same dependency, so this end stays open
			.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
		},
	};

	const VkDependencyInfo dependency {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.memoryBarrierCount = 1,
		.pMemoryBarriers = &pending.barrier,
	};
	vkCmdSetEvent2(cmd->command_buffer, event, &dependency);

	cmd->pending_signals.push_back(pending);
	cmd->events_used.push_back(event);
}

void gpuWaitBefore(GpuCommandBuffer* cmd, STAGE after, gpu* ptr, uint64_t value, OP op, HAZARD_FLAGS hazards, uint64_t mask) {
	// No Vulkan command compares a counter, so only the ordering of a matching earlier signal can be honored; anything else becomes the barrier below.
	const bool comparison_is_satisfied_by_the_pair = (op == OP_GREATER_EQUAL || op == OP_EQUAL || op == OP_ALWAYS)
		&& mask == ~(uint64_t)0;

	if(comparison_is_satisfied_by_the_pair && cmd->state == GpuCommandBuffer::Recording) {
		auto match = std::find_if(cmd->pending_signals.begin(), cmd->pending_signals.end(),
			[&](const GpuCommandBuffer::PendingSignal& pending) {
				return pending.ptr == (VkDeviceAddress)ptr && pending.value == value;
			});

		if(match != cmd->pending_signals.end()) {
			const VkDependencyInfo dependency {
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.memoryBarrierCount = 1,
				.pMemoryBarriers = &match->barrier,
			};
			auto producer_stage = match->barrier.srcStageMask;
			vkCmdWaitEvents2(cmd->command_buffer, 1, &match->event, &dependency);
			cmd->pending_signals.erase(match);

			// The event repeats its signal's dependency, so hazards need a second barrier, from the producer's stage only so the split is kept.
			if(hazards) {
				auto consumer_stage = GPU::detail::stage2vulkan(after);
				auto [src_access, dst_access] = GPU::detail::hazard2access(hazards);
				const VkMemoryBarrier2 barrier {
					.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
					.srcStageMask = producer_stage,
					.srcAccessMask = GPU::detail::access_for_stages(src_access, producer_stage),
					.dstStageMask = consumer_stage,
					.dstAccessMask = GPU::detail::access_for_stages(dst_access, consumer_stage),
				};
				const VkDependencyInfo hazard_dependency {
					.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
					.memoryBarrierCount = 1,
					.pMemoryBarriers = &barrier,
				};
				vkCmdPipelineBarrier2(cmd->command_buffer, &hazard_dependency);
			}
			return;
		}
	}

	// Without VK_KHR_dynamic_rendering_local_read a render pass holds neither events nor a barrier from outside it
	if(cmd->state != GpuCommandBuffer::Recording) {
		GPU::detail::report(cmd, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuWaitBefore: Vulkan can't record a wait or a barrier inside a render pass, so nothing waits; record the wait before gpuBeginRenderPass");
		return;
	}

	GPU::detail::report(cmd, GPU_DIAGNOSTIC_EMULATED,
		!comparison_is_satisfied_by_the_pair
			? "gpuWaitBefore: a masked or non-monotonic comparison has no equivalent Vulkan command, so this waits on every stage instead of splitting the barrier"
			: "gpuWaitBefore: no gpuSignalAfter earlier in this command buffer matches this counter and value, so this waits on every stage instead of splitting the barrier");
	gpuBarrier(cmd, STAGE_ALL, after, hazards);
}

namespace GPU::detail {
	VkColorComponentFlags mask2vulkan(uint8_t mask) {
		VkColorComponentFlags flags = 0;
		if (mask & 0x1) flags |= VK_COLOR_COMPONENT_R_BIT;
		if (mask & 0x2) flags |= VK_COLOR_COMPONENT_G_BIT;
		if (mask & 0x4) flags |= VK_COLOR_COMPONENT_B_BIT;
		if (mask & 0x8) flags |= VK_COLOR_COMPONENT_A_BIT;
		return flags;
	};
}

void gpuSetPipeline(GpuCommandBuffer* cmd, const GpuPipeline* pipeline) {
	vkCmdBindPipeline(cmd->command_buffer, pipeline->color_target_count.has_value() ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
	cmd->bound_pipeline = pipeline;

	// Blending is dynamic state a draw requires, so binding applies the pipeline's own unless gpuSetBlendState is overriding it.
	if(cmd->bound_blend_state) {
		// An explicit state outranks the pipeline's own description, and has to be re-applied because the masks it produces are relative to the pipeline now bound
		gpuSetBlendState(cmd, cmd->bound_blend_state);
	} else if(auto count = static_cast<uint32_t>(pipeline->blend_enables.size()); count > 0) {
		vkCmdSetColorBlendEnableEXT(cmd->command_buffer, 0, count, pipeline->blend_enables.data());
		vkCmdSetColorBlendEquationEXT(cmd->command_buffer, 0, count, pipeline->blend_equations.data());

		std::vector<VkColorComponentFlags> masks; masks.reserve(count);
		for(auto mask: pipeline->color_write_masks)
			masks.push_back(GPU::detail::mask2vulkan(mask));
		vkCmdSetColorWriteMaskEXT(cmd->command_buffer, 0, count, masks.data());
	}
}

struct ComputePipelinePushConstants {
	gpu* data;
	gpu* sampler_map;
};

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

void gpuDispatchIndirect(GpuCommandBuffer* cmd, gpu* dataGpu, gpu* gridDimensionsGpu) {
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

	auto grid = GPU::detail::closest_buffer(cmd->queue, gridDimensionsGpu);
	assert(grid.buffer && "The dispatch dimensions don't lie inside any allocation");
	vkCmdDispatchIndirect(cmd->command_buffer, grid.buffer, grid.offset);
}

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

// A mesh pipeline differs only in its stage bit and in having no vertex input or input assembly state.
static GpuPipeline* create_raster_pipeline(GpuQueue* queue, GpuByteSpan primitiveIR,
	GPU::shaders::SHADER_STAGE primitive_stage, GpuByteSpan fragmentIR, const GpuRasterDesc& desc
) {
	if(desc.supportDualSourceBlending && !queue->capabilities.dual_source_blending) {
		GPU::detail::report(queue, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuCreateGraphicsPipeline: this device was not set up with dualSrcBlend, so a pipeline with supportDualSourceBlending is not created");
		errno = VK_ERROR_FEATURE_NOT_PRESENT;
		return nullptr;
	}
	const bool mesh = primitive_stage == GPU::shaders::SHADER_STAGE::MESH;

	constexpr static auto topology2vulkan = [](TOPOLOGY t) -> VkPrimitiveTopology{
		switch (t) {
			// case TOPOLOGY_POINT_LIST: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST; case TOPOLOGY_LINE_LIST: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST; case TOPOLOGY_LINE_STRIP: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
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

	// Both stages compile before anything is created, so a failing shader leaves no half made pipeline.
	auto primitive_spirv = compile_shader(queue, primitiveIR, primitive_stage);
	if(!primitive_spirv) return nullptr;
	auto fragment_spirv = compile_shader(queue, fragmentIR, GPU::shaders::SHADER_STAGE::FRAGMENT);
	if(!fragment_spirv) return nullptr;

	// Constructed, not just allocated: a GpuPipeline owns vectors now, and assigning into raw storage would be writing through whatever those bytes happened to contain
	auto out = new(queue->cpu_allocator(nullptr, sizeof(GpuPipeline))) GpuPipeline{};
	out->color_target_count = desc.colorTargets.size();

	// One failure path for the modules and the pipeline alike, at the end, so nothing made before a failure is left behind
	VkResult created = VK_SUCCESS;
	std::array<VkShaderModule, 2> shader_modules = {};
	const std::array<const std::string*, 2> spirv = {&*primitive_spirv, &*fragment_spirv};
	for(size_t i = 0; i < shader_modules.size() && created == VK_SUCCESS; ++i) {
		VkShaderModuleCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = spirv[i]->size(),
			.pCode = (const uint32_t*)spirv[i]->data(),
		};
		created = vkCreateShaderModule(queue->device, &info, queue->callbacks, &shader_modules[i]);
	}

	std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages {
		VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = mesh ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_VERTEX_BIT,
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

		const bool blendEnabled = desc.blendstate.has_value;
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

			// Kept for gpuSetPipeline to apply, since the copy in the create info is ignored once these are dynamic
			out->blend_enables.push_back(state.blendEnable);
			out->blend_equations.push_back(VkColorBlendEquationEXT{
				.srcColorBlendFactor = state.srcColorBlendFactor,
				.dstColorBlendFactor = state.dstColorBlendFactor,
				.colorBlendOp = state.colorBlendOp,
				.srcAlphaBlendFactor = state.srcAlphaBlendFactor,
				.dstAlphaBlendFactor = state.dstAlphaBlendFactor,
				.alphaBlendOp = state.alphaBlendOp,
			});
			out->color_write_masks.push_back(effectiveMask);

			attachments.push_back(state);
		}
	}

	VkPipelineColorBlendStateCreateInfo color_blend_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = static_cast<uint32_t>(attachments.size()),
		.pAttachments = attachments.empty() ? nullptr : attachments.data(),
	};

	// Every vkCmdSet state must be declared dynamic or it is ignored; the zeroed create info would disable depth. The last three need extended_dynamic_state3.
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
	dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic.size());
	dynamic_state.pDynamicStates = dynamic.data();

	std::vector<VkFormat> color_formats; color_formats.reserve(desc.colorTargets.size());
	for(auto& target: desc.colorTargets)
		color_formats.emplace_back(GPU::detail::format2vulkan(target.format));

	VkPipelineRenderingCreateInfo dynamic_rendering_info {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = static_cast<uint32_t>(color_formats.size()),
		.pColorAttachmentFormats = color_formats.data(),
		.depthAttachmentFormat = GPU::detail::format2vulkan(desc.depthFormat), // FORMAT_NONE is VK_FORMAT_UNDEFINED, no attachment
		.stencilAttachmentFormat = GPU::detail::format2vulkan(desc.stencilFormat),
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
		// A mesh shader emits its own primitives, so there is no vertex input or input assembly to describe.
		.pVertexInputState = mesh ? nullptr : &vertex_input_state,
		.pInputAssemblyState = mesh ? nullptr : &assembly_state,
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
	if(created == VK_SUCCESS) created = vkCreateGraphicsPipelines(queue->device, VK_NULL_HANDLE, 1, &info, queue->callbacks, &out->pipeline);

	for(auto module: shader_modules) // A module never made is VK_NULL_HANDLE, which destroying ignores
		vkDestroyShaderModule(queue->device, module, queue->callbacks);

	if(created != VK_SUCCESS) {
		errno = created;
		out->~GpuPipeline();
		queue->cpu_allocator(out, 0);
		return nullptr;
	}
	return out;
}
}

GpuPipeline* gpuCreateGraphicsPipeline(GpuQueue* queue, GpuByteSpan vertexIR, GpuByteSpan fragmentIR, const GpuRasterDesc& desc) {
	return GPU::detail::create_raster_pipeline(queue, vertexIR, GPU::shaders::SHADER_STAGE::VERTEX, fragmentIR, desc);
}

GpuPipeline* gpuCreateGraphicsMeshletPipeline(GpuQueue* queue, GpuByteSpan meshletIR, GpuByteSpan fragmentIR, const GpuRasterDesc& desc) {
	// Checked before compiling, so a device without mesh shaders reports that rather than a shader error; reported rather than thrown, being extern "C".
	if(!queue->capabilities.mesh_shaders) {
		GPU::detail::report(queue, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuCreateGraphicsMeshletPipeline: this device was not set up with VK_EXT_mesh_shader, so no pipeline is created");
		errno = VK_ERROR_FEATURE_NOT_PRESENT;
		return nullptr;
	}

	return GPU::detail::create_raster_pipeline(queue, meshletIR, GPU::shaders::SHADER_STAGE::MESH, fragmentIR, desc);
}

GpuDepthStencilState* gpuCreateDepthStencilState(GpuQueue* queue, const GpuDepthStencilDesc& desc) {
	auto out = (GpuDepthStencilState*)queue->cpu_allocator(nullptr, sizeof(GpuDepthStencilState));
	out->descriptor = desc;
	return out;
}

GpuBlendState* gpuCreateBlendState(GpuQueue* queue, const GpuBlendDesc& desc) {
	auto out = (GpuBlendState*)queue->cpu_allocator(nullptr, sizeof(GpuBlendState));
	out->descriptor = desc;
	return out;
}

void gpuFreeDepthStencilState(GpuQueue* queue, GpuDepthStencilState* state) {
	queue->cpu_allocator(state, 0);
}

void gpuFreeBlendState(GpuQueue* queue, GpuBlendState* state) {
	queue->cpu_allocator(state, 0);
}

void gpuSetDepthStencilState(GpuCommandBuffer* cmd, const GpuDepthStencilState* state) {
	GPU::detail::apply_depth_stencil_state(cmd, state->descriptor);
}

void gpuSetBlendState(GpuCommandBuffer* cmd, const GpuBlendState* state) {
	cmd->bound_blend_state = state;
	// The state is per color target of the bound pipeline, so without one gpuSetPipeline applies it instead
	if(!cmd->bound_pipeline) return;

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

	// Each target's creation mask narrowed by this state's; always set, since a draw with it unset is invalid.
	std::vector<VkColorComponentFlags> masks; masks.reserve(count);
	for(uint32_t i = 0; i < count; ++i)
		masks.push_back(GPU::detail::mask2vulkan(
			cmd->bound_pipeline->color_write_masks[i] & state->descriptor.colorWriteMask));
	vkCmdSetColorWriteMaskEXT(cmd->command_buffer, 0, count, masks.data());
}

void gpuSetViewportEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin, float depth_min, float depth_max) {
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

void gpuSetScissorRectEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin) {
	VkRect2D scissor { // TODO: Should we support additional scissors?
		.offset = {origin.x, origin.y},
		.extent = {extent.x, extent.y},
	};
	vkCmdSetScissor(cmd->command_buffer, 0, 1, &scissor);
}

namespace GPU::detail {
	// Built per pass rather than cached on the texture, since each pass may attach a different mip and slice.
	inline VkImageView attachment_view(GpuCommandBuffer* cmd, const GpuTexture* texture, uint32_t mip, uint32_t slice) {
		auto queue = cmd->queue;
		// A 3D image is attached as the whole volume; everything else picks its slice as a layer
		bool volume = texture->descriptor.type == TEXTURE_3D;

		VkImageViewCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = texture->image,
			.viewType = volume ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D,
			.format = format2vulkan(texture->descriptor.format),
			.subresourceRange = {
				.aspectMask = texture_aspect(texture->descriptor.format),
				.baseMipLevel = mip,
				.levelCount = 1,
				.baseArrayLayer = volume ? 0 : slice,
				.layerCount = 1,
			}
		};
		VkImageView view = VK_NULL_HANDLE;
		VK_CHECK(vkCreateImageView(queue->device, &info, queue->callbacks, &view), VK_NULL_HANDLE);

		queue->views_in_flight.emplace_back(view, queue->command_submission_timeline_semaphore_next_value);
		return view;
	}

	inline void consume_available_semaphore(GpuCommandBuffer* cmd, const GpuTexture* texture) {
		if(!texture->available_semaphore) return;

		cmd->wait_semaphores.push_back(texture->available_semaphore);
		const_cast<GpuTexture*>(texture)->available_semaphore = VK_NULL_HANDLE;
	}
}

namespace GPU::detail {
	// Integer images can only be resolved by picking a sample, and floating point ones only by averaging. Filtering rules out integers, depth and 32 bit floats.
	inline bool format_is_integer(FORMAT format) {
		return !gpuFormatIsFilterableEXT(format) && !gpuFormatIsDepthStencilEXT(format)
			&& format != FORMAT_R32_FLOAT && format != FORMAT_RG32_FLOAT && format != FORMAT_RGBA32_FLOAT;
	}
}

void gpuBeginRenderPass(GpuCommandBuffer* cmd, const GpuRenderPassDesc& desc) {
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
	assert((!desc.colorAttachments.empty() || desc.depthAttachment || desc.stencilAttachment) && "A render pass needs at least one attachment");
	cmd->state = GpuCommandBuffer::RecordingRenderPass;

	GPU::detail::reclaim_views(cmd->queue);

	// From whichever attachment comes first, since a depth only pass may have no color attachment.
	uvec2 extent = {0, 0};

	std::vector<VkRenderingAttachmentInfo> color_attachments; color_attachments.reserve(desc.colorAttachments.size());
	for(auto& color: desc.colorAttachments) {
		GPU::detail::consume_available_semaphore(cmd, color.texture);

		GPU::detail::transition_texture(cmd->command_buffer, color.texture, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			color.mipLevel, color.slice, color.loadOp != LOAD_OP_LOAD
		);

		VkImageView resolve_view = VK_NULL_HANDLE;
		if(color.resolveTexture) {
			GPU::detail::consume_available_semaphore(cmd, color.resolveTexture);
			// The resolve overwrites the whole subresource, so its previous contents never matter
			GPU::detail::transition_texture(cmd->command_buffer, color.resolveTexture, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				color.mipLevel, color.slice, true
			);
			resolve_view = GPU::detail::attachment_view(cmd, color.resolveTexture, color.mipLevel, color.slice);
		}

		color_attachments.emplace_back(VkRenderingAttachmentInfo{
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = GPU::detail::attachment_view(cmd, color.texture, color.mipLevel, color.slice),
			.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.resolveMode = !resolve_view ? VK_RESOLVE_MODE_NONE : GPU::detail::format_is_integer(color.texture->descriptor.format) ? VK_RESOLVE_MODE_SAMPLE_ZERO_BIT : VK_RESOLVE_MODE_AVERAGE_BIT,
			.resolveImageView = resolve_view,
			.resolveImageLayout = resolve_view ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
			.loadOp = load2vulkan(color.loadOp),
			.storeOp = store2vulkan(color.storeOp),
			.clearValue = {
				.color = {
					.float32 = {color.clearValue.r, color.clearValue.g, color.clearValue.b, color.clearValue.a}
				}
			}
		});

		if(extent.x == 0) extent = GPU::detail::mip_extent(color.texture, color.mipLevel);
	}

	// When one texture carries both aspects only the depth half transitions it; a second transition would discard or self-barrier.
	bool stencil_shares_depth = desc.depthAttachment && desc.stencilAttachment
		&& desc.depthAttachment->texture == desc.stencilAttachment->texture;

	VkRenderingAttachmentInfo depth_attachment;
	if(desc.depthAttachment) {
		auto& attachment = *desc.depthAttachment;
		GPU::detail::consume_available_semaphore(cmd, attachment.texture);

		GPU::detail::transition_texture(cmd->command_buffer, attachment.texture, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			attachment.mipLevel, attachment.slice, attachment.loadOp != LOAD_OP_LOAD
		);

		depth_attachment = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = GPU::detail::attachment_view(cmd, attachment.texture, attachment.mipLevel, attachment.slice),
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.resolveMode = VK_RESOLVE_MODE_NONE,
			.resolveImageView = VK_NULL_HANDLE,
			.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.loadOp = load2vulkan(attachment.loadOp),
			.storeOp = store2vulkan(attachment.storeOp),
			.clearValue = {
				.depthStencil = {
					.depth = static_cast<float>(attachment.clearValue)
				}
			}
		};

		if(extent.x == 0) extent = GPU::detail::mip_extent(attachment.texture, attachment.mipLevel);
	}

	VkRenderingAttachmentInfo stencil_attachment;
	if(desc.stencilAttachment) {
		auto& attachment = *desc.stencilAttachment;
		if(!stencil_shares_depth) {
			GPU::detail::consume_available_semaphore(cmd, attachment.texture);

			GPU::detail::transition_texture(cmd->command_buffer, attachment.texture, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				attachment.mipLevel, attachment.slice, attachment.loadOp != LOAD_OP_LOAD
			);
		}

		stencil_attachment = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			// The same subresource the depth half named, when they share a texture: Vulkan wants both halves of a combined format pointing at one view
			.imageView = stencil_shares_depth
				? depth_attachment.imageView
				: GPU::detail::attachment_view(cmd, attachment.texture, attachment.mipLevel, attachment.slice),
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.resolveMode = VK_RESOLVE_MODE_NONE,
			.resolveImageView = VK_NULL_HANDLE,
			.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.loadOp = load2vulkan(attachment.loadOp),
			.storeOp = store2vulkan(attachment.storeOp),
			.clearValue = {
				.depthStencil = {
					.stencil = static_cast<uint32_t>(attachment.clearValue)
				}
			}
		};

		if(extent.x == 0) extent = GPU::detail::mip_extent(attachment.texture, attachment.mipLevel);
	}

	VkRenderingInfo info{
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = {
			.offset = {0, 0},
			.extent = {extent.x, extent.y},
		},
		.layerCount = 1,
		.colorAttachmentCount = static_cast<uint32_t>(color_attachments.size()),
		.pColorAttachments = color_attachments.data(),
		.pDepthAttachment = desc.depthAttachment ? &depth_attachment : nullptr,
		.pStencilAttachment = desc.stencilAttachment ? &stencil_attachment : nullptr,
	};
	vkCmdBeginRendering(cmd->command_buffer, &info);

	gpuSetViewportEXT(cmd, extent);
	gpuSetScissorRectEXT(cmd, extent);
}

void gpuEndRenderPass(GpuCommandBuffer* cmd, GpuOptionalRenderPassDesc desc) {
	vkCmdEndRendering(cmd->command_buffer);

	// What gets presented is the resolve target, when there is one
	if(desc) for(auto& color: desc->colorAttachments)
		GPU::detail::transition_texture(cmd->command_buffer, color.resolveTexture ? color.resolveTexture : color.texture, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			color.mipLevel, color.slice, false
		);

	cmd->state = GpuCommandBuffer::Recording;
}

struct GraphicsPipelinePushConstants {
	gpu* vertex;
	gpu* fragment;
	gpu* index;
	gpu* sampler_map;
};

namespace GPU::detail {
	VkIndexType index2vulkan(INDEX_TYPE_EXT index_type) {
		switch (index_type) {
		case INDEX_TYPE_UINT8: return VK_INDEX_TYPE_UINT8;
		case INDEX_TYPE_UINT16: return VK_INDEX_TYPE_UINT16;
		case INDEX_TYPE_UINT32: return VK_INDEX_TYPE_UINT32;
		}
		std::unreachable();
	}

	inline VkDeviceSize index_stride(INDEX_TYPE_EXT index_type) {
		switch (index_type) {
		case INDEX_TYPE_UINT8: return 1;
		case INDEX_TYPE_UINT16: return 2;
		case INDEX_TYPE_UINT32: return 4;
		}
		std::unreachable();
	}

	// Every allocation declares index usage (see gpuMalloc), so an address is already a buffer and offset vkCmdBindIndexBuffer can take.
	bool bind_index_buffer(GpuCommandBuffer* cmd, gpu* indicesGpu, INDEX_TYPE_EXT index_type) {
		auto indices = closest_buffer(cmd->queue, indicesGpu);
		assert(indices.buffer && "The index pointer doesn't lie inside any allocation");
		if(!indices.buffer) return false;

		// A misaligned offset would shift every index that follows, not just slow the draw.
		assert(indices.offset % index_stride(index_type) == 0
			&& "The index pointer isn't aligned to the size of one index");

		vkCmdBindIndexBuffer(cmd->command_buffer, indices.buffer, indices.offset, index2vulkan(index_type));
		return true;
	}
}

void gpuDrawIndexedInstanced(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* indices, uint32_t index_count, uint32_t instance_count, INDEX_TYPE_EXT index_type) {
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

	if(!GPU::detail::bind_index_buffer(cmd, indices, index_type)) return;
	vkCmdDrawIndexed(cmd->command_buffer, index_count, instance_count, 0, 0, 0);
}

void gpuDrawIndexedInstancedIndirect(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* indices, gpu* args, INDEX_TYPE_EXT index_type) {
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

	if(!GPU::detail::bind_index_buffer(cmd, indices, index_type)) return;

	auto arguments = GPU::detail::closest_buffer(cmd->queue, args);
	assert(arguments.buffer && "The indirect arguments don't lie inside any allocation");
	// Every argument struct between the provided pointer and the end of its allocation is drawn, matching what the WebGPU backend does with the same pointer
	auto count = (arguments.size - arguments.offset) / sizeof(VkDrawIndexedIndirectCommand);
	vkCmdDrawIndexedIndirect(cmd->command_buffer, arguments.buffer, arguments.offset, count, sizeof(VkDrawIndexedIndirectCommand));
}

void gpuDrawMeshlets(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, uvec3 dim) {
	// Reported rather than thrown: an exception cannot leave an extern "C" function. Programs check gpuGetCapabilitiesEXT().mesh_shaders before recording.
	if(!cmd->queue->capabilities.mesh_shaders) {
		GPU::detail::report(cmd, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuDrawMeshlets: this device was not set up with VK_EXT_mesh_shader, so nothing is drawn");
		return;
	}

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

void gpuDrawMeshletsIndirect(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, gpu* dim) {
	// See gpuDrawMeshlets for why this reports rather than throws.
	if(!cmd->queue->capabilities.mesh_shaders) {
		GPU::detail::report(cmd, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuDrawMeshletsIndirect: this device was not set up with VK_EXT_mesh_shader, so nothing is drawn");
		return;
	}

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

	auto dimensions = GPU::detail::closest_buffer(cmd->queue, dim);
	assert(dimensions.buffer && "The indirect dimensions don't lie inside any allocation");
	auto count = (dimensions.size - dimensions.offset) / sizeof(VkDrawMeshTasksIndirectCommandEXT); // VkDrawMeshTasksIndirectCommandEXT == uvec3
	vkCmdDrawMeshTasksIndirectEXT(cmd->command_buffer, dimensions.buffer, dimensions.offset, count, sizeof(VkDrawMeshTasksIndirectCommandEXT));
}

void gpuSetEnabledSamplersEXT(GpuCommandBuffer* cmd, GpuSamplerDescSpan enabled_samplers) {
	GPU::detail::bind_sampler_set(cmd, GPU::detail::ensure_sampler_set(cmd->queue, enabled_samplers));
}

namespace GPU::detail {
	// True once the presentation engine and the GPU are both done with a retired swapchain
	inline bool retired_swapchain_is_idle(GpuQueue* queue, const GpuSurface* surface, const GpuSurface::RetiredSwapchain& retired) {
		uint64_t reached = GPU::detail::finished_submission(queue);
		if(reached < retired.free_after_submission) return false; // A submission still names its views

		// Once the surface has presented more images than the retired swapchain had, the presentation engine has cycled past all of them.
		bool cycled_past = surface->presents_issued > retired.retired_at_present + retired.image_count;

		if(queue->present_fences_supported) {
			bool all_signalled = true;
			for(auto fence: retired.present_fences)
				if(vkGetFenceStatus(queue->device, fence) != VK_SUCCESS) { all_signalled = false; break; }
			// Fences answer exactly, but a failed present may never signal its fence, so the count must still release the swapchain.
			return all_signalled || cycled_past;
		}

		return cycled_past;
	}
}

// Called on acquire, so a long resize cleans up as it goes.
static void reclaim_retired_swapchains(GpuQueue* queue, GpuSurface* surface, bool force) {
	if(surface->retired.empty()) return;

	// The pile only grows if presents stopped happening, which is the one case where there is nothing left to wait for but the device itself
	if(force || surface->retired.size() > 4) {
		(void)vkDeviceWaitIdle(queue->device); // Fails only with the device lost, when destroying everything is all that is left
		for(auto& retired: surface->retired) {
			for(auto view: retired.views)
				vkDestroyImageView(queue->device, view, queue->callbacks);
			for(auto fence: retired.present_fences)
				vkDestroyFence(queue->device, fence, queue->callbacks);
			for(auto semaphore: retired.image_available)
				vkDestroySemaphore(queue->device, semaphore, queue->callbacks);
			for(auto semaphore: retired.render_finished)
				vkDestroySemaphore(queue->device, semaphore, queue->callbacks);
			if(retired.swapchain) vkb::destroy_swapchain(*retired.swapchain);
		}
		surface->retired.clear();

		// The device is idle, so even a fence a failed present never signalled is safe to let go of
		for(auto fence: surface->quarantined_present_fences)
			vkDestroyFence(queue->device, fence, queue->callbacks);
		surface->quarantined_present_fences.clear();
		return;
	}

	for(size_t i = surface->retired.size(); i--; ) {
		auto& retired = surface->retired[i];
		if(!GPU::detail::retired_swapchain_is_idle(queue, surface, retired)) continue;

		for(auto view: retired.views)
			vkDestroyImageView(queue->device, view, queue->callbacks);
		for(auto semaphore: retired.image_available)
			vkDestroySemaphore(queue->device, semaphore, queue->callbacks);
		for(auto semaphore: retired.render_finished)
			vkDestroySemaphore(queue->device, semaphore, queue->callbacks);
		for(auto fence: retired.present_fences) {
			// An unsignalled fence belongs to a failed present released by the frame count, so it can be neither reset nor destroyed yet.
			if(vkGetFenceStatus(queue->device, fence) == VK_SUCCESS && vkResetFences(queue->device, 1, &fence) == VK_SUCCESS)
				surface->free_present_fences.push_back(fence);
			else surface->quarantined_present_fences.push_back(fence);
		}
		if(retired.swapchain) vkb::destroy_swapchain(*retired.swapchain);
		surface->retired.erase(surface->retired.begin() + i);
	}
}

// Destroyed once idle rather than now, with frames still in flight against it (VUID-vkDestroySwapchainKHR-swapchain-01282).
static void retire_swapchain(GpuQueue* queue, GpuSurface* surface) {
	if(!surface->swapchain && surface->image_views.empty()) return;

	surface->retired.push_back(GpuSurface::RetiredSwapchain{
		.swapchain = std::move(surface->swapchain),
		.views = std::move(surface->image_views),
		.present_fences = std::move(surface->present_fences),
		.image_available = std::move(surface->image_available_semaphores),
		.render_finished = std::move(surface->render_finished_semaphores),
		// Everything queued up to now may name one of those views
		.free_after_submission = queue->command_submission_timeline_semaphore_next_value - 1,
		.retired_at_present = surface->presents_issued,
		.image_count = static_cast<uint32_t>(surface->images.size()),
	});
	surface->swapchain = nullptr;
	surface->image_views.clear();
	surface->present_fences.clear();
	// Emptied rather than reused, so the acquire and present paths build a fresh set for the new swapchain and nothing carries a stale signal across
	surface->image_available_semaphores.clear();
	surface->render_finished_semaphores.clear();
	surface->image_available_free_after.clear();
	surface->render_finished_free_after.clear();

	reclaim_retired_swapchains(queue, surface, false);
}

void gpuFreeSurfaceEXT(GpuQueue* queue, GpuSurface* surface) {
	retire_swapchain(queue, surface);
	// Nothing is going to present again, so there is no later point at which the pile would drain
	reclaim_retired_swapchains(queue, surface, true);

	// The semaphores went with the retired swapchain; only the recycled fences are the surface's own
	for(auto fence: surface->free_present_fences)
		vkDestroyFence(queue->device, fence, queue->callbacks);
	surface->~GpuSurface();
	queue->cpu_allocator(surface, 0);
}

namespace GPU::detail {
	// For reporting back what a swapchain settled on; formats the API can't name come back as FORMAT_NONE. Inverting format2vulkan keeps the two from drifting apart.
	inline FORMAT vulkan2format(VkFormat format) {
		for(int named = FORMAT_NONE + 1; named <= FORMAT_D32_FLOAT_S8_UINT; ++named)
			if(format2vulkan((FORMAT)named) == format) return (FORMAT)named;
		return FORMAT_NONE;
	};

	// PRESENT_MODE_BEST_AVAILABLE has no Vulkan spelling, so it is resolved against what the surface supports before this is reached (see gpuSurfaceReconfigureEXT)
	inline VkPresentModeKHR present2vulkan(PRESENT_MODE mode) {
		switch(mode) {
		case PRESENT_MODE_IMMEDIATE:
			return VK_PRESENT_MODE_IMMEDIATE_KHR;
		case PRESENT_MODE_FIFO_RELAXED:
			return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
		case PRESENT_MODE_MAILBOX:
			return VK_PRESENT_MODE_MAILBOX_KHR;
		default:
			return VK_PRESENT_MODE_FIFO_KHR;
		}
	};

	// Best first. Shared by gpuSurfaceReconfigureEXT and gpuGetSurfaceCapabilitiesEXT, so presentModes[0] is what BEST_AVAILABLE resolves to; immediate trails the tear free modes.
	constexpr static PRESENT_MODE PRESENT_MODE_PREFERENCE[] = {
		PRESENT_MODE_MAILBOX, PRESENT_MODE_FIFO_RELAXED, PRESENT_MODE_FIFO, PRESENT_MODE_IMMEDIATE
	};

	// Likewise for the presentation mode the swapchain ended up with, which resolve_present_mode always picks from this list
	inline PRESENT_MODE vulkan2presentMode(VkPresentModeKHR mode) {
		auto named = std::ranges::find_if(PRESENT_MODE_PREFERENCE, [mode](PRESENT_MODE named) { return present2vulkan(named) == mode; });
		return named != std::ranges::end(PRESENT_MODE_PREFERENCE) ? *named : PRESENT_MODE_FIFO;
	};

	// Resolved before the swapchain is built because VkSwapchainPresentModesCreateInfoKHR has to name it; fifo is the floor every surface supports.
	inline VkPresentModeKHR resolve_present_mode(VkPhysicalDevice gpu, VkSurfaceKHR surface, PRESENT_MODE requested) {
		// A query that fails leaves no modes, so the fifo floor below is chosen
		uint32_t count = 0;
		if(vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &count, nullptr) != VK_SUCCESS) count = 0;
		std::vector<VkPresentModeKHR> modes(count);
		if(count && vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &count, modes.data()) < VK_SUCCESS) count = 0;
		modes.resize(count);

		auto supported = [&modes](PRESENT_MODE mode) {
			return std::ranges::find(modes, present2vulkan(mode)) != modes.end();
		};

		if(requested != PRESENT_MODE_BEST_AVAILABLE && supported(requested))
			return present2vulkan(requested);
		// Fifo is in the list and supported by every surface, so the fallback only covers a failed query
		auto best = std::ranges::find_if(PRESENT_MODE_PREFERENCE, supported);
		return best != std::ranges::end(PRESENT_MODE_PREFERENCE) ? present2vulkan(*best) : VK_PRESENT_MODE_FIFO_KHR;
	}

	inline TEXTURE_USAGE_FLAGS vulkan2usage(VkImageUsageFlags usageFlags) {
		uint32_t result = 0;
		for(uint32_t bit = USAGE_SAMPLED; bit <= USAGE_TRANSFER_DST; bit <<= 1)
			if(usageFlags & usage2vulkan((TEXTURE_USAGE_FLAGS)bit)) result |= bit;
		return (TEXTURE_USAGE_FLAGS)result;
	};
}

void gpuSurfaceReconfigureEXT(GpuQueue* queue, GpuSurface* surface, const GpuSurfaceDescriptor& desc) {
	auto previous = surface->descriptor;
	surface->descriptor = desc;
	// A presentable texture is one 2D image without mips, layers or multisampling, as the WebGPU backend also reports.
	surface->descriptor.texture.type = TEXTURE_2D;
	surface->descriptor.texture.mipCount = 1;
	surface->descriptor.texture.sampleCount = 1;
	surface->descriptor.texture.layerCount = 1;
	surface->descriptor.texture.dimensions.z = 1;
	assert(surface->descriptor.texture.dimensions.x > 0 && surface->descriptor.texture.dimensions.y > 0
		&& "A surface can't be configured for a zero sized window");

	auto builder = vkb::SwapchainBuilder(queue->gpu, queue->device, surface->surface, queue->queue_family);
	if(surface->swapchain)
		builder.set_old_swapchain(*surface->swapchain);

	// Resolved here because VkSwapchainPresentModesCreateInfoKHR must name it, in the order gpuGetSurfaceCapabilitiesEXT reports; the builder's fallbacks remain as a safety net.
	auto present_mode = GPU::detail::resolve_present_mode(queue->gpu, surface->surface, desc.presentMode);
	builder.set_desired_present_mode(present_mode)
		.add_fallback_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
		.add_fallback_present_mode(VK_PRESENT_MODE_FIFO_RELAXED_KHR)
		.add_fallback_present_mode(VK_PRESENT_MODE_FIFO_KHR);

	// Exactly the creation mode, since mode changes rebuild the swapchain; best practices validation flags leaving this out with swapchain_maintenance1 enabled.
	VkSwapchainPresentModesCreateInfoKHR present_modes {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_KHR,
		.presentModeCount = 1,
		.pPresentModes = &present_mode,
	};
	if(queue->present_fences_supported) builder.add_pNext(&present_modes);
	// Naming VK_FORMAT_UNDEFINED would match nothing and fall back to the first format listed, not the preferred one.
	if(surface->descriptor.texture.format != FORMAT_NONE)
		builder.set_desired_format({
			GPU::detail::format2vulkan(surface->descriptor.texture.format),
			VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
		});
	auto swap = builder.set_desired_extent(surface->descriptor.texture.dimensions.x, surface->descriptor.texture.dimensions.y)
		.set_allocation_callbacks(queue->callbacks)
		.set_composite_alpha_flags(surface->descriptor.opaque ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
		.set_image_array_layer_count(surface->descriptor.texture.layerCount)
		.add_image_usage_flags(GPU::detail::usage2vulkan(surface->descriptor.texture.usage))
		.build();
	if(!swap) {
		// Nothing was created, so the surface keeps its previous swapchain, and reports the configuration that one has
		errno = swap.error().value();
		surface->descriptor = previous;
		return;
	}

	// An unpresented image's signalled acquire semaphore retires with its swapchain, so there is nothing to unwind.
	surface->image_acquired = false;
	surface->current_semaphore = uint32_t(-1);

	retire_swapchain(queue, surface);
	surface->swapchain = std::make_shared<vkb::Swapchain>(std::move(*swap));
	// What the swapchain settled on, which a FORMAT_NONE request needs in order to build a matching pipeline.
	surface->descriptor.texture.format = GPU::detail::vulkan2format(surface->swapchain->image_format);
	surface->descriptor.presentMode = GPU::detail::vulkan2presentMode(surface->swapchain->present_mode);
	surface->descriptor.texture.usage = GPU::detail::vulkan2usage(surface->swapchain->image_usage_flags);
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

GpuSurfaceCapabilities gpuGetSurfaceCapabilitiesEXT(GpuQueue* queue, GpuSurface* surface) {
	GpuSurfaceCapabilities out = {};

	// Empty lists, as documented, when the platform can't be asked
	uint32_t count = 0;
	if(vkGetPhysicalDeviceSurfaceFormatsKHR(queue->gpu, surface->surface, &count, nullptr) != VK_SUCCESS) return out;
	std::vector<VkSurfaceFormatKHR> formats(count);
	if(vkGetPhysicalDeviceSurfaceFormatsKHR(queue->gpu, surface->surface, &count, formats.data()) < VK_SUCCESS) return out;
	formats.resize(count);

	// The swapchain is only built for nonlinear sRGB, so formats offered in other color spaces are unusable.
	auto usable = [](VkSurfaceFormatKHR format) {
		return format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
			&& GPU::detail::vulkan2format(format.format) != FORMAT_NONE;
	};
	auto append = [&out](VkFormat format) {
		auto named = GPU::detail::vulkan2format(format);
		auto listed = std::span(out.formats, out.formatCount);
		if(std::ranges::find(listed, named) == listed.end() && out.formatCount < GPU_MAX_SURFACE_FORMATS)
			out.formats[out.formatCount++] = named;
	};

	// vk-bootstrap prefers these two when no format is named, so listing them first makes formats[0] what FORMAT_NONE resolves to.
	for(auto preferred: {VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB})
		for(auto format: formats)
			if(format.format == preferred && usable(format))
				append(format.format);
	for(auto format: formats)
		if(usable(format))
			append(format.format);

	count = 0;
	if(vkGetPhysicalDeviceSurfacePresentModesKHR(queue->gpu, surface->surface, &count, nullptr) != VK_SUCCESS) count = 0;
	std::vector<VkPresentModeKHR> modes(count);
	if(count && vkGetPhysicalDeviceSurfacePresentModesKHR(queue->gpu, surface->surface, &count, modes.data()) < VK_SUCCESS) count = 0;
	modes.resize(count);

	// PRESENT_MODE_PREFERENCE is the order gpuSurfaceReconfigureEXT resolves through, so presentModes[0] is what PRESENT_MODE_BEST_AVAILABLE will land on
	for(auto mode: GPU::detail::PRESENT_MODE_PREFERENCE)
		if(std::ranges::find(modes, GPU::detail::present2vulkan(mode)) != modes.end()
			&& out.presentModeCount < GPU_MAX_SURFACE_PRESENT_MODES)
			out.presentModes[out.presentModeCount++] = mode;

	VkSurfaceCapabilitiesKHR caps {};
	if(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(queue->gpu, surface->surface, &caps) != VK_SUCCESS) caps = {};
	// Inherit leaves the blending to whatever the native window was set up for, which is not a promise the compositor will honor an alpha channel
	out.supportsTransparency = caps.supportedCompositeAlpha
		& (VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR | VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR);

	return out;
}

GpuSurfaceDescriptor gpuSurfaceGetConfigurationEXT(const GpuSurface* surface) {
	return surface->descriptor;
}

namespace GPU::detail {
	// Zero means the submission waited for never happened.
	inline void wait_for_submission(GpuQueue* queue, uint64_t value) {
		if(value == 0) return;

		uint64_t reached = GPU::detail::finished_submission(queue);
		if(reached >= value) return;

		VkSemaphoreWaitInfo info {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
			.semaphoreCount = 1,
			.pSemaphores = &queue->command_submission_timeline_semaphore,
			.pValues = &value,
		};
		(void)vkWaitSemaphores(queue->device, &info, UINT64_MAX); // Fails only with the device lost, after which nothing this guards matters
	}

	// Replaces `semaphores` with `count` new binary semaphores; on failure it is left empty, with errno set
	inline bool create_semaphores(GpuQueue* queue, std::vector<VkSemaphore>& semaphores, size_t count) {
		semaphores.clear();
		VkSemaphoreCreateInfo info { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
		for(size_t i = 0; i < count; ++i) {
			VkSemaphore semaphore = VK_NULL_HANDLE;
			if(auto result = vkCreateSemaphore(queue->device, &info, queue->callbacks, &semaphore); result != VK_SUCCESS) {
				for(auto created: semaphores) vkDestroySemaphore(queue->device, created, queue->callbacks);
				semaphores.clear();
				errno = result;
				return false;
			}
			semaphores.push_back(semaphore);
		}
		return true;
	}
}

const GpuTexture* gpuSurfaceNextTextureEXT(GpuQueue* queue, GpuSurface* surface) {
	// Holding two acquired images can exceed the surface's limit and loses the held one's semaphore; presenting releases it.
	if(surface->image_acquired) {
		GPU::detail::report(queue, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuSurfaceNextTextureEXT: an image is already acquired and not yet presented, so no new one is handed out; present the one you have first");
		errno = VK_NOT_READY;
		return nullptr;
	}

	reclaim_retired_swapchains(queue, surface, false);

	// A fresh set per swapchain generation; the previous set retired with its swapchain.
	if(surface->image_available_semaphores.size() != surface->images.size()) {
		if(!GPU::detail::create_semaphores(queue, surface->image_available_semaphores, surface->images.size())) return nullptr;
		surface->image_available_free_after.assign(surface->images.size(), 0);
		surface->semaphore_counter = -1;
	}

	surface->semaphore_counter = (surface->semaphore_counter + 1) % surface->image_available_semaphores.size();
	// The frame that last used this semaphore must finish before it can be signalled again.
	GPU::detail::wait_for_submission(queue, surface->image_available_free_after[surface->semaphore_counter]);

	auto acquired = vkAcquireNextImageKHR(queue->device, surface->swapchain->swapchain, UINT64_MAX, surface->image_available_semaphores[surface->semaphore_counter], VK_NULL_HANDLE, &surface->current_image);
	// A suboptimal image can still be rendered and presented; reconfiguring is the caller's call, as on WebGPU.
	if(acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
		errno = acquired;
		return nullptr;
	}
	if(acquired == VK_SUBOPTIMAL_KHR) errno = SURFACE_SUBOPTIMAL;

	surface->image_acquired = true;
	surface->current_semaphore = surface->semaphore_counter;
	surface->images[surface->current_image].available_semaphore = surface->image_available_semaphores[surface->semaphore_counter];
	return &surface->images[surface->current_image];
}

namespace GPU::detail {
	// Nothing rendered into the image: its acquire is still signalled and its layout UNDEFINED, so a submission of its own waits and transitions.
	inline uint64_t drain_acquire_semaphore(GpuQueue* queue, GpuSurface* surface) {
		if(surface->current_semaphore == uint32_t(-1)) return 0;
		auto& image = surface->images[surface->current_image];
		if(image.available_semaphore == VK_NULL_HANDLE) return 0; // Something waited on it already

		auto cmd = gpuStartCommandRecording(queue);
		GPU::detail::consume_available_semaphore(cmd, &image);
		GPU::detail::transition_texture(cmd->command_buffer, &image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, 0, 0,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, true);
		auto drained_after = gpuSubmit(queue, {&cmd, 1});
		surface->image_available_free_after[surface->current_semaphore] = drained_after;
		return drained_after;
	}
}

void gpuSurfacePresentEXT(GpuQueue* queue, GpuSurface* surface, uint64_t wait_submission_index) {
	// Nothing was acquired, so there is no image to hand back
	if(!surface->image_acquired) {
		GPU::detail::report(queue, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuSurfacePresentEXT: no image is currently acquired, so nothing is presented; call gpuSurfaceNextTextureEXT first");
		return;
	}
	surface->image_acquired = false;

	// A drained image has a newer submission of its own for the present to wait on; otherwise the frame's submission frees the acquire.
	if(auto drained = GPU::detail::drain_acquire_semaphore(queue, surface))
		wait_submission_index = drained;
	else if(surface->current_semaphore != uint32_t(-1))
		surface->image_available_free_after[surface->current_semaphore] =
			wait_submission_index == NO_SUBMISSION_WAIT
				? queue->command_submission_timeline_semaphore_next_value - 1
				: wait_submission_index;
	surface->current_semaphore = uint32_t(-1);

	if(wait_submission_index != NO_SUBMISSION_WAIT) {
		if(surface->render_finished_semaphores.size() != surface->images.size()) {
			if(!GPU::detail::create_semaphores(queue, surface->render_finished_semaphores, surface->images.size())) return;
			surface->render_finished_free_after.assign(surface->images.size(), 0);
		}

		// Waited on by the presentation engine, so the frame that signalled it must finish before it is signalled again.
		GPU::detail::wait_for_submission(queue, surface->render_finished_free_after[surface->current_image]);
		surface->render_finished_free_after[surface->current_image] = wait_submission_index;

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

	// Only a present fence says when the engine is done with an image, so a retired swapchain can be freed without draining the device.
	VkSwapchainPresentFenceInfoKHR fence_info { .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR };
	VkFence present_fence = VK_NULL_HANDLE;
	if(queue->present_fences_supported) {
		if(!surface->free_present_fences.empty()) {
			present_fence = surface->free_present_fences.back();
			surface->free_present_fences.pop_back();
		} else {
			VkFenceCreateInfo create { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
			if(vkCreateFence(queue->device, &create, queue->callbacks, &present_fence) != VK_SUCCESS)
				present_fence = VK_NULL_HANDLE;
		}

		if(present_fence) {
			fence_info.swapchainCount = 1;
			fence_info.pFences = &present_fence;
			fence_info.pNext = info.pNext;
			info.pNext = &fence_info;
		}
	}

	auto presented = vkQueuePresentKHR(queue->queue, &info);
	// Recorded even when the present failed: the fence was consumed, so the swapchain still has to account for it.
	if(present_fence) surface->present_fences.push_back(present_fence);
	++surface->presents_issued;

	if(presented != VK_SUCCESS && presented != VK_SUBOPTIMAL_KHR) errno = presented;
}

const GpuSemaphore* gpuGetSubmissionSemaphoreEXT(GpuQueue* queue) {
	return (GpuSemaphore*)&queue->command_submission_timeline_semaphore;
}

void gpuWaitIdleEXT(GpuQueue* queue) {
	if(auto result = vkDeviceWaitIdle(queue->device); result != VK_SUCCESS) errno = result;
}

void gpuSyncMemoryEXT(GpuCommandBuffer* cmd, gpu* mem) {
	// Do nothing!
}

void gpuSyncMemoryImmediateEXT(GpuQueue* queue, gpu* mem) {
	// Do nothing!
}
