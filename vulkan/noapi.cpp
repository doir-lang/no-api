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
#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <cstdint>
#include <memory>
#include <set>
#include <stdexcept>
#include <sys/types.h>
#include <utility>
#include <vector>




namespace GPU::detail {
	// The Vulkan packed formats read their components from the high bits down, so a WebGPU style
	// name maps onto the reversed Vulkan one (rgb10a2unorm is A2B10G10R10, rg11b10ufloat is B10G11R11).
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
		// WebGPU's depth24plus only promises 24 bits of depth; X8_D24 is the Vulkan format that
		// makes the same promise, and a driver without it reports D32_SFLOAT support instead
		case FORMAT_D24_PLUS: return VK_FORMAT_X8_D24_UNORM_PACK32;
		case FORMAT_D24_PLUS_S8_UINT: return VK_FORMAT_D24_UNORM_S8_UINT;
		case FORMAT_D32_FLOAT: return VK_FORMAT_D32_SFLOAT;
		case FORMAT_D32_FLOAT_S8_UINT: return VK_FORMAT_D32_SFLOAT_S8_UINT;
		}
		std::unreachable();
	};

	// The inverse of format2vulkan, for reporting back whatever a swapchain settled on when the
	// request left the choice open. Anything the rest of the API can't name comes back as FORMAT_NONE.
	inline FORMAT vulkan2format(VkFormat format) {
		switch(format) {
		case VK_FORMAT_R8_UNORM: return FORMAT_R8_UNORM;
		case VK_FORMAT_R8_SNORM: return FORMAT_R8_SNORM;
		case VK_FORMAT_R8_UINT: return FORMAT_R8_UINT;
		case VK_FORMAT_R8_SINT: return FORMAT_R8_SINT;

		case VK_FORMAT_R16_UNORM: return FORMAT_R16_UNORM;
		case VK_FORMAT_R16_SNORM: return FORMAT_R16_SNORM;
		case VK_FORMAT_R16_UINT: return FORMAT_R16_UINT;
		case VK_FORMAT_R16_SINT: return FORMAT_R16_SINT;
		case VK_FORMAT_R16_SFLOAT: return FORMAT_R16_FLOAT;
		case VK_FORMAT_R8G8_UNORM: return FORMAT_RG8_UNORM;
		case VK_FORMAT_R8G8_SNORM: return FORMAT_RG8_SNORM;
		case VK_FORMAT_R8G8_UINT: return FORMAT_RG8_UINT;
		case VK_FORMAT_R8G8_SINT: return FORMAT_RG8_SINT;

		case VK_FORMAT_R32_UINT: return FORMAT_R32_UINT;
		case VK_FORMAT_R32_SINT: return FORMAT_R32_SINT;
		case VK_FORMAT_R32_SFLOAT: return FORMAT_R32_FLOAT;
		case VK_FORMAT_R16G16_UNORM: return FORMAT_RG16_UNORM;
		case VK_FORMAT_R16G16_SNORM: return FORMAT_RG16_SNORM;
		case VK_FORMAT_R16G16_UINT: return FORMAT_RG16_UINT;
		case VK_FORMAT_R16G16_SINT: return FORMAT_RG16_SINT;
		case VK_FORMAT_R16G16_SFLOAT: return FORMAT_RG16_FLOAT;
		case VK_FORMAT_R8G8B8A8_UNORM: return FORMAT_RGBA8_UNORM;
		case VK_FORMAT_R8G8B8A8_SRGB: return FORMAT_RGBA8_SRGB;
		case VK_FORMAT_R8G8B8A8_SNORM: return FORMAT_RGBA8_SNORM;
		case VK_FORMAT_R8G8B8A8_UINT: return FORMAT_RGBA8_UINT;
		case VK_FORMAT_R8G8B8A8_SINT: return FORMAT_RGBA8_SINT;
		case VK_FORMAT_B8G8R8A8_UNORM: return FORMAT_BGRA8_UNORM;
		case VK_FORMAT_B8G8R8A8_SRGB: return FORMAT_BGRA8_SRGB;

		case VK_FORMAT_A2B10G10R10_UINT_PACK32: return FORMAT_RGB10_A2_UINT;
		case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return FORMAT_RGB10_A2_UNORM;
		case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return FORMAT_RG11B10_UFLOAT;
		case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return FORMAT_RGB9E5_UFLOAT;

		case VK_FORMAT_R32G32_UINT: return FORMAT_RG32_UINT;
		case VK_FORMAT_R32G32_SINT: return FORMAT_RG32_SINT;
		case VK_FORMAT_R32G32_SFLOAT: return FORMAT_RG32_FLOAT;
		case VK_FORMAT_R16G16B16A16_UNORM: return FORMAT_RGBA16_UNORM;
		case VK_FORMAT_R16G16B16A16_SNORM: return FORMAT_RGBA16_SNORM;
		case VK_FORMAT_R16G16B16A16_UINT: return FORMAT_RGBA16_UINT;
		case VK_FORMAT_R16G16B16A16_SINT: return FORMAT_RGBA16_SINT;
		case VK_FORMAT_R16G16B16A16_SFLOAT: return FORMAT_RGBA16_FLOAT;

		case VK_FORMAT_R32G32B32A32_UINT: return FORMAT_RGBA32_UINT;
		case VK_FORMAT_R32G32B32A32_SINT: return FORMAT_RGBA32_SINT;
		case VK_FORMAT_R32G32B32A32_SFLOAT: return FORMAT_RGBA32_FLOAT;

		case VK_FORMAT_S8_UINT: return FORMAT_S8_UINT;
		case VK_FORMAT_D16_UNORM: return FORMAT_D16_UNORM;
		case VK_FORMAT_X8_D24_UNORM_PACK32: return FORMAT_D24_PLUS;
		case VK_FORMAT_D24_UNORM_S8_UINT: return FORMAT_D24_PLUS_S8_UINT;
		case VK_FORMAT_D32_SFLOAT: return FORMAT_D32_FLOAT;
		case VK_FORMAT_D32_SFLOAT_S8_UINT: return FORMAT_D32_FLOAT_S8_UINT;

		default:
			return FORMAT_NONE;
		}
	};

	// PRESENT_MODE_BEST_AVAILABLE has no Vulkan spelling, so it is resolved against what the surface
	// supports before this is reached (see gpuSurfaceReconfigureEXT)
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

	// Likewise for the presentation mode the swapchain ended up with
	inline PRESENT_MODE vulkan2presentMode(VkPresentModeKHR mode) {
		switch(mode) {
		case VK_PRESENT_MODE_IMMEDIATE_KHR:
			return PRESENT_MODE_IMMEDIATE;
		case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
			return PRESENT_MODE_FIFO_RELAXED;
		case VK_PRESENT_MODE_MAILBOX_KHR:
			return PRESENT_MODE_MAILBOX;
		default:
			return PRESENT_MODE_FIFO;
		}
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

	inline TEXTURE_USAGE_FLAGS vulkan2usage(VkImageUsageFlags usageFlags) {
		uint32_t result = 0;
		if (usageFlags & VK_IMAGE_USAGE_SAMPLED_BIT)
			result |= USAGE_SAMPLED;
		if (usageFlags & VK_IMAGE_USAGE_STORAGE_BIT)
			result |= USAGE_STORAGE;
		if (usageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
			result |= USAGE_COLOR_ATTACHMENT;
		if (usageFlags & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
			result |= USAGE_DEPTH_STENCIL_ATTACHMENT;
		if (usageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
			result |= USAGE_TRANSFER_SRC;
		if (usageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
			result |= USAGE_TRANSFER_DST;
		return (TEXTURE_USAGE_FLAGS)result;
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

	// Where a GPU address lives: which buffer backs it, how far into that buffer it sits, and the
	// base of the allocation holding it. The base is what the queue's other maps (allocations,
	// gpu2image, descriptor_heaps) are keyed by, so it is what a caller looks up with.
	struct BufferLocation {
		VkBuffer buffer = VK_NULL_HANDLE; // Null when the address landed outside every allocation
		VkDeviceSize offset = 0; // Byte offset of the address within `buffer`
		VkDeviceAddress base = 0; // Base address of the allocation holding it
		VkDeviceSize size = 0; // Size of that whole allocation
	};

	// The allocation holding `addr` is the one whose base is the greatest at or below it and whose
	// size actually reaches it. `allocations` is ordered by base address, so upper_bound lands one
	// past that allocation and stepping back once names it -- no candidate before it can be closer.
	//
	// Nothing here inserts into `allocations`: an address that belongs to no allocation comes back
	// with a null buffer rather than silently adding a null entry the next search would then find.
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
}




/**
 * NOAPI_BACKEND_MODULE – This backend's half of the shader ABI, as the Slang module the
 * portable `noapi` module (see slang_compiler.hpp) is written against.
 *
 * Vulkan is the easy half: the raw bits the portable module carries a pointer as really are
 * a device address, so a load is an ordinary dereference, which Slang lowers to the
 * PhysicalStorageBuffer64 addressing model. The root pointers arrive as push data
 * (vkCmdPushDataEXT), laid out to match ComputePipelinePushConstants and
 * GraphicsPipelinePushConstants below, and textures and samplers are indexed straight out
 * of the hardware descriptor heaps VK_EXT_descriptor_heap provides.
 *
 * One source rather than the two prologues this replaced: GPU_COMPUTE picks which set of
 * root pointers was pushed, and is set by compile_shader.
 */
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

// An address arrives as the portable module carries it -- the 64 bits of it, low word first --
// and here those bits are a device address, so the whole memory ABI is a pack and a
// dereference. Everything wider than a word is GpuPtr<T>'s business, not this module's.
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

// The lookup table ensure_sampler_set built: one word per packed GpuSamplerDesc, holding
// the slot that description landed in, or 0 (the default sampler) when it was never enabled
public uint gpuBackendSamplerSlot(uint packed) {
	if(gpu_pc.sampler_map == 0) return 0u;
	if(packed > 0x1ffu) return 0u;
	return *((uint*)(gpu_pc.sampler_map + uint64_t(packed) * 4u));
}

// NOTE: Only 2D textures can be sampled through here. The descriptor at a heap index is
// opaque hardware bits, so unlike WebGPU (where the heap entry is a struct the shader can
// read a dimensionality out of) there is nothing to switch on, and the view type has to be
// named statically. `layer` is therefore ignored. Layered, 3D and cube sampling needs
// typed entry points this backend does not offer yet.
public float4 gpuBackendSample(uint heap_index, uint slot, float3 uv, uint layer, float mip) {
	// The heap a program builds is an array of GpuTextureDescriptor, which is a fixed 64 bytes,
	// while this indexes in units of whatever the driver lays an image descriptor out in. The ratio
	// between the two is baked in at compile time (see heap_stride_ratio); it is 1 where the driver
	// uses the full 64 and 2 where it uses 32, leaving the back half of each slot unused.
	Texture2D<float4> texture = ResourceDescriptorHeap[heap_index * GPU_HEAP_STRIDE_RATIO];
	SamplerState sampler = SamplerDescriptorHeap[slot];
	return texture.SampleLevel(sampler, uv.xy, mip);
}
)";

/**
 * compile_shader – Turn the Slang source a pipeline was handed into SPIR-V for one stage.
 *
 * Stores are always reachable here: unlike WebGPU, Vulkan is happy to have a shader write
 * through a pointer from any stage.
 *
 * @return The SPIR-V words, or nothing (having reported why) if the shader did not compile.
 */
static std::optional<std::string> compile_shader(GpuQueue* queue, GpuByteSpan ir, GPU::shaders::SHADER_STAGE stage) {
	auto compute = stage == GPU::shaders::SHADER_STAGE::COMPUTE;
	std::string error;
	auto code = GPU::shaders::compile(SLANG_SPIRV, NOAPI_BACKEND_MODULE,
		std::string_view((const char*)ir.data(), ir.size()), stage, {
			{"GPU_COMPUTE", compute ? "1" : "0"},
			{"GPU_GRAPHICS", compute ? "0" : "1"},
			{"GPU_STORES", "1"},
			// Device dependent, so it is a macro rather than a constant in the module source. The
			// compiler's session cache keys on the macros, so two devices disagreeing about it get
			// their own sessions rather than one another's code.
			{"GPU_HEAP_STRIDE_RATIO", std::to_string(queue->heap_stride_ratio)},
		}, error);

	if(!code) {
		GPU::shaders::report_diagnostic(error);
		errno = VK_ERROR_INITIALIZATION_FAILED;
	}
	return code;
}


thread_local static VkDebugUtilsMessageSeverityFlagBitsEXT severity_filter;

void gpuDefaultErrorCallbackEXT(void* queue, int type, GpuStringView message) {
	auto mt = vkb::to_string_message_type(type);
	printf("[%s]\n%.*s\n", mt, (int)message.count, message.ptr);
}

// Copies why the setup gave up into the caller's buffer (which is allowed to be absent) and
// reports the failure
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
	severity_filter = severity_filter_set;

	// Instance
	vkb::InstanceBuilder instance_builder;
	instance_builder.set_app_name("NoAPI")
		.set_engine_name("NoAPI")
		.enable_extensions(instance_extensions.size(), instance_extensions.data())
		.request_validation_layers(debug)
		.require_api_version(1, 4, 0);
	// What VK_KHR_swapchain_maintenance1 is built on; without it that device extension cannot be
	// enabled. Asked for rather than required, and the availability check goes through
	// vkb::SystemInfo rather than vkEnumerateInstanceExtensionProperties because volk has not
	// loaded the instance entry points yet -- that happens in gpuCreateQueue, after this returns.
	// Naming an absent extension to the instance builder would fail instance creation outright.
	if(auto system = vkb::SystemInfo::get_system_info(); system)
		// Both, and only together: VK_KHR_surface_maintenance1 depends on the capabilities query,
		// and naming it without that dependency is what an instance is rejected for
		if(system->is_extension_available("VK_KHR_surface_maintenance1")
				&& system->is_extension_available("VK_KHR_get_surface_capabilities2")) {
			instance_builder.enable_extension("VK_KHR_get_surface_capabilities2");
			instance_builder.enable_extension("VK_KHR_surface_maintenance1");
		}
	for(auto layer: extra_layers)
		instance_builder.enable_layer(layer);
	if(debug) instance_builder.set_debug_callback(+[](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageTypes, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) -> VkBool32 {
			if(messageSeverity < severity_filter) return VK_FALSE;

			auto error_callback = (GpuErrorCallbackEXT)pUserData;
			error_callback(nullptr, messageTypes, {pCallbackData->pMessage, strlen(pCallbackData->pMessage)});
			return VK_FALSE;
		}).set_debug_callback_user_data_pointer((void*)error_callback);//.use_default_debug_messenger();
	auto inst = instance_builder.build();
	if (!inst) return setup_failed(out_error, out_error_capacity, inst.error().message());
	auto instance = inst.value();
	out.instance = instance.instance;
	out.messenger = instance.debug_messenger;

	out.surface = surface_loader(out.instance, surface_loader_userdata);

	auto required = gpuRequiredVulkanDeviceExtensionsEXT();
	std::vector<const char*> extensions(required.begin(), required.end());
	extensions.insert(extensions.end(), device_extensions.begin(), device_extensions.end());

	// Physical Device
	vkb::PhysicalDeviceSelector gpu_selector{instance};
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
	if (!phys) return setup_failed(out_error, out_error_capacity, phys.error().message());
	auto gpu = phys.value();
	out.gpu = gpu.physical_device;

	// The extensions the backend uses if they are there. Enabled through the feature check rather
	// than by name alone, so an extension whose feature bit is missing isn't turned on half way.
	VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR swapchain_maintenance {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR,
		.swapchainMaintenance1 = true
	};
	if(gpu.enable_extension_features_if_present(swapchain_maintenance))
		gpu.enable_extension_if_present("VK_KHR_swapchain_maintenance1");


	// Logical Device
	vkb::DeviceBuilder device_builder{gpu};
	auto dev = device_builder.add_pNext(gpuRequiredVulkanDeviceCreateInfoPnextEXT()).build();
	if (!dev) return setup_failed(out_error, out_error_capacity, dev.error().message());
	auto device = dev.value();
	out.device = device.device;

	out.graphics_queue = device.get_queue(vkb::QueueType::graphics).value();
	out.graphics_queue_family = device.get_queue_index(vkb::QueueType::graphics).value();

	*out_default = out;
	return true;
}



static std::optional<GpuSemaphore> gpuCreateSemaphoreImpl(GpuQueue* queue, uint64_t init_value);

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
		.vulkanApiVersion = VK_API_VERSION_1_4,
	};
	if(vmaCreateAllocator(&vma_info, &out->gpu_allocator) != VK_SUCCESS)
		return {};

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

	// Can any allocation be a descriptor heap on this device, or only one that asked to be?
	//
	// Asked rather than assumed, because the answer is a driver's choice and it decides whether
	// MEMORY_DESCRIPTOR_HEAP means anything. Adding the usage bit can move a buffer to a different
	// set of memory types -- on Intel/Mesa it does, from {0,1,2} to a disjoint {4,5,6} -- and
	// forcing every allocation into those would be paying a real cost for a bit almost nothing
	// uses. Where the bit changes neither the alignment nor the permitted types it is free, and
	// then every allocation may as well carry it, which spares the caller having to know in advance
	// which of its memory will end up holding descriptors.
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

	// Int64 atomics are what SIGNAL_ATOMIC_MAX and SIGNAL_ATOMIC_OR are made of. Only physical
	// device support can be checked here -- whether the VkDevice actually enabled the feature is
	// not queryable -- so this trusts that a device was built with
	// gpuEnableRequiredVulkan12FeaturesEXT, the same way the rest of the backend trusts it for
	// descriptorHeap and bufferDeviceAddress.
	{
		VkPhysicalDeviceVulkan12Features features12 { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
		VkPhysicalDeviceFeatures2 features { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &features12 };
		vkGetPhysicalDeviceFeatures2(gpu, &features);
		out->capabilities.split_barrier_signals = features12.shaderBufferInt64Atomics;
	}

	// Whether a present can carry a fence. Like the feature check above this reads the physical
	// device, so it assumes the VkDevice was built with gpuOptionalVulkanDeviceExtensionsEXT
	// enabled (which gpuSetupDefaultVulkanEXT does).
	{
		VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR maintenance {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR
		};
		VkPhysicalDeviceFeatures2 features { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &maintenance };
		vkGetPhysicalDeviceFeatures2(gpu, &features);
		out->present_fences_supported = maintenance.swapchainMaintenance1;
	}

	// Events give Vulkan a real split barrier, so a paired wait need not stall the work between the
	// halves. Mesh shading is reported off regardless of VK_EXT_mesh_shader, because there is still
	// no way to create a pipeline that could be drawn with it.
	out->capabilities.split_barriers = true;
	out->capabilities.gpu_writable_texture_heap = true;
	out->capabilities.mesh_shaders = false;

	out->command_submission_timeline_semaphore = gpuCreateSemaphoreImpl(out, 0)->semaphore;

	// A GpuTextureDescriptor is 64 bytes because that is the widest image descriptor in
	// circulation (Intel's), so a driver's own is either that or a divisor of it -- 32 on AMD and
	// Nvidia. The shader multiplies its heap index by this to land on the right slot, and whatever
	// the driver does not fill of each slot is simply left alone.
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

GpuCapabilities gpuGetCapabilitiesEXT(const GpuQueue* queue) {
	return queue->capabilities;
}

void gpuSetDiagnosticCallbackEXT(GpuQueue* queue, GpuDiagnosticCallbackEXT callback, void* userdata) {
	queue->diagnostic_callback = callback;
	queue->diagnostic_userdata = userdata;
	// A new destination has not been told any of this yet, so let it hear each cause once too
	queue->reported_diagnostics.clear();
}

namespace GPU::detail {
	// Report a call that could not be honored, or could only be honored conservatively.
	//
	// Once per distinct message per queue: every caller of this sits in a function a program calls
	// per draw or per dispatch, so reporting every occurrence would push the first one -- the only
	// one that says anything new -- out of any log worth reading.
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

void gpuFreeQueue(GpuQueue* queue) {
	// Everything below destroys objects the GPU may still be reading. Nothing else is going to be
	// submitted, so waiting here once costs nothing and makes every destruction that follows legal.
	vkDeviceWaitIdle(queue->device);

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

	// Whatever the program did not free itself. The allocator asserts on destruction if anything is
	// still live in one of its blocks, and freeing a queue is the point past which nothing could be
	// freed anyway, so this is the last chance to give the memory back rather than trip that assert.
	//
	// Images first: one is bound into the memory of the allocation it was created against, so
	// destroying that allocation while the image still exists would leave the image dangling.
	for(auto [address, image]: queue->gpu2image)
		vkDestroyImage(queue->device, image, queue->callbacks);
	queue->gpu2image.clear();

	for(auto& [address, heap]: queue->descriptor_heaps) {
		auto [buffer, allocation, _size, _heap_address] = heap;
		vmaDestroyBuffer(queue->gpu_allocator, buffer, allocation);
	}
	queue->descriptor_heaps.clear();

	for(auto& [address, allocation]: queue->allocations)
		vmaDestroyBuffer(queue->gpu_allocator, allocation.buffer, allocation.allocation);
	queue->allocations.clear();
	queue->host2gpu.clear();
	queue->gpu2host.clear();

	if(queue->gpu_allocator)
		vmaDestroyAllocator(queue->gpu_allocator);

	auto allocator = queue->cpu_allocator;
	queue->~GpuQueue(); // Get all of the caches to free their memory
	allocator(queue, 0);
}




namespace GPU::detail {
	static VkDeviceAddress ensure_sampler_set(GpuQueue* queue, std::span<const GpuSamplerDesc> requested);
	static void bind_sampler_set(GpuCommandBuffer* cmd, VkDeviceAddress sampler_map);
	void apply_depth_stencil_state(GpuCommandBuffer* cmd, const GpuDepthStencilDesc& descriptor);
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

	// Every command buffer starts out with the default sampler set bound. Without it sampler_map
	// stays null and the shader prologue's gpuGetSamplerIndex dereferences a null buffer reference
	// the moment a shader asks for a sampler nobody enabled; the WebGPU backend hands every command
	// buffer the same default set for the same reason.
	GPU::detail::bind_sampler_set(out, GPU::detail::ensure_sampler_set(queue, {}));

	// And with depth and stencil state, which the pipelines declare dynamic and a draw is therefore
	// not allowed to proceed without. The default description is no test, no write and no stencil,
	// which is what the static state said before any of this was dynamic.
	GPU::detail::apply_depth_stencil_state(out, GpuDepthStencilDesc{});

	return out;
}

void gpuFreeCommandBuffer(GpuCommandBuffer* cmd) {
	auto queue = cmd->queue; // Read before the destructor runs, since the free below still needs it
	vkFreeCommandBuffers(queue->device, queue->command_pool, 1, &cmd->command_buffer);
	cmd->~GpuCommandBuffer();
	queue->cpu_allocator(cmd, 0);
}

uint64_t gpuSubmitNoFreeEXT(GpuQueue* queue, GpuCommandBufferSpan commandBuffers, GpuSemaphore* semaphore /* = nullptr */, uint64_t signalValue /* = 0 */) {
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

		// Events this recording set become reusable once this submission has run, and not before:
		// resetting one earlier could release a wait still queued behind it
		for(auto event: cmd->events_used)
			queue->events_in_flight.emplace_back(event, queue->command_submission_timeline_semaphore_next_value);
		cmd->events_used.clear();

		// A signal whose wait never arrived leaves an event set and nothing looking at it. That is
		// harmless -- the counter was still written, and the event is recycled with the rest -- but
		// it means the pair the caller wrote was never a pair.
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
	VK_CHECK(vkQueueSubmit2(queue->queue, 1, &info, nullptr), queue->command_submission_timeline_semaphore_next_value);
	return queue->command_submission_timeline_semaphore_next_value++;
}

uint64_t gpuSubmit(GpuQueue* queue, GpuCommandBufferSpan commandBuffers, GpuSemaphore* semaphore /* = nullptr */, uint64_t signalValue /* = 0 */) {
	auto submission_index = gpuSubmitNoFreeEXT(queue, commandBuffers, semaphore, signalValue);

	for(auto& cmd: commandBuffers) {
		queue->command_buffers_pending_free.emplace_back(cmd->command_buffer, submission_index);

		cmd->~GpuCommandBuffer(); // Free the buffer
		queue->cpu_allocator(cmd, 0);
	}
	return submission_index;
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
	// Index and indirect usage go on every allocation rather than being asked for, because memory
	// here has no declared purpose: a gpu* is just an address, and which of them ends up holding
	// indices or draw arguments is the program's business, discovered when it records a draw
	// rather than when it allocates. Both bits are free to add -- neither changes the alignment or
	// the set of memory types the buffer may live in, on any driver checked -- which is what makes
	// blanket usage affordable instead of a flag the caller has to predict.
	//
	// (Indirect was previously missing entirely, so gpuDispatchIndirect and the indirect draws were
	// reading arguments out of buffers that never declared they could be.)
	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		| VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

	// Descriptor heap usage is not free in the same way: on this driver it moves the buffer to a
	// disjoint set of memory types (see the probe in gpuCreateQueue), so it goes on only what asked
	// for it -- unless the device turns out not to care, in which case every allocation gets it and
	// MEMORY_DESCRIPTOR_HEAP stops meaning anything in particular.
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
	// A heap has to start where the hardware can address one from, which is a stricter alignment
	// than a caller would think to ask for
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
	if(queue->host2gpu.contains(ptr)) {
		gpuFree(queue, (gpu*)queue->host2gpu[ptr]);
	}
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

	if(queue->gpu2image.contains(gpu_ptr)) {
		auto image = queue->gpu2image[gpu_ptr];
		vkDestroyImage(queue->device, image, queue->callbacks);
		queue->gpu2image.erase(gpu_ptr);
	}

	// Erased as well as destroyed: device addresses get recycled by the allocator, so an entry left
	// behind here would be found again by the next allocation to land on this address and hand out
	// a VkBuffer that no longer exists.
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
			// Optimal, not linear. Nothing reads these images as linear memory -- a shader reaches
			// them through a heap descriptor and the CPU reaches them through gpuCopyToTexture,
			// whose vkCmdCopyBufferToImage swizzles on the driver's side whatever the tiling is --
			// so linear bought nothing and cost a great deal: it is the reason a depth format could
			// not be created at all (no driver offers a linear depth/stencil attachment), and it
			// rules out exactly the compression and tile swizzling MEMORY_TEXTURE advertises.
			.tiling = VK_IMAGE_TILING_OPTIMAL,
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

	auto backing = queue->allocations.find((VkDeviceAddress)memory);
	if(backing == queue->allocations.end()) {
		errno = VK_ERROR_INITIALIZATION_FAILED;
		return nullptr;
	}

	auto out = (GpuTexture*)queue->cpu_allocator(nullptr, sizeof(GpuTexture));
	*out = {.descriptor = desc};

	auto info = GPU::detail::descriptor2vulkan(desc);
	VK_CHECK(vkCreateImage(queue->device, &info, queue->callbacks, &out->image), nullptr);
	// Bound through VMA rather than with vkBindImageMemory directly: an allocation is normally a
	// suballocation of a larger VkDeviceMemory block, so binding at offset 0 of that block would
	// alias whatever sits at its start instead of the memory that was actually allocated here.
	VK_CHECK(vmaBindImageMemory(queue->gpu_allocator, backing->second.allocation, out->image), nullptr);

	queue->gpu2image[(VkDeviceAddress)memory] = out->image;
	return out;
}

static GpuTextureDescriptor gpuTextureViewDescriptorImpl(GpuQueue* queue, const GpuTexture* texture, const GpuViewDesc& desc, bool read_only) {
	// GpuTextureDescriptor is a fixed 512 bits, which is the widest image descriptor in
	// circulation (Intel's); AMD's and Nvidia's are 256. vkWriteResourceDescriptorsEXT writes the
	// driver's own size regardless of the range it is handed, so a driver wider still than this
	// would overrun `out` -- and the heap the caller builds out of these would have a smaller
	// stride than the shader indexes it by, so widening the write alone would not save it. The
	// width is part of this API's ABI, so gpuCreateQueue refuses such a device outright rather
	// than letting it get here.
	// The whole 64 byte slot is offered even though the driver only fills image_size of it: the
	// write is rejected outright if the range is narrower than the driver's descriptor, and it
	// writes that many bytes whatever the range says, so the range has to be the larger of the two.
	// The slack at the end of a slot on a narrow driver is never read by anything.
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
			.aspectMask = static_cast<VkImageAspectFlags>(gpuFormatIsDepthEXT(format)
				? VK_IMAGE_ASPECT_DEPTH_BIT | (gpuFormatIsStencilEXT(format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)
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

void gpuSyncMemoryImmediateEXT(GpuQueue* queue, gpu* mem) {
	// Do nothing!
}




struct ComputePipelinePushConstants {
	gpu* data;
	gpu* sampler_map;
};

GpuPipeline* gpuCreateComputePipeline(GpuQueue* queue, GpuByteSpan computeIR) {
	auto spirv = compile_shader(queue, computeIR, GPU::shaders::SHADER_STAGE::COMPUTE);
	if(!spirv) return nullptr;

	auto out = new(queue->cpu_allocator(nullptr, sizeof(GpuPipeline))) GpuPipeline{};
	out->color_target_count = {}; // Null indicating compute pipeline

	VkShaderModule compute_module;
	{
		VkShaderModuleCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = spirv->size(),
			.pCode = (const uint32_t*)spirv->data(),
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
	pipeline->~GpuPipeline(); // Its vectors own memory of their own
	queue->cpu_allocator(pipeline, 0);
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

namespace GPU::detail {
	// Defined further down, next to the rest of the layout tracking. Declared here because the
	// copies below are the first thing in the file that has to move an image into GENERAL.
	inline void transition_texture(VkCommandBuffer cmd, const GpuTexture* texture, VkImageLayout new_layout,
		VkAccessFlags source_access_mask, VkAccessFlags destination_access_mask,
		VkPipelineStageFlags source_stage, VkPipelineStageFlags destination_stage,
		uint32_t mip, uint32_t slice, bool discard);
}

void gpuCopyToTexture(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, GpuTexture* texture) {
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_);
	assert(dest.buffer && src.buffer && "Neither end of a copy may be an address outside every allocation");

	assert(cmd->queue->gpu2image.contains(dest.base) && cmd->queue->gpu2image[dest.base] == texture->image);

	assert(dest.offset == 0); // TODO: Can we relax this restriction?

	// An image is created without a layout and every other path in this backend expects to find it
	// in GENERAL -- which is also the layout its heap descriptors are written against -- so the
	// copy has to put it there first. Without this the image is still UNDEFINED when it is
	// sampled, which is undefined contents rather than a diagnosable error. Discarding, because
	// the copy is about to overwrite what is being transitioned.
	GPU::detail::transition_texture(cmd->command_buffer, texture, VK_IMAGE_LAYOUT_GENERAL,
		0, VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 0, true);

	VkBufferImageCopy copy {
		.bufferOffset = src.offset,
		// Vulkan counts this in texels rather than bytes. The staging data is laid out tightly,
		// one image row per buffer row, which is the layout the WebGPU backend copies through too.
		.bufferRowLength = texture->descriptor.dimensions.x,
		.bufferImageHeight = texture->descriptor.dimensions.y,
		.imageSubresource = {
			.aspectMask = static_cast<VkImageAspectFlags>(gpuFormatIsDepthEXT(texture->descriptor.format)
				? VK_IMAGE_ASPECT_DEPTH_BIT | (gpuFormatIsStencilEXT(texture->descriptor.format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)
				: VK_IMAGE_ASPECT_COLOR_BIT),
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
		.imageOffset = {0, 0, 0},
		.imageExtent = {texture->descriptor.dimensions.x, texture->descriptor.dimensions.y, texture->descriptor.dimensions.z}
	};
	vkCmdCopyBufferToImage(cmd->command_buffer, src.buffer, texture->image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
}

void gpuCopyFromTexture(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, const GpuTexture* texture) {
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_);
	assert(dest.buffer && src.buffer && "Neither end of a copy may be an address outside every allocation");

	assert(cmd->queue->gpu2image.contains(src.base) && cmd->queue->gpu2image[src.base] == texture->image);

	assert(src.offset == 0); // TODO: Can we relax this restriction?

	// Into GENERAL, the layout this backend keeps textures in, in case the last thing to touch it
	// left it somewhere else. Not discarding: the contents are the entire point of a readback.
	GPU::detail::transition_texture(cmd->command_buffer, texture, VK_IMAGE_LAYOUT_GENERAL,
		VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 0, false);

	VkBufferImageCopy copy {
		// The texture is the source here, so the buffer half of the copy is the destination
		.bufferOffset = dest.offset,
		// In texels, not bytes; see gpuCopyToTexture
		.bufferRowLength = texture->descriptor.dimensions.x,
		.bufferImageHeight = texture->descriptor.dimensions.y,
		.imageSubresource = {
			.aspectMask = static_cast<VkImageAspectFlags>(gpuFormatIsDepthEXT(texture->descriptor.format)
				? VK_IMAGE_ASPECT_DEPTH_BIT | (gpuFormatIsStencilEXT(texture->descriptor.format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)
				: VK_IMAGE_ASPECT_COLOR_BIT),
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
		.imageOffset = {0, 0, 0},
		.imageExtent = {texture->descriptor.dimensions.x, texture->descriptor.dimensions.y, texture->descriptor.dimensions.z}
	};
	vkCmdCopyImageToBuffer(cmd->command_buffer, texture->image, VK_IMAGE_LAYOUT_GENERAL, dest.buffer, 1, &copy);
}

void gpuSetActiveTextureHeapPtr(GpuCommandBuffer* cmd, gpu* texture_heap) {
	const GpuQueue::Allocation* holding = nullptr;
	auto heap = GPU::detail::closest_buffer(cmd->queue, texture_heap, &holding);
	assert(heap.buffer && "The texture heap pointer doesn't lie inside any allocation");
	if(!heap.buffer) return;

	// The caller's own memory is bound when it was allocated able to be bound (MEMORY_DESCRIPTOR_HEAP,
	// or any allocation at all on a device where the usage costs nothing) and it is long enough to
	// cover the range the hardware reserves at the front of a heap. Nothing is copied, and a
	// descriptor a shader writes into it is seen by whatever samples the heap after the next
	// HAZARD_DESCRIPTORS barrier -- which is what GpuTextureDescriptor says the heap does.
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

	// Otherwise the heap sits in memory that cannot be bound as one, so what gets bound is a
	// snapshot copied into memory that can. Two things follow, and both are the reason the
	// allocation above is worth asking for: the copy happens on every call, and a descriptor a
	// shader writes into the caller's memory is invisible until the heap is set again.
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

	// The copy has to be visible to descriptor reads before anything samples through the heap.
	// Without this the bind below, and every draw after it, races the transfer that filled the
	// heap they read from.
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
			// The heap read bits come from VK_EXT_descriptor_heap, which is what this backend binds
			// heaps with. VK_ACCESS_2_DESCRIPTOR_BUFFER_READ_BIT_EXT belongs to
			// VK_EXT_descriptor_buffer, a different extension that is not enabled here, so naming it
			// made this barrier invalid -- which every use of HAZARD_DESCRIPTORS would have hit.
			dst_access |= VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT | VK_ACCESS_2_SAMPLER_HEAP_READ_BIT_EXT
				| VK_ACCESS_2_UNIFORM_READ_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT_KHR;
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

	// Drop the access bits a stage mask cannot carry.
	//
	// Vulkan ties most access flags to particular stages -- a shader read is only meaningful where
	// a shader runs, an indirect command read only at the draw-indirect stage -- and names a
	// barrier invalid, rather than merely pessimistic, if it pairs one with a stage that cannot
	// perform it. The hazard flags above describe what kind of cache to invalidate without knowing
	// which stage will be asked to consume it, so the two have to be reconciled here. What survives
	// for a stage that carries none of these is VK_ACCESS_2_MEMORY_READ/WRITE, which every stage
	// accepts and which is a superset of the rest, so filtering costs correctness nothing.
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
	// The hazard flags don't know which stages they will be paired with, and Vulkan rejects a
	// barrier that names an access a stage cannot perform
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
	// The core 1.3 entry point rather than the KHR alias: the device is created against 1.4, where
	// synchronization2 is core, and volk only resolves the suffixed alias when the extension that
	// introduced it was explicitly enabled -- which it isn't, so the alias is left null.
	vkCmdPipelineBarrier2(cmd->command_buffer, &dependency);
}

namespace GPU::detail {
	std::vector<uint32_t> compile_glsl(EShLanguage stage, std::string_view source); // With the blit shaders, below

	// gpuSignalAfter's atomic, for the two SIGNAL values that are read-modify-writes rather than
	// plain stores. One thread, because the counter is one 64 bit word; the address arrives as a
	// buffer reference in a push constant, so this needs no descriptor of any kind.
	constexpr static std::string_view SIGNAL_SHADER = R"(
#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require

layout(buffer_reference, std430, buffer_reference_align = 8) buffer Counter {
	uint64_t value;
};

layout(push_constant) uniform Push {
	Counter counter;
	uint64_t value;
	uint use_max; // 1 for SIGNAL_ATOMIC_MAX, 0 for SIGNAL_ATOMIC_OR
} push;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
	if(push.use_max != 0) atomicMax(push.counter.value, push.value);
	else atomicOr(push.counter.value, push.value);
}
)";

	struct SignalPushConstants {
		VkDeviceAddress counter;
		uint64_t value;
		uint32_t use_max;
	};

	// Built on the first signal that needs it rather than at queue creation, since a program that
	// never splits a barrier should not pay for a pipeline it never dispatches. One pipeline covers
	// both operations: which one runs is a push constant, so there is nothing to specialize.
	inline VkPipeline signal_pipeline(GpuQueue* queue) {
		if(queue->signal_pipeline) return queue->signal_pipeline;

		{
			auto spirv = compile_glsl(EShLangCompute, SIGNAL_SHADER);
			if(spirv.empty()) return VK_NULL_HANDLE;

			VkShaderModuleCreateInfo module_info {
				.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
				.codeSize = spirv.size() * sizeof(uint32_t),
				.pCode = spirv.data(),
			};
			VK_CHECK(vkCreateShaderModule(queue->device, &module_info, queue->callbacks, &queue->signal_shader_module), VK_NULL_HANDLE);

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

	// Events are recycled only once the submission that used one has finished: an event has to be
	// unsignalled before it can be set again, and resetting one that a command buffer still in
	// flight might be waiting on would let that wait through early.
	inline VkEvent acquire_event(GpuQueue* queue) {
		if(!queue->events_in_flight.empty()) {
			uint64_t finished;
			vkGetSemaphoreCounterValue(queue->device, queue->command_submission_timeline_semaphore, &finished);

			for(size_t i = queue->events_in_flight.size(); i--; ) {
				auto [event, submit] = queue->events_in_flight[i];
				if(submit > finished) continue;

				vkResetEvent(queue->device, event);
				queue->events_free.push_back(event);
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

	// Every way of writing the counter is a command a render pass forbids -- a transfer for
	// SIGNAL_ATOMIC_SET, a dispatch for the other two -- and so is the event that would pair this
	// with a wait. So a signal recorded inside a pass can do nothing at all, and says so rather
	// than recording something illegal.
	if(cmd->state != GpuCommandBuffer::Recording) {
		GPU::detail::report(cmd, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuSignalAfter: writing the counter needs a transfer or a dispatch, and neither can be recorded inside a render pass, so nothing is signalled; split the pass or signal around it");
		return;
	}

	// The counter is ordinary memory in an ordinary allocation, so writing it needs to be ordered
	// after the work whose completion it is reporting
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

		// Put back whatever the caller had bound, the way a blit does: this borrowed the compute
		// binding point, and the next draw or dispatch would otherwise run this shader
		if(cmd->bound_pipeline)
			vkCmdBindPipeline(cmd->command_buffer,
				cmd->bound_pipeline->color_target_count.has_value() ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE,
				cmd->bound_pipeline->pipeline);
	}

	// The second half of a split barrier: an event set here, for a later gpuWaitBefore to wait on
	// without stalling what sits between them.
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
			// The consumer is unknown until the wait is recorded, and Vulkan wants both halves to
			// name the same dependency, so this end stays open
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

void gpuWaitBefore(GpuCommandBuffer* cmd, STAGE after, gpu* ptr, uint64_t value, OP op, HAZARD_FLAGS hazards /* = (HAZARD_FLAGS)0 */, uint64_t mask /* = ~uint64_t(0) */) {
	// A counter comparison is not something any Vulkan command does, so what can be honored is the
	// ordering a pair expresses, not the condition itself. That is exactly the pairing case: a
	// signal recorded earlier in this command buffer on this counter and this value, with a
	// comparison that pair satisfies by construction. Everything else -- a mask over the counter, a
	// comparison that can fail after the signal, a counter some shader wrote instead, a signal in
	// another command buffer -- has no expressible form, and becomes the barrier below.
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

			// The event's dependency had to name the one its signal named, which could not know
			// these hazards yet, so the cache invalidation they ask for is a second barrier. It
			// names the producer's stage rather than every stage: going through gpuBarrier with
			// STAGE_ALL here would stall on everything recorded between the two halves and undo
			// the split the event just bought.
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

	GPU::detail::report(cmd, GPU_DIAGNOSTIC_EMULATED,
		!comparison_is_satisfied_by_the_pair
			? "gpuWaitBefore: a masked or non-monotonic comparison has no equivalent Vulkan command, so this waits on every stage instead of splitting the barrier"
			: cmd->state != GpuCommandBuffer::Recording
				? "gpuWaitBefore: Vulkan has no event commands inside a render pass, so a wait recorded in one waits on every stage instead of splitting the barrier"
				: "gpuWaitBefore: no gpuSignalAfter earlier in this command buffer matches this counter and value, so this waits on every stage instead of splitting the barrier");
	gpuBarrier(cmd, STAGE_ALL, after, hazards);
}

namespace GPU::detail {
	VkColorComponentFlags mask2vulkan(uint8_t mask); // Defined with the other blend conversions, below
}

void gpuSetPipeline(GpuCommandBuffer* cmd, const GpuPipeline* pipeline) {
	vkCmdBindPipeline(cmd->command_buffer, pipeline->color_target_count.has_value() ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
	cmd->bound_pipeline = pipeline;

	// Blending is dynamic state, so a pipeline's own blend description only takes effect if
	// something applies it. Doing that here means a program that never touches gpuSetBlendState
	// gets what its pipeline asked for, and one that does gets the override for as long as it
	// keeps setting it -- a draw is otherwise not allowed to proceed at all with these unset.
	if(cmd->bound_blend_state) {
		// An explicit state outranks the pipeline's own description, and has to be re-applied
		// because the masks it produces are relative to the pipeline now bound
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
	// Builds (or finds) the sampler heap for a list of enabled samplers and hands back the address
	// its lookup map lives at. Slot 0 is always the default sampler, so a packed description that
	// nobody enabled still resolves to something valid, matching the WebGPU backend.
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

		// The map is indexed by pack(), so it needs one word per value in [0, max_packed()]
		constexpr static auto sampler_lookup_size = GpuSamplerDesc::max_packed() + 1;
		uint32_t* sampler_mapping_cpu = gpuMalloc<uint32_t>(queue, sampler_lookup_size);
		if(!sampler_mapping_cpu) return 0;
		std::memset(sampler_mapping_cpu, 0, sampler_lookup_size * sizeof(uint32_t));
		for(size_t i = 0; i < enabled_samplers.size(); ++i)
			sampler_mapping_cpu[enabled_samplers[i].pack()] = i;
		// Kept in a local until the heap below is built: gpuMalloc can rehash sampler_cache, which
		// would leave a reference into it dangling
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
		// One range per sampler, not one range covering the lot. pDescriptors is an array parallel
		// to pSamplers, so handing over a single range while claiming samplerCount of them leaves
		// the driver reading whatever happens to follow it in memory as the destination for every
		// sampler past the first -- which wrote the default sampler into slot 0 correctly and left
		// every other slot untouched, so that any shader asking gpuGetSamplerIndex for a sampler it
		// had actually enabled sampled black.
		std::vector<VkHostAddressRangeEXT> sampler_ranges; sampler_ranges.reserve(sampler_infos.size());
		for(size_t i = 0; i < sampler_infos.size(); ++i)
			sampler_ranges.emplace_back(VkHostAddressRangeEXT{
				.address = (char*)tmp + i * queue->sampler_size,
				.size = queue->sampler_size,
			});
		vkWriteSamplerDescriptorsEXT(queue->device, sampler_infos.size(), sampler_infos.data(), sampler_ranges.data());

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

		// A private one shot submission rather than gpuStartCommandRecording: the default set is
		// installed from inside gpuStartCommandRecording, which would otherwise recurse
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
		// Built once per distinct sampler set, so stalling here is affordable, and it is what makes
		// the staging allocation safe to hand back on the next line
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
}

void gpuSetEnabledSamplersEXT(GpuCommandBuffer* cmd, GpuSamplerDescSpan enabled_samplers) {
	GPU::detail::bind_sampler_set(cmd, GPU::detail::ensure_sampler_set(cmd->queue, enabled_samplers));
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



GpuPipeline* gpuCreateGraphicsPipeline(GpuQueue* queue, GpuByteSpan vertexIR, GpuByteSpan fragmentIR, const GpuRasterDesc& desc) {
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

	// Both stages are compiled before anything is created, so a shader that does not build
	// leaves no half made pipeline behind. Handing the same source in as both is normal: the
	// entry points are told apart by stage, not by which argument they arrived in.
	auto vertex_spirv = compile_shader(queue, vertexIR, GPU::shaders::SHADER_STAGE::VERTEX);
	if(!vertex_spirv) return nullptr;
	auto fragment_spirv = compile_shader(queue, fragmentIR, GPU::shaders::SHADER_STAGE::FRAGMENT);
	if(!fragment_spirv) return nullptr;

	// Constructed, not just allocated: a GpuPipeline owns vectors now, and assigning into raw
	// storage would be writing through whatever those bytes happened to contain
	auto out = new(queue->cpu_allocator(nullptr, sizeof(GpuPipeline))) GpuPipeline{};
	out->color_target_count = desc.colorTargets.size();

	std::array<VkShaderModule, 2> shader_modules;
	{
		VkShaderModuleCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = vertex_spirv->size(),
			.pCode = (const uint32_t*)vertex_spirv->data(),
		};
		VK_CHECK(vkCreateShaderModule(queue->device, &info, queue->callbacks, &shader_modules[0]), nullptr);
	}{
		VkShaderModuleCreateInfo info {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = fragment_spirv->size(),
			.pCode = (const uint32_t*)fragment_spirv->data(),
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

			// Kept for gpuSetPipeline to apply, since the copy in the create info is ignored once
			// these are dynamic
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

	// Every one of these has to be listed for the matching vkCmdSet call to have any effect: a
	// state the pipeline does not declare dynamic keeps whatever the create info said, and the
	// create info says nothing -- VkPipelineDepthStencilStateCreateInfo above is zero initialized,
	// which is depth test and depth write disabled. Only the first two used to be counted, so
	// gpuSetDepthStencilState and gpuSetBlendState were writing state that nothing read; a depth
	// buffer stayed at whatever it was cleared to no matter what was drawn into it.
	//
	// The last three come from VK_EXT_extended_dynamic_state3, which the backend requires.
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
namespace GPU::detail {
	void apply_depth_stencil_state(GpuCommandBuffer* cmd, const GpuDepthStencilDesc& descriptor);
}

void gpuSetDepthStencilState(GpuCommandBuffer* cmd, const GpuDepthStencilState* state) {
	GPU::detail::apply_depth_stencil_state(cmd, state->descriptor);
}

// Every one of these is dynamic state the pipeline declares, which means a draw is invalid until
// each has been set. A command buffer that never calls gpuSetDepthStencilState still has to be able
// to draw, so recording one starts by applying a default-constructed description -- no depth test,
// no depth write, no stencil -- which is what the pipeline's (now ignored) static state used to say.
// Setting one later overrides it, whatever order that happens in relative to binding a pipeline,
// since nothing else touches this state.
void GPU::detail::apply_depth_stencil_state(GpuCommandBuffer* cmd, const GpuDepthStencilDesc& descriptor) {
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

	// depthTestEnable gates both the compare op AND whether writes happen
	// in classic GL/D3D semantics; Vulkan separates test-enable from
	// write-enable, so: enable the test if either read or write was
	// requested (write without a passing test wouldn't write anything
	// meaningful), and gate the actual write bit off DEPTH_WRITE.
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

		// Front/back reference values differ in your struct
		// (Stencil::reference is per-face) but vkCmdSetStencilReference
		// takes a face mask too, so two calls if front != back.
		if (descriptor.stencilFront.reference == descriptor.stencilBack.reference) {
			vkCmdSetStencilReference(cmd->command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, descriptor.stencilFront.reference);
		} else {
			vkCmdSetStencilReference(cmd->command_buffer, VK_STENCIL_FACE_FRONT_BIT, descriptor.stencilFront.reference);
			vkCmdSetStencilReference(cmd->command_buffer, VK_STENCIL_FACE_BACK_BIT, descriptor.stencilBack.reference);
		}
	}

	vkCmdSetDepthBias(cmd->command_buffer, descriptor.depthBias, descriptor.depthBiasClamp, descriptor.depthBiasSlopeFactor);
}

// TODO: Untested!
void gpuSetBlendState(GpuCommandBuffer* cmd, const GpuBlendState* state) {
	assert(cmd->bound_pipeline);
	cmd->bound_blend_state = state;

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

	// The mask each target was created with, narrowed by the one this state carries. Not optional
	// now that it is dynamic state the pipeline declares: a draw with it unset is invalid, and
	// leaving it to whatever was set last would silently take its write mask from another pipeline.
	std::vector<VkColorComponentFlags> masks; masks.reserve(count);
	for(uint32_t i = 0; i < count; ++i)
		masks.push_back(GPU::detail::mask2vulkan(
			cmd->bound_pipeline->color_write_masks[i] & state->descriptor.colorWriteMask));
	vkCmdSetColorWriteMaskEXT(cmd->command_buffer, 0, count, masks.data());
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
	// The aspects a format's views and barriers have to name. A depth/stencil image described with
	// VK_IMAGE_ASPECT_COLOR_BIT is invalid, which is what the barriers here used to do.
	inline VkImageAspectFlags texture_aspect(FORMAT format) {
		if(!gpuFormatIsDepthStencilEXT(format)) return VK_IMAGE_ASPECT_COLOR_BIT;
		return (gpuFormatIsDepthEXT(format) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0)
			| (gpuFormatIsStencilEXT(format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
	}

	inline uvec2 mip_extent(const GpuTexture* texture, uint32_t mip) {
		return {
			std::max(texture->descriptor.dimensions.x >> mip, 1u),
			std::max(texture->descriptor.dimensions.y >> mip, 1u)
		};
	}

	// Destroys every view whose submission has finished. Each view names one subresource, so unlike
	// a descriptor set there is nothing about one worth recycling.
	inline void reclaim_views(GpuQueue* queue) {
		if(queue->views_in_flight.empty()) return;

		uint64_t current_finished_submission;
		vkGetSemaphoreCounterValue(queue->device, queue->command_submission_timeline_semaphore, &current_finished_submission);

		for(size_t i = queue->views_in_flight.size(); i--; ) {
			auto [view, submit] = queue->views_in_flight[i];
			if(submit > current_finished_submission) continue;

			vkDestroyImageView(queue->device, view, queue->callbacks);
			queue->views_in_flight.erase(queue->views_in_flight.begin() + i);
		}
	}

	// The single mip and single layer an attachment actually renders into. Built per pass and
	// retired with the submission that used it rather than cached on the texture, because the same
	// texture can be attached through a different subresource every pass — caching one view of the
	// whole image is what used to make mipLevel and slice silently resolve to 0.
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

	// Moves a texture into `new_layout` from wherever this backend last left it, keeping the record
	// on the texture current so the next transition knows where it starts from. `discard` gives up
	// the previous contents, which is what an attachment whose load op isn't LOAD_OP_LOAD asks for
	// and the only thing that is valid for an image nothing has written yet.
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

	// An acquired surface image comes with a semaphore the first use of it has to wait on. It is
	// handed over exactly once: waiting on the same binary semaphore twice never completes.
	// Block until the queue's submission timeline reaches `value`, which is how the surface waits
	// for the frame that last used one of its semaphores to be done with it. Zero means the thing
	// being waited for never happened, so there is nothing to wait for.
	inline void wait_for_submission(GpuQueue* queue, uint64_t value) {
		if(value == 0) return;

		uint64_t reached = 0;
		vkGetSemaphoreCounterValue(queue->device, queue->command_submission_timeline_semaphore, &reached);
		if(reached >= value) return;

		VkSemaphoreWaitInfo info {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
			.semaphoreCount = 1,
			.pSemaphores = &queue->command_submission_timeline_semaphore,
			.pValues = &value,
		};
		vkWaitSemaphores(queue->device, &info, UINT64_MAX);
	}

	inline void consume_available_semaphore(GpuCommandBuffer* cmd, const GpuTexture* texture) {
		if(!texture->available_semaphore) return;

		cmd->wait_semaphores.push_back(texture->available_semaphore);
		const_cast<GpuTexture*>(texture)->available_semaphore = VK_NULL_HANDLE;
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

	// Taken from whichever attachment comes first rather than from the color attachments: a depth
	// only pass (a shadow map, say) is allowed to have no color attachment at all
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
			.resolveMode = resolve_view ? VK_RESOLVE_MODE_SAMPLE_ZERO_BIT : VK_RESOLVE_MODE_NONE,
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

	// Depth and stencil stay separate attachments here, as they are in Vulkan, but when one texture
	// carries both aspects only the depth half transitions it — a second transition would either
	// discard what the first just preserved or barrier the image against itself
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
			// The same subresource the depth half named, when they share a texture: Vulkan wants
			// both halves of a combined format pointing at one view
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

void gpuEndRenderPass(GpuCommandBuffer* cmd, GpuOptionalRenderPassDesc desc /*= {}*/) {
	vkCmdEndRendering(cmd->command_buffer);

	if(desc) for(auto& color: desc->colorAttachments)
		GPU::detail::transition_texture(cmd->command_buffer, color.texture, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			color.mipLevel, color.slice, false
		);

	cmd->state = GpuCommandBuffer::Recording;
}




namespace GPU::detail {

	constexpr static std::string_view BLIT_VERTEX_SHADER = R"(
#version 460

layout(location = 0) out vec2 out_uv;

void main() {
	// An oversized triangle covering the whole viewport, its uvs run 0..1 across the covered area
	vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	// Vulkan's clip space puts -Y at the top of the viewport, which is also where v = 0 sits
	gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
	out_uv = uv;
}
)";

	constexpr static std::string_view BLIT_FRAGMENT_SHADER = R"(
#version 460

layout(set = 0, binding = 0) uniform sampler2D blit_source;

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

void main() {
	// The view covers a single mip, so there is never another level to pick between
	out_color = textureLod(blit_source, in_uv, 0.0);
}
)";

	inline std::vector<uint32_t> compile_glsl(EShLanguage stage, std::string_view source) {
		static const bool initialized = []{ glslang::InitializeProcess(); return true; }();
		(void)initialized;

		auto data = source.data();
		int length = source.size();

		glslang::TShader shader(stage);
		shader.setStringsWithLengths(&data, &length, 1);
		shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 460);
		shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_3);
		shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);

		if(!shader.parse(GetDefaultResources(), 460, false, EShMsgDefault)) {
			assert(false && "An internal shader failed to compile");
			return {};
		}

		glslang::TProgram program;
		program.addShader(&shader);
		if(!program.link(EShMsgDefault)) {
			assert(false && "An internal shader failed to link");
			return {};
		}

		std::vector<uint32_t> spirv;
		glslang::GlslangToSpv(*program.getIntermediate(stage), spirv);
		return spirv;
	}

	// The descriptor set layout, the pipeline layout, the shader modules, and the two samplers are
	// all format independent, so they are built once and then shared by every blit on this queue
	inline bool ensure_blit_layout(GpuQueue* queue) {
		if(queue->blit_pipeline_layout) return true;

		constexpr static std::array stages = {
			std::pair{EShLangVertex, BLIT_VERTEX_SHADER},
			std::pair{EShLangFragment, BLIT_FRAGMENT_SHADER}
		};
		for(size_t i = 0; i < stages.size(); ++i) {
			auto spirv = compile_glsl(stages[i].first, stages[i].second);
			if(spirv.empty()) return false;

			VkShaderModuleCreateInfo info {
				.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
				.codeSize = spirv.size() * sizeof(uint32_t),
				.pCode = spirv.data(),
			};
			VK_CHECK(vkCreateShaderModule(queue->device, &info, queue->callbacks, &queue->blit_shader_modules[i]), false);
		}

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

		// The triangle is generated from gl_VertexIndex, so nothing is fetched and nothing is indexed
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
		// Deliberately no VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT: the blit binds its source
		// the classic way rather than through whatever heap the user happens to have active
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

	// Hands back everything whose submission has finished. Views are shared with render pass
	// attachments and retired by reclaim_views; sets go on the free list to be handed straight out.
	inline void reclaim_blit_transients(GpuQueue* queue) {
		reclaim_views(queue);
		if(queue->blit_descriptor_sets_in_flight.empty()) return;

		uint64_t current_finished_submission;
		vkGetSemaphoreCounterValue(queue->device, queue->command_submission_timeline_semaphore, &current_finished_submission);

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

		// Sets are never freed individually (they are recycled instead), so a pool is only ever
		// grown by adding another one next to it when the newest has been drained
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
		bool linear_filter /* = true */,
		uint32_t destination_mip /* = 0 */, uint32_t destination_slice /* = 0 */,
		uint32_t source_mip /* = 0 */, uint32_t source_slice /* = 0 */) {
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

	// Both sides are addressed as a single mip of a single layer, which is what lets the shader stay
	// a plain sampler2D no matter what the textures it was handed actually are
	constexpr static auto subresource_view = [](GpuQueue* queue, const GpuTexture* texture, uint32_t mip, uint32_t slice) {
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
		vkCreateImageView(queue->device, &info, queue->callbacks, &view);
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
		// Textures in this backend live in VK_IMAGE_LAYOUT_GENERAL, the same layout their heap
		// descriptors are written against
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

	// The shader samples the source in GENERAL, the layout this backend keeps textures in; a source
	// that was just rendered into is still sitting in a render target layout without this
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

	// Bind indices out of the allocation they were written into.
	//
	// Every allocation declares index usage (see gpuMalloc), so there is nothing to prepare: the
	// address names a buffer and an offset into it, and that is what vkCmdBindIndexBuffer takes.
	// This used to keep a device local index buffer per allocation and copy into it before each
	// draw, through a submission of its own -- which cost a copy of the whole index range per
	// draw, and got the ordering wrong as well, since a separate submission made at record time
	// does not see writes recorded earlier in the command buffer being recorded.
	bool bind_index_buffer(GpuCommandBuffer* cmd, gpu* indicesGpu, INDEX_TYPE_EXT index_type) {
		auto indices = closest_buffer(cmd->queue, indicesGpu);
		assert(indices.buffer && "The index pointer doesn't lie inside any allocation");
		if(!indices.buffer) return false;

		// Vulkan reads indices at their natural width from this offset, so a misaligned one is not
		// a slow path but a wrong one: it would shift every index that follows.
		assert(indices.offset % index_stride(index_type) == 0
			&& "The index pointer isn't aligned to the size of one index");

		vkCmdBindIndexBuffer(cmd->command_buffer, indices.buffer, indices.offset, index2vulkan(index_type));
		return true;
	}
}



void gpuDrawIndexedInstanced(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* indices, uint32_t index_count, uint32_t instance_count, INDEX_TYPE_EXT index_type /* = INDEX_TYPE_UINT32 */) {
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

// TODO: Untested!
void gpuDrawIndexedInstancedIndirect(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* indices, gpu* args, INDEX_TYPE_EXT index_type /* = INDEX_TYPE_UINT32 */) {
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
	// Every argument struct between the provided pointer and the end of its allocation is drawn,
	// matching what the WebGPU backend does with the same pointer
	auto count = (arguments.size - arguments.offset) / sizeof(VkDrawIndexedIndirectCommand);
	vkCmdDrawIndexedIndirect(cmd->command_buffer, arguments.buffer, arguments.offset, count, sizeof(VkDrawIndexedIndirectCommand));
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

	// Reported rather than thrown: this is declared inside an extern "C" block, and an exception
	// leaving one is not something a C caller can catch -- it terminates. gpuGetCapabilitiesEXT's
	// mesh_shaders is the form of this a program can test before it records anything.
	if(!vkCmdDrawMeshTasksEXT) {
		GPU::detail::report(cmd, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuDrawMeshlets: this device was not set up with VK_EXT_mesh_shader, so nothing is drawn");
		return;
	}
	vkCmdDrawMeshTasksEXT(cmd->command_buffer, dim.x, dim.y, dim.z);
}

// TODO: Untested!
void gpuDrawMeshletsIndirect(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, gpu* dim) {
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

	if(!vkCmdDrawMeshTasksIndirectEXT) {
		GPU::detail::report(cmd, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuDrawMeshletsIndirect: this device was not set up with VK_EXT_mesh_shader, so nothing is drawn");
		return;
	}

	auto dimensions = GPU::detail::closest_buffer(cmd->queue, dim);
	assert(dimensions.buffer && "The indirect dimensions don't lie inside any allocation");
	auto count = (dimensions.size - dimensions.offset) / sizeof(VkDrawMeshTasksIndirectCommandEXT); // VkDrawMeshTasksIndirectCommandEXT == uvec3
	vkCmdDrawMeshTasksIndirectEXT(cmd->command_buffer, dimensions.buffer, dimensions.offset, count, sizeof(VkDrawMeshTasksIndirectCommandEXT));
}




GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, VkSurfaceKHR surface, const GpuSurfaceDescriptor& desc) {
	auto out = (GpuSurface*)queue->cpu_allocator(nullptr, sizeof(GpuSurface));
	new(out) GpuSurface {
		.surface = surface,
	};

	gpuSurfaceReconfigureEXT(queue, out, desc);
	return out;
}

namespace GPU::detail {
	// Drain a signalled acquire semaphore nothing waited on, by submitting a wait of its own.
	//
	// A binary semaphore cannot be reset from the host, so an acquire whose image was never
	// rendered into leaves one signalled with no way to clear it but to consume it. Returns the
	// submission after which the semaphore is idle again, or zero if there was nothing to drain.
	inline uint64_t drain_acquire_semaphore(GpuQueue* queue, GpuSurface* surface) {
		if(surface->current_semaphore == uint32_t(-1)) return 0;
		auto semaphore = surface->images[surface->current_image].available_semaphore;
		if(semaphore == VK_NULL_HANDLE) return 0; // Something waited on it already

		VkSemaphoreSubmitInfo wait {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.semaphore = semaphore,
			.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
		};
		VkSemaphoreSubmitInfo signal {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.semaphore = queue->command_submission_timeline_semaphore,
			.value = queue->command_submission_timeline_semaphore_next_value,
			.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
		};
		VkSubmitInfo2 info {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
			.waitSemaphoreInfoCount = 1,
			.pWaitSemaphoreInfos = &wait,
			.signalSemaphoreInfoCount = 1,
			.pSignalSemaphoreInfos = &signal
		};
		if(vkQueueSubmit2(queue->queue, 1, &info, VK_NULL_HANDLE) != VK_SUCCESS) return 0;

		surface->images[surface->current_image].available_semaphore = VK_NULL_HANDLE;
		auto drained_after = queue->command_submission_timeline_semaphore_next_value++;
		surface->image_available_free_after[surface->current_semaphore] = drained_after;
		return drained_after;
	}

	// True once the presentation engine and the GPU are both done with a retired swapchain
	inline bool retired_swapchain_is_idle(GpuQueue* queue, const GpuSurface* surface, const GpuSurface::RetiredSwapchain& retired) {
		uint64_t reached = 0;
		vkGetSemaphoreCounterValue(queue->device, queue->command_submission_timeline_semaphore, &reached);
		if(reached < retired.free_after_submission) return false; // A submission still names its views

		// How many presents have gone out since this was retired. Once the surface has presented
		// more images than the retired swapchain even had, the presentation engine has cycled past
		// all of them -- which is the only answer available without a fence to ask.
		bool cycled_past = surface->presents_issued > retired.retired_at_present + retired.image_count;

		if(queue->present_fences_supported) {
			bool all_signalled = true;
			for(auto fence: retired.present_fences)
				if(vkGetFenceStatus(queue->device, fence) != VK_SUCCESS) { all_signalled = false; break; }
			// The fences give the exact answer, and earlier than the count does -- but a present
			// that failed may never signal the fence it was handed, so the count still has to be
			// able to release the swapchain or a failed present would strand it here forever.
			return all_signalled || cycled_past;
		}

		return cycled_past;
	}
}

// Destroy whatever retired swapchains are now idle, and hand their present fences back.
//
// Called from the acquire path, so a long resize cleans up as it goes rather than at the end.
static void reclaim_retired_swapchains(GpuQueue* queue, GpuSurface* surface, bool force) {
	if(surface->retired.empty()) return;

	// The pile only grows if presents stopped happening, which is the one case where there is
	// nothing left to wait for but the device itself
	if(force || surface->retired.size() > 4) {
		vkDeviceWaitIdle(queue->device);
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
			// Only a signalled fence is known to be out of the driver's hands. An unsignalled one
			// here belongs to a present that failed and was released by the frame count instead, so
			// it can be neither reset nor destroyed yet.
			if(vkGetFenceStatus(queue->device, fence) == VK_SUCCESS) {
				vkResetFences(queue->device, 1, &fence);
				surface->free_present_fences.push_back(fence);
			} else surface->quarantined_present_fences.push_back(fence);
		}
		if(retired.swapchain) vkb::destroy_swapchain(*retired.swapchain);
		surface->retired.erase(surface->retired.begin() + i);
	}
}

// Move the current swapchain, its views and its presents aside to be destroyed once idle, instead
// of destroying them now. This is what VUID-vkDestroySwapchainKHR-swapchain-01282 was reporting:
// reconfiguring used to destroy the outgoing swapchain the moment the new one was built, with
// frames still in flight against it.
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
	// Emptied rather than reused, so the acquire and present paths build a fresh set for the new
	// swapchain and nothing carries a stale signal across
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

	for(auto semaphore: surface->image_available_semaphores)
		vkDestroySemaphore(queue->device, semaphore, queue->callbacks);
	for(auto semaphore: surface->render_finished_semaphores)
		vkDestroySemaphore(queue->device, semaphore, queue->callbacks);
	for(auto fence: surface->free_present_fences)
		vkDestroyFence(queue->device, fence, queue->callbacks);
	surface->~GpuSurface();
	queue->cpu_allocator(surface, 0);
}

void gpuSurfaceReconfigureEXT(GpuQueue* queue, GpuSurface* surface, const GpuSurfaceDescriptor& desc) {
	surface->descriptor = desc;
	// A presentable texture is a single 2D image with no mips, no array layers and no
	// multisampling, whatever was asked for. The WebGPU backend normalizes to exactly this, so
	// gpuSurfaceGetConfigurationEXT reports the same thing on both.
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
	// The requested mode first, then the fallbacks a PRESENT_MODE_BEST_AVAILABLE request starts
	// from: mailbox (tear free at the lowest latency), then relaxed fifo, then the fifo every
	// surface has. Spelled out rather than left to use_default_present_mode_selection, whose order
	// skips relaxed fifo, so that both backends resolve BEST_AVAILABLE the same way and
	// gpuGetSurfaceCapabilities can report one order for it.
	if(desc.presentMode != PRESENT_MODE_BEST_AVAILABLE)
		builder.set_desired_present_mode(GPU::detail::present2vulkan(desc.presentMode));
	builder.add_fallback_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
		.add_fallback_present_mode(VK_PRESENT_MODE_FIFO_RELAXED_KHR)
		.add_fallback_present_mode(VK_PRESENT_MODE_FIFO_KHR);
	// FORMAT_NONE means "whatever the surface prefers", which is what leaving the desired format
	// unset asks for. Naming VK_FORMAT_UNDEFINED instead would match nothing and fall back to
	// whichever format the surface happens to list first.
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
		// Nothing was created, so the surface keeps the swapchain it already had rather than
		// having *swap dereferenced out from under it (a zero sized window lands here)
		errno = swap.error().value();
		return;
	}

	// An image acquired from the outgoing swapchain and never presented leaves its acquire semaphore
	// signalled with nothing left to consume it, but that semaphore is retired along with the
	// swapchain and destroyed once idle, so there is nothing to unwind here.
	surface->image_acquired = false;
	surface->current_semaphore = uint32_t(-1);

	retire_swapchain(queue, surface);
	surface->swapchain = std::make_shared<vkb::Swapchain>(std::move(*swap));
	// What the swapchain settled on, which is not necessarily what was asked for: a FORMAT_NONE
	// request in particular means "whatever the surface prefers", and the caller still has to be able
	// to build a pipeline targeting it
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

	uint32_t count = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(queue->gpu, surface->surface, &count, nullptr);
	std::vector<VkSurfaceFormatKHR> formats(count);
	vkGetPhysicalDeviceSurfaceFormatsKHR(queue->gpu, surface->surface, &count, formats.data());

	// The swapchain is only ever built for the nonlinear sRGB color space, so a format offered in
	// any other one is not a format this surface could actually be configured with
	auto usable = [](VkSurfaceFormatKHR format) {
		return format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
			&& GPU::detail::vulkan2format(format.format) != FORMAT_NONE;
	};
	auto append = [&out](VkFormat format) {
		auto named = GPU::detail::vulkan2format(format);
		auto listed = out.formatList();
		if(std::ranges::find(listed, named) == listed.end() && out.formatCount < GPU_MAX_SURFACE_FORMATS)
			out.formats[out.formatCount++] = named;
	};

	// vk-bootstrap asks for these two first whenever the caller names no format of its own, and
	// falls back to whatever the driver listed first, so walking them in that order makes
	// formats[0] the format a FORMAT_NONE request resolves to
	for(auto preferred: {VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB})
		for(auto format: formats)
			if(format.format == preferred && usable(format))
				append(format.format);
	for(auto format: formats)
		if(usable(format))
			append(format.format);

	vkGetPhysicalDeviceSurfacePresentModesKHR(queue->gpu, surface->surface, &count, nullptr);
	std::vector<VkPresentModeKHR> modes(count);
	vkGetPhysicalDeviceSurfacePresentModesKHR(queue->gpu, surface->surface, &count, modes.data());

	// The fallback order gpuSurfaceReconfigureEXT hands the swapchain builder, so presentModes[0]
	// is what PRESENT_MODE_BEST_AVAILABLE resolves to. Immediate trails the tear free modes rather
	// than leading on its latency, matching the mode that fallback list never reaches for.
	for(auto mode: {PRESENT_MODE_MAILBOX, PRESENT_MODE_FIFO_RELAXED, PRESENT_MODE_FIFO, PRESENT_MODE_IMMEDIATE})
		if(std::ranges::find(modes, GPU::detail::present2vulkan(mode)) != modes.end()
			&& out.presentModeCount < GPU_MAX_SURFACE_PRESENT_MODES)
			out.presentModes[out.presentModeCount++] = mode;

	VkSurfaceCapabilitiesKHR caps {};
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(queue->gpu, surface->surface, &caps);
	// Inherit leaves the blending to whatever the native window was set up for, which is not a
	// promise the compositor will honor an alpha channel
	out.supportsTransparency = caps.supportedCompositeAlpha
		& (VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR | VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR);

	return out;
}

GpuSurfaceDescriptor gpuSurfaceGetConfigurationEXT(const GpuSurface* surface) {
	return surface->descriptor;
}

const GpuTexture* gpuSurfaceNextTextureEXT(GpuQueue* queue, GpuSurface* surface) {
	// One acquired image at a time. A surface only lets a few be held at once, and holding one
	// while asking for another is how that limit gets exceeded; it also loses track of the
	// semaphore the held image was acquired with. Presenting is what releases it.
	if(surface->image_acquired) {
		GPU::detail::report(queue, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuSurfaceNextTextureEXT: an image is already acquired and not yet presented, so no new one is handed out; present the one you have first");
		errno = VK_NOT_READY;
		return nullptr;
	}

	// Swapchains a resize left behind, destroyed here as they fall idle so that a long drag cleans
	// up as it goes instead of piling up until it ends
	reclaim_retired_swapchains(queue, surface, false);

	// A fresh set for each swapchain generation; the previous one went into retirement with the
	// swapchain it was used against, so there is nothing here to destroy or wait for
	if(surface->image_available_semaphores.size() != surface->images.size()) {
		surface->image_available_semaphores.resize(surface->images.size());
		for(auto& semaphore: surface->image_available_semaphores) {
			VkSemaphoreCreateInfo info { info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
			vkCreateSemaphore(queue->device, &info, queue->callbacks, &semaphore);
		}
		surface->image_available_free_after.assign(surface->images.size(), 0);
		surface->semaphore_counter = -1;
	}

	surface->semaphore_counter = (surface->semaphore_counter + 1) % surface->image_available_semaphores.size();
	// The frame that last used this semaphore has to have finished before it can be signalled
	// again, which is a wait the CPU does here rather than a race it loses later
	GPU::detail::wait_for_submission(queue, surface->image_available_free_after[surface->semaphore_counter]);

	auto acquired = vkAcquireNextImageKHR(queue->device, surface->swapchain->swapchain, UINT64_MAX, surface->image_available_semaphores[surface->semaphore_counter], VK_NULL_HANDLE, &surface->current_image);
	// A suboptimal swapchain no longer matches the window but can still be rendered into and
	// presented, so the image is handed back and it is up to the caller whether that is worth a
	// reconfiguration. The WebGPU backend reports the same condition the same way.
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

void gpuSurfacePresentEXT(GpuQueue* queue, GpuSurface* surface, uint64_t wait_submission_index /*= NO_SUBMISSION_WAIT */) {
	// Nothing was acquired, so there is no image to hand back
	if(!surface->image_acquired) {
		GPU::detail::report(queue, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuSurfacePresentEXT: no image is currently acquired, so nothing is presented; call gpuSurfaceNextTextureEXT first");
		return;
	}
	surface->image_acquired = false;

	// The acquire semaphore is waited on by whatever recorded a render pass against the image. If
	// nothing did -- a frame that acquired and then changed its mind -- it is still signalled, and
	// signalling it again on its next turn would be invalid, so it gets drained here. Otherwise the
	// submission being presented is the one that waited on it, and that is when it comes free.
	if(!GPU::detail::drain_acquire_semaphore(queue, surface) && surface->current_semaphore != uint32_t(-1))
		surface->image_available_free_after[surface->current_semaphore] =
			wait_submission_index == NO_SUBMISSION_WAIT
				? queue->command_submission_timeline_semaphore_next_value - 1
				: wait_submission_index;
	surface->current_semaphore = uint32_t(-1);

	if(wait_submission_index != NO_SUBMISSION_WAIT) {
		if(surface->render_finished_semaphores.size() != surface->images.size()) {
			surface->render_finished_semaphores.resize(surface->images.size());
			for(auto& semaphore: surface->render_finished_semaphores) {
				VkSemaphoreCreateInfo info { info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
				vkCreateSemaphore(queue->device, &info, queue->callbacks, &semaphore);
			}
			surface->render_finished_free_after.assign(surface->images.size(), 0);
		}

		// This one is signalled below and waited on by the presentation engine, so it is only free
		// to be signalled again once that present has been consumed. The image index it is keyed by
		// does not come back around until the presentation engine is done with it, but the frame
		// that signalled it still has to have finished for the signal itself to be idle.
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

	// A fence per present, where the device can give one. It is the only thing that says when the
	// presentation engine has finished with an image, and so the only thing that can tell a retired
	// swapchain it is safe to destroy without draining the whole device to find out.
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
	// Recorded whether or not the present succeeded: a fence handed to a failed present is not
	// signalled, but it was still consumed, and the swapchain still needs it accounted for
	if(present_fence) surface->present_fences.push_back(present_fence);
	++surface->presents_issued;

	if(presented != VK_SUCCESS && presented != VK_SUBOPTIMAL_KHR) errno = presented;
}