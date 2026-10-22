#include "noapi.hpp"
#include "../slang_compiler.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif


namespace GPU {
	inline bool operator==(const GpuStencil& a, const GpuStencil& b) {
		return a.test == b.test && a.failOp == b.failOp && a.passOp == b.passOp
			&& a.depthFailOp == b.depthFailOp && a.reference == b.reference;
	}

	inline bool operator==(const GpuDepthStencilDesc& a, const GpuDepthStencilDesc& b) {
		return a.depthMode == b.depthMode && a.depthTest == b.depthTest && a.depthBias == b.depthBias
			&& a.depthBiasSlopeFactor == b.depthBiasSlopeFactor && a.depthBiasClamp == b.depthBiasClamp
			&& a.stencilReadMask == b.stencilReadMask && a.stencilWriteMask == b.stencilWriteMask
			&& a.stencilFront == b.stencilFront && a.stencilBack == b.stencilBack;
	}

	inline bool operator==(const GpuBlendDesc& a, const GpuBlendDesc& b) {
		return a.colorOp == b.colorOp && a.srcColorFactor == b.srcColorFactor && a.dstColorFactor == b.dstColorFactor
			&& a.alphaOp == b.alphaOp && a.srcAlphaFactor == b.srcAlphaFactor && a.dstAlphaFactor == b.dstAlphaFactor
			&& a.colorWriteMask == b.colorWriteMask;
	}
}

void gpuDefaultErrorCallbackEXT(void* queue, int type, GpuStringView message) {
	std::cerr << "WGPU Device Error " << type << ": " << std::string_view(message) << std::endl;
#ifdef _WIN32
	system("pause");
#endif
	exit(-1);
}

// Copies why the setup gave up into the caller's buffer (which is allowed to be absent) and reports the failure
static bool setup_failed(char* out_error, size_t out_error_capacity, std::string_view message) {
	if(out_error && out_error_capacity) {
		auto length = std::min(message.size(), out_error_capacity - 1);
		memcpy(out_error, message.data(), length);
		out_error[length] = '\0';
	}
	return false;
}

namespace GPU {
#ifdef __EMSCRIPTEN__
	constexpr static WGPUFeatureName optional_features[] = {WGPUFeatureName_Depth32FloatStencil8, WGPUFeatureName_DualSourceBlending};
#else
	constexpr static WGPUFeatureName optional_features[] = {WGPUFeatureName_Depth32FloatStencil8, WGPUFeatureName_DualSourceBlending, WGPUFeatureName_TextureFormatsTier1};
#endif
}

bool gpuSetupDefaultWebGPUEXT(GpuWebGPUSurfaceLoaderEXT surface_loader, void* surface_loader_userdata,
	GpuErrorCallbackEXT error_callback, bool prefer_high_power,
	GpuWebGPUDefault* out_default, char* out_error, size_t out_error_capacity
) {
	GpuWebGPUDefault out;

	{ // Instance
		WGPUInstanceDescriptor d {

		};
		out.instance = wgpuCreateInstance(&d);
	}
	if(!out.instance) return setup_failed(out_error, out_error_capacity, "Failed to create WebGPU instance");

	out.surface = surface_loader ? surface_loader(out.instance, surface_loader_userdata) : nullptr;

	{ // Adapter
		WGPURequestAdapterOptions adapter_opts {
			.powerPreference = prefer_high_power ? WGPUPowerPreference_HighPerformance : WGPUPowerPreference_LowPower,
			.compatibleSurface = out.surface,
		};

		struct RequestAdapterResult {
			WGPUAdapter adapter;
			volatile bool done;
		} result = {};
		wgpuInstanceRequestAdapter(out.instance, &adapter_opts, {
			.mode = WGPUCallbackMode_AllowSpontaneous,
			.callback = +[](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* userdata1, void* userdata2) {
				RequestAdapterResult* result = (RequestAdapterResult*)userdata1;
				auto error_callback = *(GpuErrorCallbackEXT*)userdata2;
				result->done = true;
				if (status != WGPURequestAdapterStatus_Success)
					error_callback(nullptr, WGPUErrorType_Validation, {message.data, message.length});
				result->adapter = adapter;
			},
			.userdata1 = &result,
			.userdata2 = &error_callback,
		});
		while (!result.done) {
			wgpuInstanceProcessEvents(out.instance);
		#ifdef __EMSCRIPTEN__
			emscripten_sleep(1); // yields back to the browser event loop
		#endif
		}
		out.adapter = result.adapter;
	}
	// Nothing outlives a failure, as the header promises
	auto fail = [&](std::string_view message) {
		if(out.adapter) wgpuAdapterRelease(out.adapter);
		if(out.surface) wgpuSurfaceRelease(out.surface);
		wgpuInstanceRelease(out.instance);
		return setup_failed(out_error, out_error_capacity, message);
	};
	if(!out.adapter) return fail("Failed to create WebGPU adapter");

	out.limits = {}; // Emscripten apparently asserts that the limits have been zeroed out!
	wgpuAdapterGetLimits(out.adapter, &out.limits);

	{ // Device
		// The formats compute.h marks EXT and dual source blending; asked for only where the adapter has them, so the queue can tell the difference
		std::vector<WGPUFeatureName> features;
		for(auto feature: GPU::optional_features)
			if(wgpuAdapterHasFeature(out.adapter, feature)) features.push_back(feature);

		WGPUDeviceDescriptor device_desc = {
			.requiredFeatureCount = features.size(),
			.requiredFeatures = features.data(),
			.requiredLimits = &out.limits,
			.deviceLostCallbackInfo = {
				.mode = WGPUCallbackMode_AllowSpontaneous,
				.callback = +[](WGPUDevice const* device, WGPUDeviceLostReason reason, WGPUStringView message, void* userdata1, void* _) {
					if(reason == WGPUDeviceLostReason_Destroyed) return;
					((GpuErrorCallbackEXT)userdata1)(nullptr, (int)reason, {message.data, message.length});
				},
				.userdata1 = (void*)error_callback
			},
			.uncapturedErrorCallbackInfo = {
				.callback = +[](WGPUDevice const* device, WGPUErrorType type, WGPUStringView message, void* userdata1, void* _){
					((GpuErrorCallbackEXT)userdata1)(nullptr, (int)type, {message.data, message.length});
				},
				.userdata1 = (void*)error_callback
			}
		};

		struct RequestDeviceResult {
			WGPUDevice device;
			volatile bool done;
		} result = {};
		wgpuAdapterRequestDevice(out.adapter, &device_desc, {
			.mode = WGPUCallbackMode_AllowSpontaneous,
			.callback = +[](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1, void* userdata2) {
				RequestDeviceResult* result = (RequestDeviceResult*)userdata1;
				auto error_callback = *(GpuErrorCallbackEXT*)userdata2;
				result->done = true;
				if (status != WGPURequestDeviceStatus_Success)
					error_callback(nullptr, WGPUErrorType_Validation, {message.data, message.length});
				result->device = device;
			},
			.userdata1 = &result,
			.userdata2 = &error_callback,
		});
		while (!result.done) {
			wgpuInstanceProcessEvents(out.instance);
		#ifdef __EMSCRIPTEN__
			emscripten_sleep(1); // yields back to the browser event loop
		#endif
		}
		out.device = result.device;
	}
	if(!out.device) return fail("Failed to create WebGPU device");

	*out_default = out;
	return true;
}

// A narrower pointer would truncate the monobuffer tag, so the web build targets wasm64 (see -sMEMORY64 in the top level CMakeLists).
static_assert(sizeof(gpu*) == sizeof(uint64_t), "A gpu* has to be 64 bits wide to carry a device address");

// An address is a monobuffer tag (its index plus one, so no address is null) above this many bits of offset
constexpr static uint32_t gpu_address_bits = 61;

constexpr static uint64_t gpu_address_max = (uint64_t(1) << gpu_address_bits) - 1;

gpu* gpuEncodeWebGPUAddressEXT(uint8_t monobuffer, uint64_t address) {
	assert(monobuffer < GpuQueue{}.monobuffers.size());
	assert(address <= gpu_address_max);

	return (gpu*)(uint64_t(monobuffer + 1) << gpu_address_bits | address);
}

GpuWebGPUAddressEXT gpuDecodeWebGPUAddressEXT(gpu* addr) {
	auto address = (uint64_t)addr;
	return {(uint8_t)((address >> gpu_address_bits) - 1), address & gpu_address_max};
}

namespace GPU {
	static const char* semaphore_wgsl_code = R"WGSL(
struct SemaphoreData {
	lo: atomic<u32>,
	hi: atomic<u32>,
}

struct SetValue {
	lo: u32,
	hi: u32,
}

@group(0) @binding(0) var<storage, read_write> semaphore_buf: SemaphoreData;
@group(0) @binding(1) var<uniform> set_value: SetValue;

@compute @workgroup_size(1)
fn cs_increment() {
	loop {
		let old_lo = atomicLoad(&semaphore_buf.lo);
		let new_lo = old_lo + 1u; // wraps 0xFFFFFFFF -> 0 on overflow
		let r = atomicCompareExchangeWeak(&semaphore_buf.lo, old_lo, new_lo);
		if (r.exchanged) {
			if (new_lo == 0u) {
				atomicAdd(&semaphore_buf.hi, 1u);
			}
			break;
		}
		// else: another invocation raced us on `lo`, retry.
	}
}

@compute @workgroup_size(1)
fn cs_set() {
	atomicStore(&semaphore_buf.lo, set_value.lo);
	atomicStore(&semaphore_buf.hi, set_value.hi);
}

@compute @workgroup_size(1)
fn cs_set_max() {
	let cur_lo = atomicLoad(&semaphore_buf.lo);
	let cur_hi = atomicLoad(&semaphore_buf.hi);
	let new_lo = set_value.lo;
	let new_hi = set_value.hi;

	let is_greater = (new_hi > cur_hi) || (new_hi == cur_hi && new_lo > cur_lo);
	if (is_greater) {
		atomicStore(&semaphore_buf.lo, new_lo);
		atomicStore(&semaphore_buf.hi, new_hi);
	}
})WGSL";

	inline void ensure_semaphore_pipelines(GpuQueue* queue) {
		if(queue->semaphore_bind_group_layout) return;

		std::array<WGPUBindGroupLayoutEntry, 2> entries = {
			WGPUBindGroupLayoutEntry{
				.binding = 0,
				.visibility = WGPUShaderStage_Compute,
				.buffer = {
					.type = WGPUBufferBindingType_Storage,
					.minBindingSize = sizeof(uint64_t),
				}
			}, WGPUBindGroupLayoutEntry{
				.binding = 1,
				.visibility = WGPUShaderStage_Compute,
				.buffer = {
					.type = WGPUBufferBindingType_Uniform,
					.minBindingSize = sizeof(uint64_t),
				}
			}
		};

		WGPUBindGroupLayoutDescriptor bind_group{
			.entryCount = entries.size(),
			.entries = entries.data(),
		};
		queue->semaphore_bind_group_layout = wgpuDeviceCreateBindGroupLayout(queue->device, &bind_group);

		WGPUPipelineLayoutDescriptor pipeline_layout_desc{
			.bindGroupLayoutCount = 1,
			.bindGroupLayouts = &queue->semaphore_bind_group_layout,
		};
		WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(queue->device, &pipeline_layout_desc);

		WGPUShaderSourceWGSL wgsl{
			.chain = {.sType = WGPUSType_ShaderSourceWGSL},
			.code = {semaphore_wgsl_code, WGPU_STRLEN}
		};
		WGPUShaderModuleDescriptor shader_module_desc{
			.nextInChain = &wgsl.chain
		};
		WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(queue->device, &shader_module_desc);

		WGPUComputePipelineDescriptor increment{
			.layout = pipeline_layout,
			.compute = {
				.module = shader_module,
				.entryPoint = {"cs_increment", WGPU_STRLEN},
			},
		};
		queue->semaphore_increment_pipeline = wgpuDeviceCreateComputePipeline(queue->device, &increment);

		WGPUComputePipelineDescriptor set{
			.layout = pipeline_layout,
			.compute = {
				.module = shader_module,
				.entryPoint = {"cs_set", WGPU_STRLEN},
			},
		};
		queue->semaphore_set_pipeline = wgpuDeviceCreateComputePipeline(queue->device, &set);

		WGPUComputePipelineDescriptor set_max{
			.layout = pipeline_layout,
			.compute = {
				.module = shader_module,
				.entryPoint = {"cs_set_max", WGPU_STRLEN},
			},
		};
		queue->semaphore_set_max_pipeline = wgpuDeviceCreateComputePipeline(queue->device, &set_max);

		wgpuShaderModuleRelease(shader_module);
		wgpuPipelineLayoutRelease(pipeline_layout);
	}

	inline uint64_t semaphore_cpu_set(GpuQueue* queue, GpuSemaphore sema, uint64_t value = 1) {
		wgpuQueueWriteBuffer(queue->queue, sema.buffer, 0, &value, sizeof(value));
		return value;
	}

	inline GpuSemaphore semaphore_initialize(GpuQueue* queue, uint64_t initial_value) {
		GpuSemaphore out{};

		WGPUBufferDescriptor buffer{
			.label = {"NoAPI Semaphore Storage Buffer", WGPU_STRLEN},
			.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst,
			.size = sizeof(uint64_t),
		};
		out.buffer = wgpuDeviceCreateBuffer(queue->device, &buffer);

		WGPUBufferDescriptor readback{
			.label = {"NoAPI Semaphore Value Readback Buffer", WGPU_STRLEN},
			.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
			.size = sizeof(uint64_t),
		};
		out.readback_buffer = wgpuDeviceCreateBuffer(queue->device, &readback);

		WGPUBufferDescriptor upload{
			.label = {"NoAPI Semaphore Value Upload Buffer", WGPU_STRLEN},
			.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
			.size = sizeof(uint64_t),
		};
		out.upload_buffer = wgpuDeviceCreateBuffer(queue->device, &upload);

		ensure_semaphore_pipelines(queue);

		std::array<WGPUBindGroupEntry, 2> entries = {
			WGPUBindGroupEntry{
				.binding = 0,
				.buffer = out.buffer,
				.size = sizeof(uint64_t),
			}, WGPUBindGroupEntry{
				.binding = 1,
				.buffer = out.upload_buffer,
				.size = sizeof(uint64_t),
			}
		};
		WGPUBindGroupDescriptor bind_group{
			.layout = queue->semaphore_bind_group_layout,
			.entryCount = entries.size(),
			.entries = entries.data(),
		};
		out.bind_group = wgpuDeviceCreateBindGroup(queue->device, &bind_group);

		semaphore_cpu_set(queue, out, initial_value);
		return out;
	}
}

namespace GPU::detail {
	// The map is indexed by packed description, so it needs one word per value up to the largest
	constexpr static uint32_t sampler_lookup_size = gpuSamplerDescMaxPackedEXT() + 1;

	constexpr static WGPUAddressMode address2wgpu(ADDRESS_MODE mode) {
		switch (mode) {
		case ADDRESS_MODE_CLAMP: return WGPUAddressMode_ClampToEdge;
		case ADDRESS_MODE_MIRROR_REPEAT: return WGPUAddressMode_MirrorRepeat;
		case ADDRESS_MODE_REPEAT: return WGPUAddressMode_Repeat;
		}
		std::unreachable();
	}

	constexpr static WGPUFilterMode filter2wgpu(FILTER filter) {
		return filter == FILTER_NEAREST ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear;
	}

	constexpr static WGPUMipmapFilterMode mip_filter2wgpu(FILTER filter) {
		return filter == FILTER_NEAREST ? WGPUMipmapFilterMode_Nearest : WGPUMipmapFilterMode_Linear;
	}

	// Slot 0 is always the default sampler, as on Vulkan, so an unenabled description still resolves to something valid.
	inline GpuQueue::SamplerSet* ensure_sampler_set(GpuQueue* queue, std::span<const GpuSamplerDesc> requested) {
		// One slot is spent on the default sampler, so the caller gets the rest
		assert(requested.size() < gpu_sampler_slot_count && "More samplers enabled at once than there are sampler slots");
		// The assert is gone in release builds, where the clamp would silently give extra samplers the default.
		if(requested.size() >= gpu_sampler_slot_count) errno = WGPUErrorType_Validation;
		auto count = std::min<size_t>(requested.size(), gpu_sampler_slot_count - 1);

		std::vector<GpuSamplerDesc> enabled(count + 1, GpuSamplerDesc{});
		std::copy_n(requested.begin(), count, enabled.begin() + 1);

		if(auto found = queue->sampler_cache.find(enabled); found != queue->sampler_cache.end())
			return &found->second;

		auto& set = queue->sampler_cache[enabled];

		// Every slot must be filled to match the layout, so unrequested ones get the default sampler.
		set.samplers.reserve(gpu_sampler_slot_count);
		for(uint32_t i = 0; i < gpu_sampler_slot_count; ++i) {
			auto& desc = enabled[i < enabled.size() ? i : 0];
			WGPUSamplerDescriptor d {
				.label = {"NoAPI Sampler", WGPU_STRLEN},
				.addressModeU = address2wgpu(desc.address_mode_u),
				.addressModeV = address2wgpu(desc.address_mode_v),
				.addressModeW = address2wgpu(desc.address_mode_w),
				.magFilter = filter2wgpu(desc.mag_filter),
				.minFilter = filter2wgpu(desc.min_filter),
				.mipmapFilter = mip_filter2wgpu(desc.mip_filter),
				.lodMinClamp = 0,
				.lodMaxClamp = 32, // Well past the mip count of any texture WebGPU can allocate
				.maxAnisotropy = 1,
			};
			set.samplers.emplace_back(wgpuDeviceCreateSampler(queue->device, &d));
		}

		std::vector<WGPUBindGroupEntry> entries; entries.reserve(set.samplers.size());
		for(uint32_t i = 0; i < set.samplers.size(); ++i)
			entries.push_back({
				.binding = i,
				.sampler = set.samplers[i],
			});
		WGPUBindGroupDescriptor bind_group {
			.label = {"NoAPI Samplers", WGPU_STRLEN},
			.layout = queue->current_bind_group_layout3,
			.entryCount = static_cast<uint32_t>(entries.size()),
			.entries = entries.data(),
		};
		set.bind_group = wgpuDeviceCreateBindGroup(queue->device, &bind_group);

		// Walked backwards so that the lowest slot wins if the same description was enabled twice
		std::vector<uint32_t> lookup(sampler_lookup_size, 0);
		for(size_t i = enabled.size(); i--; )
			lookup[gpuSamplerDescPackEXT(enabled[i])] = i;

		WGPUBufferDescriptor buffer {
			.label = {"NoAPI Sampler Map", WGPU_STRLEN},
			.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
			.size = lookup.size() * sizeof(uint32_t),
		};
		set.lookup_buffer = wgpuDeviceCreateBuffer(queue->device, &buffer);
		wgpuQueueWriteBuffer(queue->queue, set.lookup_buffer, 0, lookup.data(), buffer.size);

		return &set;
	}
}

namespace GPU {
	inline WGPUTextureViewDimension texture_view2wgpu(TEXTURE type){
		switch (type){
		case TEXTURE_1D: // Never reached: GPU::texture_supported declines 1D textures before any view exists
		case TEXTURE_2D:
		case TEXTURE_2D_ARRAY:
			return WGPUTextureViewDimension_2DArray;
		// A monotexture packs several cubes, so even a single one is viewed as an array
		case TEXTURE_CUBE:
		case TEXTURE_CUBE_ARRAY:
			return WGPUTextureViewDimension_CubeArray;
		case TEXTURE_3D:
			return WGPUTextureViewDimension_3D;
		}
		std::unreachable();
	}

	inline WGPUTextureFormat format2wgpu(FORMAT format){
		switch (format) {
		case FORMAT_R8_UNORM: return WGPUTextureFormat_R8Unorm;
		case FORMAT_R8_SNORM: return WGPUTextureFormat_R8Snorm;
		case FORMAT_R8_UINT: return WGPUTextureFormat_R8Uint;
		case FORMAT_R8_SINT: return WGPUTextureFormat_R8Sint;

		// The 16-bit normalized formats come from Dawn's unorm16/snorm16 extensions; emdawnwebgpu doesn't expose them, so on the web they fall through to WGPUTextureFormat_Undefined
#ifndef __EMSCRIPTEN__
		case FORMAT_R16_UNORM: return WGPUTextureFormat_R16Unorm;
		case FORMAT_R16_SNORM: return WGPUTextureFormat_R16Snorm;
#endif
		case FORMAT_R16_UINT: return WGPUTextureFormat_R16Uint;
		case FORMAT_R16_SINT: return WGPUTextureFormat_R16Sint;
		case FORMAT_R16_FLOAT: return WGPUTextureFormat_R16Float;
		case FORMAT_RG8_UNORM: return WGPUTextureFormat_RG8Unorm;
		case FORMAT_RG8_SNORM: return WGPUTextureFormat_RG8Snorm;
		case FORMAT_RG8_UINT: return WGPUTextureFormat_RG8Uint;
		case FORMAT_RG8_SINT: return WGPUTextureFormat_RG8Sint;

		case FORMAT_R32_UINT: return WGPUTextureFormat_R32Uint;
		case FORMAT_R32_SINT: return WGPUTextureFormat_R32Sint;
		case FORMAT_R32_FLOAT: return WGPUTextureFormat_R32Float;
#ifndef __EMSCRIPTEN__
		case FORMAT_RG16_UNORM: return WGPUTextureFormat_RG16Unorm;
		case FORMAT_RG16_SNORM: return WGPUTextureFormat_RG16Snorm;
#endif
		case FORMAT_RG16_UINT: return WGPUTextureFormat_RG16Uint;
		case FORMAT_RG16_SINT: return WGPUTextureFormat_RG16Sint;
		case FORMAT_RG16_FLOAT: return WGPUTextureFormat_RG16Float;
		case FORMAT_RGBA8_UNORM: return WGPUTextureFormat_RGBA8Unorm;
		case FORMAT_RGBA8_SRGB: return WGPUTextureFormat_RGBA8UnormSrgb;
		case FORMAT_RGBA8_SNORM: return WGPUTextureFormat_RGBA8Snorm;
		case FORMAT_RGBA8_UINT: return WGPUTextureFormat_RGBA8Uint;
		case FORMAT_RGBA8_SINT: return WGPUTextureFormat_RGBA8Sint;
		case FORMAT_BGRA8_UNORM: return WGPUTextureFormat_BGRA8Unorm;
		case FORMAT_BGRA8_SRGB: return WGPUTextureFormat_BGRA8UnormSrgb;

		case FORMAT_RGB10_A2_UINT: return WGPUTextureFormat_RGB10A2Uint;
		case FORMAT_RGB10_A2_UNORM: return WGPUTextureFormat_RGB10A2Unorm;
		case FORMAT_RG11B10_UFLOAT: return WGPUTextureFormat_RG11B10Ufloat;
		case FORMAT_RGB9E5_UFLOAT: return WGPUTextureFormat_RGB9E5Ufloat;

		case FORMAT_RG32_UINT: return WGPUTextureFormat_RG32Uint;
		case FORMAT_RG32_SINT: return WGPUTextureFormat_RG32Sint;
		case FORMAT_RG32_FLOAT: return WGPUTextureFormat_RG32Float;
#ifndef __EMSCRIPTEN__
		case FORMAT_RGBA16_UNORM: return WGPUTextureFormat_RGBA16Unorm;
		case FORMAT_RGBA16_SNORM: return WGPUTextureFormat_RGBA16Snorm;
#endif
		case FORMAT_RGBA16_UINT: return WGPUTextureFormat_RGBA16Uint;
		case FORMAT_RGBA16_SINT: return WGPUTextureFormat_RGBA16Sint;
		case FORMAT_RGBA16_FLOAT: return WGPUTextureFormat_RGBA16Float;

		case FORMAT_RGBA32_UINT: return WGPUTextureFormat_RGBA32Uint;
		case FORMAT_RGBA32_SINT: return WGPUTextureFormat_RGBA32Sint;
		case FORMAT_RGBA32_FLOAT: return WGPUTextureFormat_RGBA32Float;

		case FORMAT_S8_UINT: return WGPUTextureFormat_Stencil8;
		case FORMAT_D16_UNORM: return WGPUTextureFormat_Depth16Unorm;
		case FORMAT_D24_PLUS: return WGPUTextureFormat_Depth24Plus;
		case FORMAT_D24_PLUS_S8_UINT: return WGPUTextureFormat_Depth24PlusStencil8;
		case FORMAT_D32_FLOAT: return WGPUTextureFormat_Depth32Float;
		case FORMAT_D32_FLOAT_S8_UINT: return WGPUTextureFormat_Depth32FloatStencil8;

		default:
			return WGPUTextureFormat_Undefined;
		}
	}
}

// Group 0 holds the buffers. WebGPU forbids writable storage buffers in the vertex stage, so the graphics flavor binds the monobuffers read only.
WGPUBindGroupLayout create_buffer_bind_group_layout(GpuQueue* queue, bool compute) {
	auto visibility = compute ? WGPUShaderStage_Compute : WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;

	uint32_t binding = 0;
	std::vector<WGPUBindGroupLayoutEntry> group0;

	for(uint32_t i = 0; i < queue->monobuffers.size(); ++i)
		group0.push_back({
			.binding = binding++,
			.visibility = visibility,
			.buffer = {
				.type = compute ? WGPUBufferBindingType_Storage : WGPUBufferBindingType_ReadOnlyStorage,
				.hasDynamicOffset = false,
				.minBindingSize = 0,
			}
		});

	// The texture heap, then the sampler lookup map and monotexture sizes; both read only
	for(uint32_t i = 0; i < 2; ++i)
		group0.push_back({
			.binding = binding++,
			.visibility = visibility,
			.buffer = {
				.type = WGPUBufferBindingType_ReadOnlyStorage,
				.hasDynamicOffset = false,
				.minBindingSize = 0,
			}
		});

	group0.push_back({
		.binding = binding++,
		.visibility = visibility,
		.buffer = {
			.type = WGPUBufferBindingType_Uniform,
			.hasDynamicOffset = false,
			.minBindingSize = 0,
		}
	});

	WGPUBindGroupLayoutDescriptor d{
		.entryCount = static_cast<uint32_t>(group0.size()),
		.entries = group0.data(),
	};
	return wgpuDeviceCreateBindGroupLayout(queue->device, &d);
}

// Group 3 is one binding per sampler slot; its shape is fixed, so it is built once and shared.
WGPUBindGroupLayout create_sampler_bind_group_layout(GpuQueue* queue) {
	std::vector<WGPUBindGroupLayoutEntry> group3;
	for(uint32_t i = 0; i < gpu_sampler_slot_count; ++i)
		group3.push_back({
			.binding = i,
			.visibility = WGPUShaderStage_Compute | WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
			.sampler = {
				// Filtering accepts a nearest sampler too, and group 2 already declares every sampled monotexture as a filterable float texture
				.type = WGPUSamplerBindingType_Filtering,
			}
		});

	WGPUBindGroupLayoutDescriptor d{
		.entryCount = static_cast<uint32_t>(group3.size()),
		.entries = group3.data(),
	};
	return wgpuDeviceCreateBindGroupLayout(queue->device, &d);
}

// A storage monotexture is also bound sampled when created sampleable with a filterable format; numbered after the sampled monotextures, in generate_backend_module's order.
static std::vector<int> storage_sampled_bindings(GpuQueue* queue) {
	std::vector<int> out(queue->storage_monotextures.size(), -1);
	auto binding = static_cast<int>(queue->sampled_monotextures.size());

	for(size_t i = 0; i < queue->storage_monotextures.size(); ++i) {
		auto const& [_cap, texture, view, desc] = queue->storage_monotextures[i];
		if(!view || !(wgpuTextureGetUsage(texture) & WGPUTextureUsage_TextureBinding)) continue;
		if(!gpuFormatIsFilterableEXT(desc.format)) continue;
		out[i] = binding++;
	}
	return out;
}

void update_pipeline_layouts(GpuQueue* queue) {
	auto create_layout_and_group = [&](std::vector<WGPUBindGroupLayoutEntry>& layoutEntries, std::vector<WGPUBindGroupEntry>& entries) -> std::pair<WGPUBindGroupLayout, WGPUBindGroup> {
		WGPUBindGroupLayoutDescriptor layout{
			.entryCount = static_cast<uint32_t>(layoutEntries.size()),
			.entries = layoutEntries.data(),
		};

		WGPUBindGroupDescriptor d {
			.layout = wgpuDeviceCreateBindGroupLayout(queue->device, &layout),
			.entryCount = static_cast<uint32_t>(entries.size()),
			.entries = entries.data(),
		};
		return {d.layout, wgpuDeviceCreateBindGroup(queue->device, &d)};
	};

	// Group 0: buffers, one flavor per pipeline type.
	if(!queue->current_compute_bind_group_layout0)
		queue->current_compute_bind_group_layout0 = create_buffer_bind_group_layout(queue, true);
	if(!queue->current_graphics_bind_group_layout0)
		queue->current_graphics_bind_group_layout0 = create_buffer_bind_group_layout(queue, false);

	// Groups 1 and 2 come in compute and graphics flavors: a pass can't bind one texture both writable and sampled.
	std::vector<WGPUBindGroupLayoutEntry> group1;
	std::vector<WGPUBindGroupEntry> group1entries;

	uint32_t binding = 0;

	// Released with the bind group they were made for, since that is what holds them
	std::vector<WGPUTextureView> storage_views;

	for(auto const& [_cap, texture, view, desc] : queue->storage_monotextures) {
		// Write only: core WebGPU only allows read/write storage for three 32 bit single channel formats.
		group1.push_back({
			.binding = binding,
			.visibility = WGPUShaderStage_Compute,
			.storageTexture = {
				.access = WGPUStorageTextureAccess_WriteOnly,
				.format = GPU::format2wgpu(desc.format),
				.viewDimension = GPU::texture_view2wgpu(desc.type),
			}
		});

		// A storage binding names exactly one mip, and this view is the whole atlas, so level 0 it is (gpuRWTextureViewDescriptor reports otherwise).
		WGPUTextureViewDescriptor d {
			.format = GPU::format2wgpu(desc.format),
			.dimension = GPU::texture_view2wgpu(desc.type),
			.baseMipLevel = 0,
			.mipLevelCount = 1,
			.baseArrayLayer = 0,
			.arrayLayerCount = desc.type == TEXTURE_3D ? 1 : wgpuTextureGetDepthOrArrayLayers(texture),
			.aspect = WGPUTextureAspect_All,
			.usage = WGPUTextureUsage_StorageBinding,
		};
		storage_views.push_back(wgpuTextureCreateView(texture, &d));

		group1entries.push_back({
			.binding = binding++,
			.textureView = storage_views.back(),
		});
	}

	// Group 2, which both flavors share the front of: one entry per sampled monotexture
	std::vector<WGPUBindGroupLayoutEntry> group2;
	std::vector<WGPUBindGroupEntry> group2entries;

	binding = 0;

	auto sampled_binding = [&](const GpuQueue::TextureHash& desc, WGPUTextureView view, uint32_t at) {
		group2.push_back({
			.binding = at,
			.visibility = WGPUShaderStage_Compute | WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
			.texture = {
				.sampleType = WGPUTextureSampleType_Float,
				.viewDimension = GPU::texture_view2wgpu(desc.type),
				.multisampled = desc.sampleCount > 1,
			}
		});
		group2entries.push_back({
			.binding = at,
			.textureView = view,
		});
	};

	for(auto const& [_cap, texture, view, desc] : queue->sampled_monotextures)
		sampled_binding(desc, view, binding++);

	// Everything above is the compute flavor of both groups, whose group 2 stops here
	auto release_later = [&](WGPUBindGroupLayout layout, WGPUBindGroup group, std::vector<WGPUTextureView> views = {}) {
		if(!layout) return;
		queue->code_pending_submission_finished.emplace_back([layout, group, views = std::move(views)](){
			wgpuBindGroupLayoutRelease(layout);
			wgpuBindGroupRelease(group);
			for(auto view: views) wgpuTextureViewRelease(view);
		}, queue->next_submission_index);
	};

	release_later(queue->current_compute_bind_group_layout1, queue->current_compute_bind_group1,
		std::move(queue->current_storage_views));
	queue->current_storage_views = std::move(storage_views);
	std::tie(queue->current_compute_bind_group_layout1, queue->current_compute_bind_group1)
		= create_layout_and_group(group1, group1entries);

	release_later(queue->current_compute_bind_group_layout2, queue->current_compute_bind_group2);
	std::tie(queue->current_compute_bind_group_layout2, queue->current_compute_bind_group2)
		= create_layout_and_group(group2, group2entries);

	// The graphics flavor: no storage textures, and sampleable storage monotextures appended to group 2 (see storage_sampled_bindings).
	auto storage_sampled = storage_sampled_bindings(queue);
	for(size_t i = 0; i < queue->storage_monotextures.size(); ++i) {
		if(storage_sampled[i] < 0) continue;
		auto const& [_cap, texture, view, desc] = queue->storage_monotextures[i];
		sampled_binding(desc, view, static_cast<uint32_t>(storage_sampled[i]));
	}

	std::vector<WGPUBindGroupLayoutEntry> no_group1;
	std::vector<WGPUBindGroupEntry> no_group1entries;

	release_later(queue->current_graphics_bind_group_layout1, queue->current_graphics_bind_group1);
	std::tie(queue->current_graphics_bind_group_layout1, queue->current_graphics_bind_group1)
		= create_layout_and_group(no_group1, no_group1entries);

	release_later(queue->current_graphics_bind_group_layout2, queue->current_graphics_bind_group2);
	std::tie(queue->current_graphics_bind_group_layout2, queue->current_graphics_bind_group2)
		= create_layout_and_group(group2, group2entries);

	// Group 3: samplers, whose shape never changes, so the layout outlives every sampler set.
	if(!queue->current_bind_group_layout3)
		queue->current_bind_group_layout3 = create_sampler_bind_group_layout(queue);

	auto create_pipeline_layout = [&](WGPUBindGroupLayout bgl0, WGPUBindGroupLayout bgl1, WGPUBindGroupLayout bgl2) {
		WGPUBindGroupLayout layouts[] = {bgl0, bgl1, bgl2, queue->current_bind_group_layout3};

		WGPUPipelineLayoutDescriptor pd{
			.bindGroupLayoutCount = 4,
			.bindGroupLayouts = layouts,
		};
		return wgpuDeviceCreatePipelineLayout(queue->device, &pd);
	};

	if(queue->current_compute_pipeline_layout) wgpuPipelineLayoutRelease(queue->current_compute_pipeline_layout);
	queue->current_compute_pipeline_layout = create_pipeline_layout(queue->current_compute_bind_group_layout0,
		queue->current_compute_bind_group_layout1, queue->current_compute_bind_group_layout2);

	if(queue->current_graphics_pipeline_layout) wgpuPipelineLayoutRelease(queue->current_graphics_pipeline_layout);
	queue->current_graphics_pipeline_layout = create_pipeline_layout(queue->current_graphics_bind_group_layout0,
		queue->current_graphics_bind_group_layout1, queue->current_graphics_bind_group_layout2);
}

GpuQueue* gpuCreateQueue(WGPUAdapter adapter, WGPUDevice device, WGPULimits limits, CpuAllocatorFunc allocator) {
	auto out = (GpuQueue*)allocator(nullptr, sizeof(GpuQueue));
	new(out) GpuQueue{
		.adapter = adapter,
		.device = device,
		.limits = limits,
		.cpu_allocator = allocator,
	};
	assert(limits.maxSamplersPerShaderStage >= gpu_sampler_slot_count && "Device is below the sampler limit WebGPU guarantees");

	out->queue = wgpuDeviceGetQueue(device);

	out->current_submission_timeline_semaphore = GPU::semaphore_initialize(out, out->next_submission_index - 1);

	WGPUBufferDescriptor d {
		.label = {"NoAPI Empty Monobuffer", WGPU_STRLEN},
		.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc | WGPUBufferUsage_Index | WGPUBufferUsage_Indirect,
		.size = 4
	};
	out->empty_buffer = wgpuDeviceCreateBuffer(out->device, &d);
	for(auto& buffer: out->monobuffers)
		buffer = wgpuDeviceCreateBuffer(out->device, &d);
	std::fill(out->monobuffer_sizes.begin(), out->monobuffer_sizes.end(), d.size);

	update_pipeline_layouts(out);

	// Every set resolves unknown descriptions to slot 0, so the default is a set of one.
	out->default_sampler_set = GPU::detail::ensure_sampler_set(out, {});

	// Properties of WebGPU itself: no mesh shaders, no 64 bit atomics, no waits on memory, and no buffer bound both writable and as a heap. Dual source blending is an optional feature.
	out->capabilities = GpuCapabilities {
		.mesh_shaders = false,
		.dual_source_blending = wgpuDeviceHasFeature(out->device, WGPUFeatureName_DualSourceBlending) != 0,
		.split_barrier_signals = false,
		.split_barriers = false,
		.gpu_writable_texture_heap = false,
		.descriptor_heap_in_any_memory = true,
		.texture_descriptor_stride_ratio = 1,
	};
	return out;
}

GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, WGPUSurface surface, const GpuSurfaceDescriptor& desc) {
	if(!surface) {
		errno = WGPUErrorType_Validation;
		return nullptr;
	}

	auto out = (GpuSurface*)queue->cpu_allocator(nullptr, sizeof(GpuSurface));
	new(out) GpuSurface {
		.surface = surface,
	};

	gpuSurfaceReconfigureEXT(queue, out, desc);
	return out;
}

namespace GPU {
	inline void semaphore_destroy(GpuSemaphore sema) {
		wgpuBindGroupRelease(sema.bind_group);
		wgpuBufferRelease(sema.upload_buffer);
		wgpuBufferRelease(sema.readback_buffer);
		wgpuBufferRelease(sema.buffer);
	}
}

namespace GPU::detail {
	inline void free_sampler_set(GpuQueue::SamplerSet& set) {
		if(set.lookup_buffer) wgpuBufferRelease(set.lookup_buffer);
		if(set.bind_group) wgpuBindGroupRelease(set.bind_group);
		for(auto sampler: set.samplers)
			wgpuSamplerRelease(sampler);
	}
}

void gpuFreeQueue(GpuQueue* queue) {
	gpuWaitIdleEXT(queue); // As on Vulkan, nothing submitted is still running once the queue is gone

	// Free what the program did not, along with the textures made on it
	std::vector<gpu*> leftovers;
	for(auto& [address, _allocation]: queue->allocations) leftovers.push_back(address);
	for(auto address: leftovers) gpuFreeDevicePointerEXT(queue, address);

	GPU::semaphore_destroy(queue->current_submission_timeline_semaphore);

	for(auto& [_enabled, set]: queue->sampler_cache)
		GPU::detail::free_sampler_set(set);
	if(queue->current_bind_group_layout3) wgpuBindGroupLayoutRelease(queue->current_bind_group_layout3);

	for(auto [_key, pipeline]: queue->blit_pipelines)
		wgpuRenderPipelineRelease(pipeline);
	for(size_t i = 0; i < queue->blit_bind_group_layouts.size(); ++i) {
		if(queue->blit_samplers[i]) wgpuSamplerRelease(queue->blit_samplers[i]);
		if(queue->blit_pipeline_layouts[i]) wgpuPipelineLayoutRelease(queue->blit_pipeline_layouts[i]);
		if(queue->blit_bind_group_layouts[i]) wgpuBindGroupLayoutRelease(queue->blit_bind_group_layouts[i]);
	}
	if(queue->blit_shader) wgpuShaderModuleRelease(queue->blit_shader);

	auto allocator = queue->cpu_allocator;
	queue->~GpuQueue();
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

namespace GPU::detail {
	// Kept sorted by monobuffer and then start, which gpuMalloc relies on to join neighbouring holes
	inline void add_free_range(GpuQueue* queue, GpuQueue::MonobufferRange range) {
		auto at = std::ranges::upper_bound(queue->buffer_freelist, range, [](const GpuQueue::MonobufferRange& a, const GpuQueue::MonobufferRange& b) {
			return std::tie(a.buffer, a.start) < std::tie(b.buffer, b.start);
		});
		queue->buffer_freelist.insert(at, range);
	}
}

void* gpuMalloc(GpuQueue* queue, size_t bytes, size_t align, MEMORY memory) {
	constexpr static auto align_up = [](size_t addr, size_t align) {
		return (addr + align - 1) & ~(align - 1);
	};
	constexpr static auto allocation_bookkeeping = [](GpuQueue* queue, uint8_t active_monobuffer, size_t gpu_address, size_t bytes, MEMORY memory) -> void* {
		auto gpu = gpuEncodeWebGPUAddressEXT(active_monobuffer, gpu_address);
		// GPU only memory has no CPU shadow, and hands back its address the way the Vulkan backend does
		const bool host_visible = memory != MEMORY_GPU && memory != MEMORY_TEXTURE;
		auto cpu = host_visible ? queue->cpu_allocator(nullptr, bytes) : nullptr;
		queue->allocations[gpu] = {GpuQueue::MonobufferRange{active_monobuffer, static_cast<uint32_t>(gpu_address), static_cast<uint32_t>(gpu_address + bytes)}, cpu, memory};
		if(!host_visible) return gpu;
		queue->cpu2gpu[cpu] = gpu;
		return cpu;
	};

	if(bytes > queue->limits.maxStorageBufferBindingSize) {
		errno = WGPUErrorType_OutOfMemory;
		return nullptr;
	}

	for(size_t i = 0; i < queue->buffer_freelist.size(); ++i) {
		// Strip any empty ranges from the list
		if(queue->buffer_freelist[i].start == queue->buffer_freelist[i].end) {
			queue->buffer_freelist.erase(queue->buffer_freelist.begin() + i);
			--i;
			continue;
		}

		// Sorted by monobuffer then start, so a hole touching or overlapping the one before it joins it, and is tried whole
		if(i > 0) {
			auto prev = queue->buffer_freelist[i - 1];
			auto& hole = queue->buffer_freelist[i];
			if(prev.buffer == hole.buffer && hole.start <= prev.end) {
				hole.start = prev.start;
				hole.end = std::max(hole.end, prev.end);
				queue->buffer_freelist.erase(queue->buffer_freelist.begin() + (i - 1));
				--i;
			}
		}

		// If we find a free space big enough for the allocation... use that
		auto& [active_buffer, start, end] = queue->buffer_freelist[i];
		auto aligned = align_up(start, align);
		// Aligning up can walk past a small hole's end, which would wrap the unsigned size; >= reuses exactly fitting holes.
		if(aligned <= end && size_t(end) - aligned >= bytes) {
			start = aligned + bytes;
			return allocation_bookkeeping(queue, active_buffer, aligned, bytes, memory);
		}
	}

	auto new_start = align_up(queue->active_monobuffer_size, align);
	auto new_end = new_start + bytes;
	auto new_active_monobuffer = queue->active_monobuffer;

	// Reallocate the monobuffer if necessary
	bool activate_next_monobuffer = false;
	if(new_end > queue->active_monobuffer_capacity) {
		// Work in flight may still bind the buffer being replaced
		queue->code_pending_submission_finished.emplace_back([buffer = queue->monobuffers[queue->active_monobuffer]]() {
			wgpuBufferRelease(buffer);
		}, queue->next_submission_index);

		queue->active_monobuffer_capacity = std::max(queue->active_monobuffer_capacity * 2, new_end); // Doubles the size of the capacity each time
		if(queue->active_monobuffer_capacity > queue->limits.maxStorageBufferBindingSize) {
			queue->active_monobuffer_capacity = queue->limits.maxStorageBufferBindingSize;
			activate_next_monobuffer = true;
		}
		if(new_end > queue->limits.maxStorageBufferBindingSize) {
			new_start = 0;
			new_end = bytes;
			++new_active_monobuffer;
		}

		WGPUBufferDescriptor d {
			.label = {"NoAPI Monobuffer", WGPU_STRLEN},
			.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc | WGPUBufferUsage_Index | WGPUBufferUsage_Indirect,
			.size = queue->active_monobuffer_capacity
		};
		auto old_monobuffer = queue->monobuffers[queue->active_monobuffer];
		queue->monobuffers[queue->active_monobuffer] = wgpuDeviceCreateBuffer(queue->device, &d);

		if(queue->active_monobuffer_size > 0) { // What was already allocated moves with it
			auto cmd = wgpuDeviceCreateCommandEncoder(queue->device, nullptr);
			wgpuCommandEncoderCopyBufferToBuffer(cmd, old_monobuffer, 0, queue->monobuffers[queue->active_monobuffer], 0, queue->active_monobuffer_size);
			auto to_submit = wgpuCommandEncoderFinish(cmd, nullptr);
			wgpuQueueSubmit(queue->queue, 1, &to_submit);
			wgpuCommandEncoderRelease(cmd);
			wgpuCommandBufferRelease(to_submit);
		}

		// A monobuffer at the binding size limit is full: what is left of it goes on the freelist, where it is bound whole, and allocation moves on to the next
		if(activate_next_monobuffer && queue->active_monobuffer + 1 < queue->monobuffers.size()) {
			auto full = queue->active_monobuffer;
			bool stayed = new_active_monobuffer == full; // The allocation still fit at the end of the full one
			GPU::detail::add_free_range(queue, {full, uint32_t(stayed ? new_end : queue->active_monobuffer_size), uint32_t(queue->limits.maxStorageBufferBindingSize)});
			queue->monobuffer_sizes[full] = queue->limits.maxStorageBufferBindingSize;

			++queue->active_monobuffer;
			WGPUBufferDescriptor d {
				.label = {"NoAPI Monobuffer", WGPU_STRLEN},
				.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc | WGPUBufferUsage_Index | WGPUBufferUsage_Indirect,
				.size = bytes
			};
			queue->monobuffers[queue->active_monobuffer] = wgpuDeviceCreateBuffer(queue->device, &d);
			queue->monobuffer_sizes[queue->active_monobuffer] = queue->active_monobuffer_size = 0;
			queue->active_monobuffer_capacity = bytes;
		} else if(new_active_monobuffer != queue->active_monobuffer) {
			// It needs a monobuffer of its own, and the last one is already in use
			errno = WGPUErrorType_OutOfMemory;
			return nullptr;
		}
	}

	// The allocation is in the active monobuffer unless it fit at the end of one that just filled up
	if(new_active_monobuffer == queue->active_monobuffer)
		queue->monobuffer_sizes[queue->active_monobuffer] = queue->active_monobuffer_size = new_end;
	return allocation_bookkeeping(queue, new_active_monobuffer, new_start, new_end - new_start, memory);
}

void gpuFree(GpuQueue* queue, void* ptr) {
	auto device = gpuHostToDevicePointer(queue, ptr);
	gpuFreeDevicePointerEXT(queue, device ? device : (gpu*)ptr);
}

namespace GPU {
	inline WGPUTextureUsage usage2wgpu(TEXTURE_USAGE_FLAGS flags) {
		WGPUTextureUsage usage = WGPUTextureUsage_None;

		if (flags & USAGE_TRANSFER_SRC)
			usage |= WGPUTextureUsage_CopySrc;

		if (flags & USAGE_TRANSFER_DST)
			usage |= WGPUTextureUsage_CopyDst;

		if (flags & USAGE_SAMPLED)
			usage |= WGPUTextureUsage_TextureBinding;

		if (flags & USAGE_STORAGE)
			usage |= WGPUTextureUsage_StorageBinding;

		if (flags & USAGE_COLOR_ATTACHMENT || flags & USAGE_DEPTH_STENCIL_ATTACHMENT)
			usage |= WGPUTextureUsage_RenderAttachment;

		return usage;
	}

	inline WGPUTextureDimension texture2wgpu(TEXTURE type){
		switch (type){
		case TEXTURE_1D: // Never reached: GPU::texture_supported declines 1D textures
		case TEXTURE_2D:
		case TEXTURE_2D_ARRAY:
		case TEXTURE_CUBE:
		case TEXTURE_CUBE_ARRAY:
			return WGPUTextureDimension_2D;

		case TEXTURE_3D:
			return WGPUTextureDimension_3D;
		}
		std::unreachable();
	}

	inline WGPUTextureDescriptor texture2wgpu(const GpuTextureDesc& src, std::string_view label = "") {
		return WGPUTextureDescriptor{
			.label = { .data = label.data(), .length = label.size() },
			.usage = usage2wgpu(src.usage),
			.dimension = texture2wgpu(src.type),
			.size = {
				.width  = src.dimensions.x,
				.height = src.dimensions.y,
				.depthOrArrayLayers = (src.type == TEXTURE_2D_ARRAY || src.type == TEXTURE_CUBE_ARRAY) ? src.layerCount : src.dimensions.z,
			},
			.format = format2wgpu(src.format),
			.mipLevelCount = src.mipCount,
			.sampleCount   = src.sampleCount,
			.viewFormatCount = 0,
			.viewFormats     = nullptr,
		};
	}
}

// Releases an emptied atlas but keeps its slot, whose index heap entries and shaders depend on, as a 1x1 stand in gpuCreateTexture can reuse.
static void collapse_monotexture_if_empty(GpuQueue* queue, const GpuQueue::MonotextureRange& freed) {
	auto& monotextures = freed.storage() ? queue->storage_monotextures : queue->sampled_monotextures;
	auto& [size, texture, full_view, desc] = monotextures.at(freed.index());

	// `size` only grows, so it is exactly how many layers were handed out.
	uint32_t returned = 0;
	for(auto const& range: queue->texture_freelist)
		if(range._index == freed._index) returned += range.end - range.start;
	if(returned < size) return;

	// These layers belong to the atlas about to go, and the stand in has only one.
	std::erase_if(queue->texture_freelist, [&](const GpuQueue::MonotextureRange& range) {
		return range._index == freed._index;
	});

	// Work in flight may still bind them, so they go once it retires, as on the grow path.
	queue->code_pending_submission_finished.emplace_back([texture, full_view]() {
		if(full_view) wgpuTextureViewRelease(full_view);
		wgpuTextureRelease(texture);
	}, queue->next_submission_index);

	// The stand in keeps the binding's format, view dimension and sample count, so the layout and generated module are unaffected.
	auto usage = wgpuTextureGetUsage(texture);
	// A cube array view needs whole cubes
	const uint32_t layers = GPU::texture_view2wgpu(desc.type) == WGPUTextureViewDimension_CubeArray ? 6 : 1;
	{
		WGPUTextureDescriptor d {
			.label = {"NoAPI Collapsed Monotexture", WGPU_STRLEN},
			.usage = usage,
			.dimension = GPU::texture2wgpu(desc.type),
			.size = {1, 1, layers},
			.format = GPU::format2wgpu(desc.format),
			.mipLevelCount = 1,
			.sampleCount = desc.sampleCount,
		};
		texture = wgpuDeviceCreateTexture(queue->device, &d);
	}{
		// Storage bindings build their own single level views, so a storage only stand in needs none.
		auto view_usage = usage & ~WGPUTextureUsage_StorageBinding;
		WGPUTextureViewDescriptor d {
			.format = GPU::format2wgpu(desc.format),
			.dimension = GPU::texture_view2wgpu(desc.type),
			.baseMipLevel = 0,
			.mipLevelCount = 1,
			.baseArrayLayer = 0,
			.arrayLayerCount = layers,
			.aspect = WGPUTextureAspect_All,
			.usage = view_usage,
		};
		full_view = view_usage ? wgpuTextureCreateView(texture, &d) : nullptr;
	}

	size = 0;
	// Otherwise the next texture of this shape would be handed a layer of the stand in.
	auto& lookup = freed.storage() ? queue->storage_monotextures_lookup : queue->sampled_monotextures_lookup;
	lookup.erase(desc);
	queue->collapsed_monotextures.insert(freed._index);

	// The bind groups hold views of the atlas that just went away
	update_pipeline_layouts(queue);
}

void gpuFreeDevicePointerEXT(GpuQueue* queue, gpu* ptr) {
	if(!queue->allocations.contains(ptr)) return;

	// TODO: Extra buffers go here
	if(queue->gpu2textures.contains(ptr)) {
		auto texture = queue->gpu2textures[ptr];
		if(texture->range) {
			auto freed = *texture->range; // A copy, so the monotexture is still named below
			// Kept sorted by monotexture (whose _index carries the storage bit) then start, like buffer_freelist, so gpuCreateTexture can join neighbours
			auto at = std::ranges::upper_bound(queue->texture_freelist, freed, [](const GpuQueue::MonotextureRange& a, const GpuQueue::MonotextureRange& b) {
				return std::tie(a._index, a.start) < std::tie(b._index, b.start);
			});
			queue->texture_freelist.insert(at, freed);

			// That may have been the last texture the monotexture was holding
			collapse_monotexture_if_empty(queue, freed);
		} else {
			wgpuTextureRelease(texture->texture);
		}

		queue->cpu_allocator(texture, 0);
		// The address may be reused, which a stale entry would make gpuCreateTexture's one texture per allocation guard reject.
		queue->gpu2textures.erase(ptr);
	}

	auto found = queue->allocations.find(ptr);
	auto [range, cpu, _memory_type] = found->second; // A copy, so erasing below is safe
	queue->allocations.erase(found);
	if(cpu) {
		queue->cpu_allocator(cpu, 0);
		queue->cpu2gpu.erase(cpu);
	}

	GPU::detail::add_free_range(queue, range);
}

gpu* gpuHostToDevicePointer(GpuQueue* queue, void* ptr) {
	if(queue->cpu2gpu.contains(ptr))
		return queue->cpu2gpu[ptr];
	return nullptr;
}

void* gpuDeviceToHostPointerEXT(GpuQueue* queue, gpu* ptr) {
	auto found = queue->allocations.find(ptr);
	if(found == queue->allocations.end()) return nullptr;
	return std::get<void*>(found->second);
}

namespace GPU {
	// WGSL's names for the formats gpuFormatIsStorableEXT accepts, the only ones a storage texture is created in
	inline const char* storage_format_name(FORMAT format) {
		switch (format) {
		case FORMAT_R32_FLOAT: return "r32f";
		case FORMAT_RG32_FLOAT: return "rg32f";
		case FORMAT_RGBA8_SNORM: return "rgba8_snorm";
		case FORMAT_RGBA16_FLOAT: return "rgba16f";
		case FORMAT_RGBA32_FLOAT: return "rgba32f";
		default: return "rgba8";
		}
	}

	// WebGPU has 1D textures, but not ones that can be arrays, which is how monotextures pack several
	inline bool texture_supported(GpuQueue* queue, const GpuTextureDesc& desc) {
		auto format = GPU::format2wgpu(desc.format);
		if(desc.type == TEXTURE_1D || format == WGPUTextureFormat_Undefined) return false;
		if((desc.usage & USAGE_STORAGE) && !gpuFormatIsStorableEXT(desc.format)) return false;
		if(desc.sampleCount != 1 && desc.sampleCount != 4) return false;
		if(format == WGPUTextureFormat_Depth32FloatStencil8) return wgpuDeviceHasFeature(queue->device, WGPUFeatureName_Depth32FloatStencil8);
		bool norm16 = desc.format == FORMAT_R16_UNORM || desc.format == FORMAT_R16_SNORM || desc.format == FORMAT_RG16_UNORM
			|| desc.format == FORMAT_RG16_SNORM || desc.format == FORMAT_RGBA16_UNORM || desc.format == FORMAT_RGBA16_SNORM;
		return !norm16 || wgpuDeviceHasFeature(queue->device, WGPUFeatureName_TextureFormatsTier1);
	}
}

GpuTextureSizeAlign gpuTextureSizeAlign(GpuQueue* queue, const GpuTextureDesc& desc) {
	if(!GPU::texture_supported(queue, desc)) {
		errno = WGPUErrorType_Validation;
		return {};
	}
	return {16, 16}; // Waste the minimum amount of space dealing with an allocation
}

namespace GPU {
	// Once per distinct message per queue: callers sit in per draw functions, and repeats would bury the first report.
	inline void report(GpuQueue* queue, GPU_DIAGNOSTIC kind, std::string message) {
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

namespace GPU::detail {
	// An atlas sized for its shape's bucket, the largest texture that can share it, and copyable so growing can move what it holds.
	inline std::pair<WGPUTexture, WGPUTextureView> create_monotexture(GpuQueue* queue, const GpuQueue::TextureHash& shape, uint32_t layers) {
		auto copies = shape.sampleCount == 1 ? WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst : WGPUTextureUsage_None;
		uint32_t mips = shape.sampleCount == 1 ? std::bit_width(std::max(shape.dimensions.x, shape.dimensions.y)) : 1;
		WGPUTextureDescriptor d {
			.usage = GPU::usage2wgpu(shape.usage) | copies,
			.dimension = GPU::texture2wgpu(shape.type),
			.size = {shape.dimensions.x, shape.dimensions.y, shape.type == TEXTURE_3D ? shape.dimensions.z : layers},
			.format = GPU::format2wgpu(shape.format),
			.mipLevelCount = mips,
			.sampleCount = shape.sampleCount,
		};
		auto texture = wgpuDeviceCreateTexture(queue->device, &d);

		// Everything but storage, which needs a single mip view; update_pipeline_layouts makes that one separately.
		auto view_usage = GPU::usage2wgpu(shape.usage) & ~WGPUTextureUsage_StorageBinding;
		WGPUTextureViewDescriptor v {
			.format = d.format,
			.dimension = GPU::texture_view2wgpu(shape.type),
			.baseMipLevel = 0,
			.mipLevelCount = mips,
			.baseArrayLayer = 0,
			.arrayLayerCount = shape.type == TEXTURE_3D ? 1 : layers,
			.aspect = WGPUTextureAspect_All,
			.usage = view_usage,
		};
		// A storage only texture has no other usage to view it through.
		return {texture, view_usage ? wgpuTextureCreateView(texture, &v) : nullptr};
	}
}

GpuTexture* gpuCreateTexture(GpuQueue* queue, const GpuTextureDesc& desc, gpu* memory) {
	// Only one texture can be associated with a memory
	if(queue->gpu2textures.contains(memory) || !queue->allocations.contains(memory)) {
		errno = WGPUErrorType_Validation;
		return nullptr;
	}

	if(!GPU::texture_supported(queue, desc)) {
		GPU::report(queue, GPU_DIAGNOSTIC_UNSUPPORTED, "gpuCreateTexture: this device can't create this texture (see gpuTextureSizeAlign: no 1D textures, storage only in formats gpuFormatIsStorableEXT accepts, 1 or 4 samples, and some formats need optional features), so it was not created");
		errno = WGPUErrorType_Validation;
		return nullptr;
	}

	// Sampled monotextures are all declared filterable float; any other sample type would invalidate the queue's bind groups
	if((desc.usage & USAGE_SAMPLED) && !gpuFormatIsFilterableEXT(desc.format)) {
		GPU::report(queue, GPU_DIAGNOSTIC_UNSUPPORTED, "gpuCreateTexture: WebGPU can only sample textures whose format a sampler can filter, so this texture was not created");
		errno = WGPUErrorType_Validation;
		return nullptr;
	}

	auto hash = GpuQueue::TextureHash::from_descriptor(desc);
	std::unordered_map<GpuQueue::TextureHash, size_t, GpuQueue::TextureHash::Hasher>* monotextures_lookup = nullptr;
	std::vector<std::tuple<uint32_t, WGPUTexture, WGPUTextureView, GpuQueue::TextureHash>>* monotextures = nullptr;
	bool is_storage_texture = desc.usage & USAGE_STORAGE;
	if(is_storage_texture) {
		monotextures = &queue->storage_monotextures;
		monotextures_lookup = &queue->storage_monotextures_lookup;
	} else if(desc.usage & USAGE_SAMPLED) {
		monotextures = &queue->sampled_monotextures;
		monotextures_lookup = &queue->sampled_monotextures_lookup;
	}

	std::optional<GpuQueue::MonotextureRange> range = {};

	if(monotextures != nullptr) {
		if(!monotextures_lookup->contains(hash)) {
			// Reuse a collapsed slot, lowest first, so resizing through many sizes doesn't hold a binding for each.
			auto reused = queue->collapsed_monotextures.end();
			for(auto at = queue->collapsed_monotextures.begin(); at != queue->collapsed_monotextures.end(); ++at)
				if(bool(*at & GpuQueue::MonotextureRange::STORAGE_BIT) == is_storage_texture) {
					reused = at;
					break;
				}

			// Only a list that has to grow spends a binding, so only that has a limit to answer to
			if(reused == queue->collapsed_monotextures.end()) {
				if(is_storage_texture && queue->storage_monotextures.size() >= queue->limits.maxStorageTexturesPerShaderStage) {
					errno = WGPUErrorType_OutOfMemory;
					return nullptr;
				}
				// Group 2 holds sampled monotextures and then sampleable storage ones, so both share this budget; a storage only one doesn't spend from it.
				bool sampleable = !is_storage_texture || ((desc.usage & USAGE_SAMPLED) && gpuFormatIsFilterableEXT(desc.format));
				auto sampled_bindings = queue->sampled_monotextures.size();
				for(auto binding: storage_sampled_bindings(queue))
					if(binding >= 0) ++sampled_bindings;
				if(sampleable && sampled_bindings >= queue->limits.maxSampledTexturesPerShaderStage) {
					errno = WGPUErrorType_OutOfMemory;
					return nullptr;
				}
			}

			assert(desc.type != TEXTURE_3D || desc.layerCount == 1 && "3D textures can't be arrays in WebGPU");

			size_t slot;
			if(reused != queue->collapsed_monotextures.end()) {
				slot = *reused & GpuQueue::MonotextureRange::INDEX_MASK;
				queue->collapsed_monotextures.erase(reused);

				// Released on the same terms as the atlas it replaced: work in flight may still bind it.
				auto [_size, stand_in, stand_in_view, _description] = (*monotextures)[slot];
				queue->code_pending_submission_finished.emplace_back([stand_in, stand_in_view]() {
					if(stand_in_view) wgpuTextureViewRelease(stand_in_view);
					wgpuTextureRelease(stand_in);
				}, queue->next_submission_index);
			} else {
				slot = monotextures->size();
				monotextures->emplace_back();
			}

			(*monotextures_lookup)[hash] = slot;
			auto& [size, texture, full_view, description] = (*monotextures)[slot];
			description = hash;
			size = 0;
			std::tie(texture, full_view) = GPU::detail::create_monotexture(queue, hash, desc.type == TEXTURE_3D ? 1 : desc.layerCount * 2);

			// We have a new monotexture the pipeline layouts should be aware of
			update_pipeline_layouts(queue);
		}

		auto texture_index = monotextures_lookup->at(hash);
		auto& [size, texture, full_view, _description] = monotextures->at(texture_index);

		for(size_t i = 0; i < queue->texture_freelist.size(); ++i) {
			// Strip any empty ranges from the list
			if(queue->texture_freelist[i].start == queue->texture_freelist[i].end) {
				queue->texture_freelist.erase(queue->texture_freelist.begin() + i);
				--i;
				continue;
			}

			// Sorted by monotexture then start, so layers touching or overlapping the hole before them join it, and are tried whole
			if(i > 0) {
				auto prev = queue->texture_freelist[i - 1];
				auto& hole = queue->texture_freelist[i];
				if(prev._index == hole._index && hole.start <= prev.end) {
					hole.start = prev.start;
					hole.end = std::max(hole.end, prev.end);
					queue->texture_freelist.erase(queue->texture_freelist.begin() + (i - 1));
					--i;
				}
			}

			// Only free layers of this exact monotexture atlas will do, and only enough of them
			auto& free = queue->texture_freelist[i];
			if(free.storage() == is_storage_texture && free.index() == texture_index && free.end - free.start >= desc.layerCount) {
				range = {._index = free._index, .start = free.start, .end = free.start + desc.layerCount};
				free.start += desc.layerCount;
				break;
			}
		}

		if(!range) {
			auto capacity = desc.type == TEXTURE_3D ? 1 : wgpuTextureGetDepthOrArrayLayers(texture);
			if(size + desc.layerCount > capacity) {
				auto [grown, grown_view] = GPU::detail::create_monotexture(queue, hash, std::max(size * 2, size + desc.layerCount));

				// What the atlas already holds moves with it, every mip of it; a multisampled one can't be copied
				if(size && desc.sampleCount == 1) {
					auto encoder = wgpuDeviceCreateCommandEncoder(queue->device, nullptr);
					for(uint32_t mip = 0; mip < wgpuTextureGetMipLevelCount(texture); ++mip) {
						WGPUTexelCopyTextureInfo from {.texture = texture, .mipLevel = mip}, to {.texture = grown, .mipLevel = mip};
						WGPUExtent3D extent {std::max(wgpuTextureGetWidth(texture) >> mip, 1u), std::max(wgpuTextureGetHeight(texture) >> mip, 1u), size};
						wgpuCommandEncoderCopyTextureToTexture(encoder, &from, &to, &extent);
					}
					auto commands = wgpuCommandEncoderFinish(encoder, nullptr);
					wgpuQueueSubmit(queue->queue, 1, &commands);
					wgpuCommandBufferRelease(commands);
					wgpuCommandEncoderRelease(encoder);
				}

				// Work in flight may still bind the old atlas
				queue->code_pending_submission_finished.emplace_back([texture, full_view]() {
					if(full_view) wgpuTextureViewRelease(full_view);
					wgpuTextureRelease(texture);
				}, queue->next_submission_index);
				texture = grown;
				full_view = grown_view;

				// Handles made before the move still name the old atlas, and so do the bind groups
				for(auto& [_memory, handle]: queue->gpu2textures)
					if(handle->range && handle->range->storage() == is_storage_texture && handle->range->index() == texture_index)
						handle->texture = texture;
				update_pipeline_layouts(queue);
			}

			range = {.start = size};
			range->set_storage(is_storage_texture);
			range->set_index(texture_index);
			size += desc.layerCount;
			range->end = size; // TODO: Plus 1?
		}
	}

	auto out = (GpuTexture*)queue->cpu_allocator(nullptr, sizeof(GpuTexture));
	new(out) GpuTexture{.descriptor = desc, .range = range};
	if(range) {
		out->texture = std::get<WGPUTexture>((*monotextures)[range->index()]);
		// TODO: Texture view?
	} else { // If we are creating a purely bound texture rather than a lookup texture we are fine
		WGPUTextureDescriptor d {
			.usage = GPU::usage2wgpu(desc.usage),
			.dimension = GPU::texture2wgpu(desc.type),
			.size = {desc.dimensions.x, desc.dimensions.y, desc.type == TEXTURE_3D ? desc.dimensions.z : desc.layerCount},
			.format = GPU::format2wgpu(desc.format),
			.mipLevelCount = desc.mipCount,
			.sampleCount = desc.sampleCount,
		};
		out->texture = wgpuDeviceCreateTexture(queue->device, &d);
		// TODO: Texture view?
	}

	queue->gpu2textures[memory] = out;
	return out;
}

const GpuQueue::MonotextureRange GpuQueue::MonotextureRange::INVALID = {static_cast<uint32_t>(-1), static_cast<uint32_t>(-1), static_cast<uint32_t>(-1)};

GpuTextureDescriptor gpuTextureViewDescriptor(GpuQueue* queue, const GpuTexture* texture, const GpuViewDesc& desc) {
	// Only textures in a monotexture can be described: not ones without USAGE_SAMPLED or USAGE_STORAGE, nor presentable ones. Blit those into a sampled texture.
	assert(texture->range && "This texture has no monotexture slot, so no descriptor can name it");

	GpuTextureDescriptorImpl out {
		.type = static_cast<uint32_t>(texture->descriptor.type),
		.baseMip = desc.baseMip,
		.mipCount = desc.mipCount == ALL_MIPS ? texture->descriptor.mipCount - desc.baseMip : desc.mipCount,
		.width = texture->descriptor.dimensions.x,
		.height = texture->descriptor.dimensions.y,
		.depth = texture->descriptor.dimensions.z,
		.range = texture->range ? *texture->range : GpuQueue::MonotextureRange::INVALID
	};

	// A power of two texture fills its slot, so it gets scale 1, no limit, and REPEAT tiles; others stay half a texel inside.
	const uint32_t image[3] = {
		texture->descriptor.dimensions.x, texture->descriptor.dimensions.y, texture->descriptor.dimensions.z
	};
	const uint32_t slot[3] = {
		std::max(wgpuTextureGetWidth(texture->texture), 1u),
		std::max(wgpuTextureGetHeight(texture->texture), 1u),
		texture->descriptor.type == TEXTURE_3D
			? std::max(wgpuTextureGetDepthOrArrayLayers(texture->texture), 1u) : std::max(image[2], 1u),
	};
	for(int axis = 0; axis < 3; ++axis) {
		out.uvScale[axis] = float(image[axis]) / float(slot[axis]);
		out.uvMax[axis] = image[axis] >= slot[axis]
			? GPU_UV_UNCLAMPED
			: (float(image[axis]) - 0.5f) / float(slot[axis]);
	}

	if(texture->range) {
		auto check = texture->range;
		out.range.start += desc.baseLayer;
		if(desc.layerCount != ALL_LAYERS)
			out.range.end = out.range.start + desc.layerCount;
		assert(out.range.start >= check->start);
		assert(out.range.end <= check->end);
	}
	return (GpuTextureDescriptor&)out;
}

GpuTextureDescriptor gpuRWTextureViewDescriptor(GpuQueue* queue, const GpuTexture* texture, const GpuViewDesc& desc) {
	// A storage binding views the whole atlas at a single mip, so every texture in it is stored at level 0.
	if(desc.baseMip != 0)
		GPU::report(queue, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuRWTextureViewDescriptor: WebGPU binds mip level 0 of a monotexture as its storage texture, so a store through this entry writes level 0 rather than the base mip it names");

	return gpuTextureViewDescriptor(queue, texture, desc);
}

static_assert(std::tuple_size_v<decltype(GpuQueue::monobuffers)> + 1 < (uint64_t(1) << (64 - gpu_address_bits)), "The monobuffer tag has to fit above the offset");
// A gpu pointer reaches WGSL as a vec2<u32>, so the shader needs the address bits of the high word and where the tag starts.
constexpr static uint32_t gpu_address_max_hi = static_cast<uint32_t>(gpu_address_max >> 32);

constexpr static uint32_t gpu_address_tag_shift = gpu_address_bits - 32;

// Regenerated whenever the monobuffer, monotexture or sampler counts change, since WGSL can only reach those through switches. Bindings must match the layouts above.
std::string generate_backend_module(GpuQueue* queue, bool compute) {
	// Must match GPU::texture_view2wgpu, which only hands out 2D array, cube array and 3D views, since monotextures pack several textures.
	constexpr static auto sampled_type = [](TEXTURE type) {
		switch (GPU::texture_view2wgpu(type)) {
		case WGPUTextureViewDimension_3D: return "Texture3D";
		case WGPUTextureViewDimension_CubeArray: return "TextureCubeArray";
		default: return "Texture2DArray";
		}
	};
	// Write only (Slang's W prefix) to match the bind group layout; see update_pipeline_layouts.
	constexpr static auto storage_type = [](TEXTURE type) {
		switch (GPU::texture_view2wgpu(type)) {
		case WGPUTextureViewDimension_3D: return "WTexture3D";
		default: return "WTexture2DArray";
		}
	};

	// A store takes a texel position, not a normalized coordinate, and WGSL has no storage cube textures.
	constexpr static auto store_coordinate = [](TEXTURE type) {
		switch (GPU::texture_view2wgpu(type)) {
		case WGPUTextureViewDimension_3D: return "texel";
		default: return "uint3(texel.xy, layer)";
		}
	};

	// Coordinates are scaled onto the texture's share of the atlas and pinned half a texel inside it (see GpuTextureDescriptorImpl).
	constexpr static auto sample_remap = [](TEXTURE type) {
		switch (GPU::texture_view2wgpu(type)) {
		case WGPUTextureViewDimension_3D:
			return "\t\tlet at = min(uv * scale, limit);\n";
		case WGPUTextureViewDimension_CubeArray:
			return "\t\tlet at = uv;\n";
		default:
			return "\t\tlet at = min(uv.xy * scale.xy, limit.xy);\n";
		}
	};

	// The coordinate SampleLevel is actually given. The dimensions that have no array index simply drop it.
	constexpr static auto sample_coordinate = [](TEXTURE type) {
		switch (GPU::texture_view2wgpu(type)) {
		case WGPUTextureViewDimension_2DArray: return "float3(at, float(layer))";
		case WGPUTextureViewDimension_CubeArray: return "float4(at, float(layer))";
		default: return "at";
		}
	};

	std::string out = "module noapi_backend;\n\n";

	// The rasterizer only gets the monobuffers read only, matching create_buffer_bind_group_layout.
	uint32_t binding = 0;
	for(uint32_t i = 0; i < queue->monobuffers.size(); ++i)
		out += std::format("[[vk::binding({}, 0)]] public {}<uint> mono{};\n",
			binding++, compute ? "RWStructuredBuffer" : "StructuredBuffer", i);

	out += std::format("\n"
		"[[vk::binding({}, 0)]] public StructuredBuffer<uint> texture_heap;\n"
		"[[vk::binding({}, 0)]] public StructuredBuffer<uint> gpu_sampler_map;\n\n", binding, binding + 1);
	binding += 2;

	out += std::string(compute
		? "public struct GpuShaderData {\n\tpublic uint2 compute;\n}\n"
		: "public struct GpuShaderData {\n\tpublic uint2 vertex;\n\tpublic uint2 fragment;\n\tpublic uint2 index;\n}\n")
		+ std::format("[[vk::binding({}, 0)]] public ConstantBuffer<GpuShaderData> shader_data;\n\n", binding++);

	// Pointers: an offset tagged with its monobuffer, reached by a switch; the portable module builds every wider access from this one word load.
	std::string load_cases, store_cases;
	for(uint32_t i = 1; i < queue->monobuffers.size(); ++i) {
		load_cases += std::format("\tcase {}u: return mono{}[index];\n", i, i);
		store_cases += std::format("\tcase {}u: mono{}[index] = value; break;\n", i, i);
	}

	out += std::format(
		"static const uint GPU_ADDRESS_MAX_HI = {:#x}u; // The bits of the high word that are still address\n"
		"static const uint GPU_ADDRESS_TAG_SHIFT = {}u; // Where the monobuffer tag starts in that word\n"
		"\n"
		"public uint2 gpuBackendEncodeAddress(uint monobuffer, uint2 address) {{\n"
		"\tlet tag = monobuffer + 1u; // Zero is left meaning \"no pointer\"\n"
		"\treturn uint2(address.x, (address.y & GPU_ADDRESS_MAX_HI) | (tag << GPU_ADDRESS_TAG_SHIFT));\n"
		"}}\n"
		"\n"
		"// Which monobuffer an address names. The tag is the index plus one, so that a zeroed\n"
		"// pointer (tag 0) is \"no pointer\" rather than a valid address into the first one; it\n"
		"// wraps to a tag no case matches and lands on monobuffer 0, which is the empty buffer.\n"
		"public uint gpuBackendMonobuffer(uint2 address) {{\n"
		"\treturn (address.y >> GPU_ADDRESS_TAG_SHIFT) - 1u;\n"
		"}}\n"
		"\n"
		"// The load half of the memory ABI: one word at the address the portable module hands\n"
		"// over as the 64 bits of the pointer, low word first. Everything wider than a word --\n"
		"// pointers, vectors, matrices, a shader's own structs -- is built out of this one call\n"
		"// by the portable module, so this switch is the whole of what the emulation costs.\n"
		"public uint gpuBackendLoadU32(uint2 address) {{\n"
		"\tlet index = address.x / 4u;\n"
		"\tswitch(gpuBackendMonobuffer(address)) {{\n"
		"{}"
		"\tdefault: return mono0[index];\n"
		"\t}}\n"
		"}}\n"
		"\n", gpu_address_max_hi, gpu_address_tag_shift, load_cases);

	// Compute only, since the rasterizer gets the monobuffers read only; GPU_STORES makes a graphics store a compile error.
	if(compute)
		out += std::format(
			"public void gpuBackendStoreU32(uint2 address, uint value) {{\n"
			"\tlet index = address.x / 4u;\n"
			"\tswitch(gpuBackendMonobuffer(address)) {{\n"
			"{}"
			"\tdefault: mono0[index] = value; break;\n"
			"\t}}\n"
			"}}\n"
			"\n", store_cases);

	// Root data
	out += compute
		? "public uint2 gpuBackendRootCompute() { return shader_data.compute; }\n\n"
		: "public uint2 gpuBackendRootVertex() { return shader_data.vertex; }\n"
			"public uint2 gpuBackendRootFragment() { return shader_data.fragment; }\n"
			"public uint2 gpuBackendRootIndex() { return shader_data.index; }\n\n";

	// Group 1 is compute only; a graphics layout binds an empty group there (see update_pipeline_layouts).
	binding = 0;
	if(compute)
		for (auto const& [_cap, texture, view, desc] : queue->storage_monotextures) {
			out += std::format("[[vk::binding({}, 1)]] [format(\"{}\")] public {}<float4> storage_tex_{};\n",
				binding, GPU::storage_format_name(desc.format), storage_type(desc.type), binding);
			++binding;
		}

	binding = 0;
	for (auto const& [_cap, texture, view, desc] : queue->sampled_monotextures) {
		out += std::format("[[vk::binding({}, 2)]] public {}<float4> sampled_tex_{};\n",
			binding, sampled_type(desc.type), binding);
		++binding;
	}

	// Sampleable storage monotextures again, numbered as storage_sampled_bindings does; graphics only, since only its group 2 has them.
	auto storage_sampled = storage_sampled_bindings(queue);
	if(compute) storage_sampled.assign(queue->storage_monotextures.size(), -1);
	for (size_t i = 0; i < queue->storage_monotextures.size(); ++i) {
		if(storage_sampled[i] < 0) continue;
		auto const& [_cap, texture, view, desc] = queue->storage_monotextures[i];
		out += std::format("[[vk::binding({}, 2)]] public {}<float4> sampled_tex_{};\n",
			storage_sampled[i], sampled_type(desc.type), storage_sampled[i]);
	}

	// Samplers, and the metadata buffer that makes them addressable
	out += "\n";
	for (uint32_t i = 0; i < gpu_sampler_slot_count; ++i)
		out += std::format("[[vk::binding({}, 3)]] public SamplerState gpu_sampler_{};\n", i, i);

	auto storage_count = static_cast<uint32_t>(queue->storage_monotextures.size());
	auto sampled_count = static_cast<uint32_t>(queue->sampled_monotextures.size());

	out += std::format("\n"
		"public static const uint GPU_SAMPLER_SLOT_COUNT = {}u;\n"
		"public static const uint GPU_SAMPLER_LOOKUP_SIZE = {}u;\n"
		"public static const uint GPU_STORAGE_TEXTURE_COUNT = {}u;\n"
		"public static const uint GPU_SAMPLED_TEXTURE_COUNT = {}u;\n"
		"\n"
		"// Turns a packed GpuSamplerDesc into the slot holding it, or slot 0 (the default\n"
		"// sampler) when that description was never handed to gpuSetEnabledSamplersEXT\n"
		"public uint gpuBackendSamplerSlot(uint packed) {{\n"
		"\tif(packed >= GPU_SAMPLER_LOOKUP_SIZE) return 0u;\n"
		"\treturn gpu_sampler_map[packed];\n"
		"}}\n"
		"\n"
		"static const uint GPU_MONOTEXTURE_STORAGE_BIT = 0x80000000u;\n"
		"static const uint GPU_MONOTEXTURE_INDEX_MASK = 0x7FFFFFFFu;\n"
		"\n"
		"// One entry of the texture heap, as gpuTextureViewDescriptor lays it out: sixteen words,\n"
		"// one per field. `size` is the texture's own rather than the monotexture's, and `uv_scale`\n"
		"// and `uv_max` are its share of that monotexture and the coordinate a bilinear tap has to\n"
		"// stay inside of -- WGSL can report the monotexture's own size, but not how much of it this\n"
		"// texture owns, so both are worked out on the host and read back here.\n"
		"public struct GpuTextureDescriptor {{\n"
		"\tpublic uint texture_type;\n"
		"\tpublic uint base_mip;\n"
		"\tpublic uint mip_count;\n"
		"\tpublic uint3 size; // width, height, depth\n"
		"\tpublic uint monotexture; // Top bit set means a storage monotexture, the rest is the index\n"
		"\tpublic uint first_layer; // Absolute layer within that monotexture\n"
		"\tpublic uint end_layer;\n"
		"\tpublic float3 uv_scale;\n"
		"\tpublic float3 uv_max;\n"
		"}}\n"
		"\n"
		"public GpuTextureDescriptor gpuBackendTextureDescriptor(uint heap_index) {{\n"
		"\tlet at = heap_index * 16u;\n"
		"\treturn GpuTextureDescriptor(texture_heap[at], texture_heap[at + 1u], texture_heap[at + 2u],\n"
		"\t\tuint3(texture_heap[at + 4u], texture_heap[at + 5u], texture_heap[at + 6u]),\n"
		"\t\ttexture_heap[at + 7u], texture_heap[at + 8u], texture_heap[at + 9u],\n"
		"\t\tfloat3(asfloat(texture_heap[at + 10u]), asfloat(texture_heap[at + 11u]), asfloat(texture_heap[at + 12u])),\n"
		"\t\tfloat3(asfloat(texture_heap[at + 13u]), asfloat(texture_heap[at + 14u]), asfloat(texture_heap[at + 15u])));\n"
		"}}\n"
		"\n"
		"public uint gpuBackendMonotextureIndex(GpuTextureDescriptor descriptor) {{\n"
		"\treturn descriptor.monotexture & GPU_MONOTEXTURE_INDEX_MASK;\n"
		"}}\n"
		"\n"
		"public bool gpuBackendMonotextureIsStorage(GpuTextureDescriptor descriptor) {{\n"
		"\treturn (descriptor.monotexture & GPU_MONOTEXTURE_STORAGE_BIT) != 0u;\n"
		"}}\n"
		"\n",
		gpu_sampler_slot_count, GPU::detail::sampler_lookup_size, storage_count, sampled_count);

	// WGSL can't index arrays of samplers or textures, so picking either at runtime is a switch over every combination.
	auto sample_function = [&](int at, TEXTURE type) {
		auto coordinate = sample_coordinate(type);
		auto body = std::format("float4 gpu_sample_tex_{}(uint slot, float3 scale, float3 limit, float3 uv, uint layer, float mip) {{\n{}"
			"\tswitch(slot) {{\n", at, sample_remap(type));
		for (uint32_t i = 1; i < gpu_sampler_slot_count; ++i)
			body += std::format("\tcase {}u: return sampled_tex_{}.SampleLevel(gpu_sampler_{}, {}, mip);\n",
				i, at, i, coordinate);
		return body + std::format("\tdefault: return sampled_tex_{}.SampleLevel(gpu_sampler_0, {}, mip);\n"
			"\t}}\n"
			"}}\n"
			"\n", at, coordinate);
	};

	binding = 0;
	for (auto const& [_cap, texture, view, desc] : queue->sampled_monotextures)
		out += sample_function(binding++, desc.type);

	for (size_t i = 0; i < queue->storage_monotextures.size(); ++i) {
		if(storage_sampled[i] < 0) continue;
		out += sample_function(storage_sampled[i], std::get<3>(queue->storage_monotextures[i]).type);
	}

	out += "// Samples monotexture `texture` on `layer`, through the sampler in `slot` (see\n"
		"// gpuBackendSamplerSlot). `uv` runs 0..1 over the texture itself and is scaled by `scale`\n"
		"// onto whatever part of the monotexture that texture occupies, then held to `limit` so it\n"
		"// stays inside it. `layer` and `mip` are absolute. `storage` picks which of the two lists\n"
		"// of monotextures `texture` indexes, since the heap entry's index is relative to its own.\n"
		"public float4 gpuBackendSampleTexture(uint texture, bool storage, uint slot, float3 scale, float3 limit, float3 uv, uint layer, float mip) {\n"
		"\tif(storage) {\n"
		"\t\tswitch(texture) {\n";
	for (size_t i = 0; i < queue->storage_monotextures.size(); ++i)
		if(storage_sampled[i] >= 0)
			out += std::format("\t\tcase {}u: return gpu_sample_tex_{}(slot, scale, limit, uv, layer, mip);\n", i, storage_sampled[i]);
	out += "\t\t// A storage monotexture that could not be bound as a sampled one as well: either\n"
		"\t\t// nothing that landed in it asked for USAGE_SAMPLED, or its format is one no sampler\n"
		"\t\t// will filter. Reads black rather than failing to compile.\n"
		"\t\tdefault: return float4(0.0);\n"
		"\t\t}\n"
		"\t}\n"
		"\tswitch(texture) {\n";
	for (uint32_t i = 0; i < sampled_count; ++i)
		out += std::format("\tcase {}u: return gpu_sample_tex_{}(slot, scale, limit, uv, layer, mip);\n", i, i);
	out += "\tdefault: return float4(0.0);\n"
		"\t}\n"
		"}\n"
		"\n"
		"// Samples a texture straight out of the heap. `uv` runs 0..1 over the texture, and\n"
		"// `layer` and `mip` are relative to the view the descriptor describes rather than to the\n"
		"// monotexture.\n"
		"public float4 gpuBackendSample(uint heap_index, uint slot, float3 uv, uint layer, float mip) {\n"
		"\tlet descriptor = gpuBackendTextureDescriptor(heap_index);\n"
		"\tlet level = float(descriptor.base_mip) + clamp(mip, 0.0, max(float(descriptor.mip_count), 1.0) - 1.0);\n"
		"\tlet at = descriptor.first_layer + min(layer, descriptor.end_layer - descriptor.first_layer - 1u);\n"
		"\treturn gpuBackendSampleTexture(gpuBackendMonotextureIndex(descriptor),\n"
		"\t\tgpuBackendMonotextureIsStorage(descriptor), slot,\n"
		"\t\tdescriptor.uv_scale, descriptor.uv_max, uv, at, level);\n"
		"}\n"
		"\n";

	// Stores
	if(compute) out += "// Stores one texel of monotexture `texture` on absolute layer `layer`. Only mip level 0\n"
		"// of a monotexture is bound as a storage texture, so unlike the sample above this takes no\n"
		"// level; see update_pipeline_layouts.\n"
		"void gpuBackendStoreTextureAt(uint texture, uint3 texel, uint layer, float4 value) {\n"
		"\tswitch(texture) {\n";
	if(compute) {
		binding = 0;
		for (auto const& [_cap, texture, view, desc] : queue->storage_monotextures) {
			out += std::format("\tcase {}u: storage_tex_{}.Store({}, value); break;\n",
				binding, binding, store_coordinate(desc.type));
			++binding;
		}
		out += "\tdefault: break;\n"
			"\t}\n"
			"}\n"
			"\n"
			"// Stores through a heap entry. `texel` is a texel of the texture itself, so a coordinate\n"
			"// past its own size is dropped rather than allowed to reach the rest of the monotexture\n"
			"// it shares, and `layer` is relative to the entry the way a sample's is.\n"
			"public void gpuBackendStoreTexture(uint heap_index, uint3 texel, uint layer, float4 value) {\n"
			"\tlet descriptor = gpuBackendTextureDescriptor(heap_index);\n"
			"\t// A sampled entry names no storage binding, so there is nothing to store through\n"
			"\tif(!gpuBackendMonotextureIsStorage(descriptor)) return;\n"
			"\tif(any(texel >= max(descriptor.size, uint3(1u)))) return;\n"
			"\tlet at = descriptor.first_layer + min(layer, descriptor.end_layer - descriptor.first_layer - 1u);\n"
			"\tgpuBackendStoreTextureAt(gpuBackendMonotextureIndex(descriptor), texel, at, value);\n"
			"}\n";
	}

	return out;
}

namespace GPU::detail {
	// Slang has no WGSL spelling for a second blend source, and emits SV_Target1 as @location(1); dual source blending wants both outputs at location 0
	inline std::string dual_source_outputs(std::string_view wgsl) {
		auto arrow = wgsl.find("->", wgsl.find("@fragment"));
		auto name_start = wgsl.find_first_not_of(' ', arrow + 2);
		auto name = std::string(wgsl.substr(name_start, wgsl.find_first_of(" \n{", name_start) - name_start));
		auto start = wgsl.find("struct " + name);
		auto end = wgsl.find('}', start);
		assert(start != std::string_view::npos && "A dual source fragment shader returns a struct with SV_Target0 and SV_Target1");

		auto outputs = std::string(wgsl.substr(start, end - start));
		for(auto [from, to]: {std::pair{"@location(0) ", "@location(0) @blend_src(0) "}, {"@location(1) ", "@location(0) @blend_src(1) "}})
			if(auto at = outputs.find(from); at != std::string::npos) outputs.replace(at, strlen(from), to);
		return "enable dual_source_blending;\n" + std::string(wgsl.substr(0, start)) + outputs + std::string(wgsl.substr(end));
	}
}

WGPUShaderModule create_shader_module(GpuQueue* queue, std::string_view IR, GPU::shaders::SHADER_STAGE stage, bool dual_source = false) {
	auto compute = stage == GPU::shaders::SHADER_STAGE::COMPUTE;

	std::string error;
	auto wgsl = GPU::shaders::compile(SLANG_WGSL, generate_backend_module(queue, compute), IR, stage, {
		{"GPU_COMPUTE", compute ? "1" : "0"},
		{"GPU_GRAPHICS", compute ? "0" : "1"},
		// The rasterizer only gets the monobuffers read only (see create_buffer_bind_group_layout), so the portable module only offers stores to a compute shader
		{"GPU_STORES", compute ? "1" : "0"},
		// Storage textures are only bound for compute passes; see update_pipeline_layouts.
		{"GPU_TEXTURE_STORES", compute ? "1" : "0"},
	}, error);

	if(!wgsl) {
		GPU::shaders::report_diagnostic(error);
		errno = WGPUErrorType_Validation;
		return nullptr;
	}

	if(dual_source) wgsl = GPU::detail::dual_source_outputs(*wgsl);
	WGPUShaderSourceWGSL source {
		.chain {.sType = WGPUSType_ShaderSourceWGSL },
		.code = {wgsl->data(), wgsl->size()}
	};
	WGPUShaderModuleDescriptor module {
		.nextInChain = &source.chain
	};
	return wgpuDeviceCreateShaderModule(queue->device, &module);
}

bool update_compute_pipeline(GpuQueue* queue, const GpuPipeline* pipeline) {
	auto& cache = std::get<GpuPipeline::ComputeCache>(pipeline->cache);
	if(cache.pipeline) wgpuComputePipelineRelease(cache.pipeline);
	cache.pipeline = nullptr;

	auto module = create_shader_module(queue, cache.IR, GPU::shaders::SHADER_STAGE::COMPUTE);
	if(!module) return false;

	WGPUComputePipelineDescriptor d {
		.layout = queue->current_compute_pipeline_layout,
		.compute = {
			.module = module,
			// Leaving the entry point undefined picks the module's only one, which is all Slang emitted for this stage
			.entryPoint = WGPU_STRING_VIEW_INIT,
		}
	};
	cache.pipeline = wgpuDeviceCreateComputePipeline(queue->device, &d);
	pipeline->reference_layout = queue->current_compute_pipeline_layout;

	wgpuShaderModuleRelease(module);
	return cache.pipeline;
}

GpuPipeline* gpuCreateComputePipeline(GpuQueue* queue, GpuByteSpan computeIR) {
	auto out = (GpuPipeline*)queue->cpu_allocator(nullptr, sizeof(GpuPipeline));
	new(out) GpuPipeline{
		.reference_layout = nullptr,
		.cache = GpuPipeline::ComputeCache {
			.IR = {(char*)computeIR.data(), (char*)(computeIR.data() + computeIR.size())},
		}
	};

	if(!update_compute_pipeline(queue, out)) {
		gpuFreePipeline(queue, out);
		return nullptr;
	}
	return out;
}

void gpuFreePipeline(GpuQueue* queue, GpuPipeline* pipeline) {
	if(std::holds_alternative<GpuPipeline::ComputeCache>(pipeline->cache)) {
		auto& cache = std::get<GpuPipeline::ComputeCache>(pipeline->cache);
		if(cache.pipeline) wgpuComputePipelineRelease(cache.pipeline);
	} else {
		auto& cache = std::get<GpuPipeline::RenderCache>(pipeline->cache);
		if(cache.pipeline) wgpuRenderPipelineRelease(cache.pipeline);
	}

	pipeline->~GpuPipeline();
	queue->cpu_allocator(pipeline, 0);
}

namespace GPU {
	// Reads the index gpuSubmitNoFree's completion callback publishes, not the semaphore: a blocking readback in the browser would yield and expire the canvas texture.
	inline void process_pending_code(GpuQueue* queue) {
#ifndef __EMSCRIPTEN__
		wgpuDeviceTick(queue->device);
#endif
		auto current_finished_submission = queue->last_finished_submission;
		if(queue->code_pending_submission_finished.size())
			for(size_t i = queue->code_pending_submission_finished.size(); i--; ) {
				auto& [code, submit] = queue->code_pending_submission_finished[i];
				if(submit <= current_finished_submission) {
					code();
					queue->code_pending_submission_finished.erase(queue->code_pending_submission_finished.begin() + i);
				}
			}
	}
}

GpuCommandBuffer* gpuStartCommandRecording(GpuQueue* queue) {
	auto out = (GpuCommandBuffer*)queue->cpu_allocator(nullptr, sizeof(GpuCommandBuffer));
	new(out) GpuCommandBuffer{
		.queue = queue,
		.encoder = wgpuDeviceCreateCommandEncoder(queue->device, nullptr)
	};

	GPU::process_pending_code(queue);

	return out;
}

void gpuFreeCommandBuffer(GpuCommandBuffer* cmd) {
	wgpuCommandEncoderRelease(cmd->encoder);

	auto allocator = cmd->queue->cpu_allocator;
	cmd->~GpuCommandBuffer();
	allocator(cmd, 0);
}

uint64_t gpuSubmit(GpuQueue* queue, GpuCommandBufferSpan command_buffers, GpuSemaphore* semaphore, uint64_t signal_value) {
	auto submission = gpuSubmitNoFreeEXT(queue, command_buffers, semaphore, signal_value);
	for(auto cmd: command_buffers)
		gpuFreeCommandBuffer(cmd);
	return submission;
}

namespace GPU {
	inline void semaphore_gpu_increment(GpuCommandBuffer* cmd, GpuSemaphore sema) {
		ensure_semaphore_pipelines(cmd->queue);

		WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(cmd->encoder, nullptr);
		wgpuComputePassEncoderSetPipeline(pass, cmd->queue->semaphore_increment_pipeline);
		wgpuComputePassEncoderSetBindGroup(pass, 0, sema.bind_group, 0, nullptr);
		wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
		wgpuComputePassEncoderEnd(pass);
		wgpuComputePassEncoderRelease(pass);
	}

	inline void semaphore_gpu_set_max(GpuCommandBuffer* cmd, GpuSemaphore sema, uint64_t value = 1) {
		wgpuQueueWriteBuffer(cmd->queue->queue, sema.upload_buffer, 0, &value, sizeof(value));

		ensure_semaphore_pipelines(cmd->queue);

		WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(cmd->encoder, nullptr);
		wgpuComputePassEncoderSetPipeline(pass, cmd->queue->semaphore_set_max_pipeline);
		wgpuComputePassEncoderSetBindGroup(pass, 0, sema.bind_group, 0, nullptr);
		wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
		wgpuComputePassEncoderEnd(pass);
		wgpuComputePassEncoderRelease(pass);
	}
}

void endRenderPass(GpuCommandBuffer* cmd) {
	if(!cmd->render_pass) return;

	wgpuRenderPassEncoderEnd(cmd->render_pass);
	cmd->queue->code_pending_submission_finished.emplace_back([pass = cmd->render_pass] {
		wgpuRenderPassEncoderRelease(pass);
	}, cmd->queue->next_submission_index);
	cmd->render_pass = nullptr;
	cmd->bound_render_pipeline = nullptr; // Nothing set on a pass outlives it
}

void endComputePass(GpuCommandBuffer* cmd) {
	if(!cmd->compute_pass) return;

	wgpuComputePassEncoderEnd(cmd->compute_pass);
	cmd->queue->code_pending_submission_finished.emplace_back([pass = cmd->compute_pass] {
		wgpuComputePassEncoderRelease(pass);
	}, cmd->queue->next_submission_index);
	cmd->compute_pass = nullptr;
}

void endCurrentPass(GpuCommandBuffer* cmd) {
	endRenderPass(cmd);
	endComputePass(cmd);
}

uint64_t gpuSubmitNoFreeEXT(GpuQueue* queue, GpuCommandBufferSpan command_buffers, GpuSemaphore* semaphore, uint64_t signal_value) {
	std::vector<WGPUCommandBuffer> buffers; buffers.reserve(command_buffers.size());
	for(auto buffer: command_buffers) {
		endCurrentPass(buffer);
		buffers.emplace_back(wgpuCommandEncoderFinish(buffer->encoder, nullptr));
	}

	GpuCommandBuffer cmd {
		.queue = queue,
		.encoder = wgpuDeviceCreateCommandEncoder(queue->device, nullptr),
	};
	GPU::semaphore_gpu_increment(&cmd, queue->current_submission_timeline_semaphore);
	if(semaphore)
		GPU::semaphore_gpu_set_max(&cmd, *semaphore, signal_value);

	auto signal = wgpuCommandEncoderFinish(cmd.encoder, nullptr);
	wgpuCommandEncoderRelease(cmd.encoder);

	// Separate submits, still in queue order: an invalid command buffer fails its whole submit, and taking the signal with it would hang every later wait
	wgpuQueueSubmit(queue->queue, buffers.size(), buffers.data());
	wgpuQueueSubmit(queue->queue, 1, &signal);
	for(auto buffer: buffers)
		wgpuCommandBufferRelease(buffer);
	wgpuCommandBufferRelease(signal);

	// Lets process_pending_code reclaim without reading the semaphore back; the index rides in userdata, which is 64 bits on every target (see gpu_address_max).
	wgpuQueueOnSubmittedWorkDone(
		queue->queue,
		WGPUQueueWorkDoneCallbackInfo {
			.mode = WGPUCallbackMode_AllowSpontaneous,
#ifdef __EMSCRIPTEN__
			.callback = [](WGPUQueueWorkDoneStatus status, WGPU_NULLABLE void* userdata1, WGPU_NULLABLE void* userdata2) {
#else
			.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView message, WGPU_NULLABLE void* userdata1, WGPU_NULLABLE void* userdata2) {
#endif
				auto queue = static_cast<GpuQueue*>(userdata1);
				auto finished = reinterpret_cast<size_t>(userdata2);
				// Completion callbacks can arrive out of order, and an older one must not walk the mark backwards over deletes a newer one already released
				if(finished > queue->last_finished_submission)
					queue->last_finished_submission = finished;
			},
			.userdata1 = queue,
			.userdata2 = reinterpret_cast<void*>(queue->next_submission_index)
		}
	);

	for(auto buffer: command_buffers)
		for(auto code: buffer->code_pending_submission_finished)
			queue->code_pending_submission_finished.emplace_back(code, queue->next_submission_index);

	return queue->next_submission_index++;
}

GpuSemaphore* gpuCreateSemaphore(GpuQueue* queue, uint64_t initial_value) {
	auto out = (GpuSemaphore*)queue->cpu_allocator(nullptr, sizeof(GpuSemaphore));

	*out = GPU::semaphore_initialize(queue, initial_value);
	return out;
}

namespace GPU {
	inline void wait_for_buffer_map(GpuQueue* queue, WGPUBuffer buffer, WGPUMapMode mode, uint64_t offset, uint64_t size) {
		struct Wait { volatile bool done = false; };
		Wait wait;

		WGPUBufferMapCallbackInfo callback{
			.mode = WGPUCallbackMode_AllowSpontaneous,
			.callback = [](WGPUMapAsyncStatus _status, WGPUStringView _message, void* userdata1, void* _userdata2) {
				static_cast<Wait*>(userdata1)->done = true;
			},
			.userdata1 = &wait,
		};
		wgpuBufferMapAsync(buffer, mode, offset, size, callback);

		while (!wait.done) {
		#ifdef __EMSCRIPTEN__
			emscripten_sleep(1); // yields back to the browser event loop
		#else
			wgpuDeviceTick(queue->device);
		#endif
		}
	}

	inline uint64_t semaphore_value(GpuQueue* queue, GpuSemaphore sema) {
		WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(queue->device, nullptr);
		wgpuCommandEncoderCopyBufferToBuffer(encoder, sema.buffer, 0, sema.readback_buffer, 0, sizeof(uint64_t));

		WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
		wgpuCommandEncoderRelease(encoder);

		wgpuQueueSubmit(queue->queue, 1, &cmd);
		wgpuCommandBufferRelease(cmd);

		wait_for_buffer_map(queue, sema.readback_buffer, WGPUMapMode_Read, 0, sizeof(uint64_t));

		const void* mapped = wgpuBufferGetConstMappedRange(sema.readback_buffer, 0, sizeof(uint64_t));
		uint64_t result = 0;
		std::memcpy(&result, mapped, sizeof(uint64_t));
		wgpuBufferUnmap(sema.readback_buffer);

		return result;
	}

	inline bool semaphore_wait(GpuQueue* queue, GpuSemaphore sema, uint64_t target_value, std::chrono::nanoseconds timeout = std::chrono::nanoseconds::max(), std::chrono::nanoseconds poll_interval = std::chrono::milliseconds(1)) {
		const bool infinite = (timeout == std::chrono::nanoseconds::max());
		const auto deadline = infinite ? std::chrono::steady_clock::time_point::max() : std::chrono::steady_clock::now() + timeout;

		while (true) {
			if (semaphore_value(queue, sema) >= target_value)
				return true;
			if (!infinite && std::chrono::steady_clock::now() >= deadline)
				return false;
#ifdef __EMSCRIPTEN__
			emscripten_sleep(std::chrono::duration_cast<std::chrono::milliseconds>(poll_interval).count());
#else
			std::this_thread::sleep_for(poll_interval);
#endif
		}
	}
}

uint64_t gpuWaitSemaphore(GpuQueue* queue, const GpuSemaphore* semaphore, uint64_t value, uint64_t timeout) {
	if(value == GPU_GET_VALUE)
		return GPU::semaphore_value(queue, *semaphore);

	// UINT64_MAX overflows a signed nanosecond count, so forever is spelled the way semaphore_wait expects
	auto wait = timeout >= (uint64_t)std::chrono::nanoseconds::max().count() ? std::chrono::nanoseconds::max() : std::chrono::nanoseconds(timeout);
	bool reached = GPU::semaphore_wait(queue, *semaphore, value, wait);
	GPU::process_pending_code(queue);
	return reached ? value : 0;
}

void gpuFreeSemaphore(GpuQueue* queue, GpuSemaphore* semaphore) {
	GPU::semaphore_destroy(*semaphore);
	queue->cpu_allocator(semaphore, 0);
}

namespace GPU::detail {
	// Members ordered for the `auto [range, offset, base]` call sites; base is what allocations and gpu2textures are keyed by.
	struct BufferLocation {
		GpuQueue::MonobufferRange range = {};
		ptrdiff_t offset = 0;
		gpu* base = nullptr; // Null when the address landed outside every allocation
	};

	// Addresses order as (monobuffer, offset), so upper_bound lands one past the holding allocation; an unowned address returns null rather than inserting.
	inline BufferLocation closest_buffer(GpuQueue* queue, gpu* addr) {
		auto after = queue->allocations.upper_bound(addr);
		// Below every allocation, or past the end of the nearest one below it
		bool inside = after != queue->allocations.begin()
			&& (uintptr_t)addr - (uintptr_t)std::prev(after)->first < std::get<GpuQueue::MonobufferRange>(std::prev(after)->second).size();
		assert(inside && "The address doesn't lie inside any allocation");
		if(!inside) return {};

		auto found = std::prev(after);
		return {std::get<GpuQueue::MonobufferRange>(found->second), ptrdiff_t((uintptr_t)addr - (uintptr_t)found->first), found->first};
	}

	inline std::array<BufferLocation, 2> closest_buffer(GpuQueue* queue, gpu* addrA, gpu* addrB) {
		return {closest_buffer(queue, addrA), closest_buffer(queue, addrB)};
	}
}

void gpuMemCpy(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, size_t bytes) {
	endCurrentPass(cmd); // A copy is not something a pass can hold, and a dispatch may have opened one
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_);
	auto [dest_range, dest_offset, dest_addr] = dest; auto [src_range, src_offset, src_addr] = src;

	if(src_range.buffer == dest_range.buffer) {
		// We can't copy from a buffer to itself for some reason... so we need to create a temporary transfer buffer.
		WGPUBufferDescriptor d {
			.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
			.size = bytes
		};
		auto tmp = wgpuDeviceCreateBuffer(cmd->queue->device, &d);

		wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, cmd->queue->monobuffers[src_range.buffer], src_range.start + src_offset, tmp, 0, bytes);
		wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, tmp, 0, cmd->queue->monobuffers[dest_range.buffer], dest_range.start + dest_offset, bytes);

		cmd->queue->code_pending_submission_finished.emplace_back([tmp](){ wgpuBufferRelease(tmp); }, cmd->queue->next_submission_index);
	} else wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, cmd->queue->monobuffers[src_range.buffer], src_range.start + src_offset, cmd->queue->monobuffers[dest_range.buffer], dest_range.start + dest_offset, bytes);
}

namespace GPU::detail {
	// Bytes per texel for the uncompressed, single plane formats the API exposes
	inline uint32_t format_bytes(FORMAT format) {
		switch (format) {
		case FORMAT_R8_UNORM:
		case FORMAT_R8_SNORM:
		case FORMAT_R8_UINT:
		case FORMAT_R8_SINT:
		case FORMAT_S8_UINT: return 1;

		case FORMAT_R16_UNORM:
		case FORMAT_R16_SNORM:
		case FORMAT_R16_UINT:
		case FORMAT_R16_SINT:
		case FORMAT_R16_FLOAT:
		case FORMAT_RG8_UNORM:
		case FORMAT_RG8_SNORM:
		case FORMAT_RG8_UINT:
		case FORMAT_RG8_SINT:
		case FORMAT_D16_UNORM: return 2;

		case FORMAT_R32_UINT:
		case FORMAT_R32_SINT:
		case FORMAT_R32_FLOAT:
		case FORMAT_RG16_UNORM:
		case FORMAT_RG16_SNORM:
		case FORMAT_RG16_UINT:
		case FORMAT_RG16_SINT:
		case FORMAT_RG16_FLOAT:
		case FORMAT_RGBA8_UNORM:
		case FORMAT_RGBA8_SRGB:
		case FORMAT_RGBA8_SNORM:
		case FORMAT_RGBA8_UINT:
		case FORMAT_RGBA8_SINT:
		case FORMAT_BGRA8_UNORM:
		case FORMAT_BGRA8_SRGB:
		case FORMAT_RGB10_A2_UINT:
		case FORMAT_RGB10_A2_UNORM:
		case FORMAT_RG11B10_UFLOAT:
		case FORMAT_RGB9E5_UFLOAT:
		case FORMAT_D32_FLOAT:
		case FORMAT_D32_FLOAT_S8_UINT: return 4; // Only the depth of a combined format is copied

		case FORMAT_RG32_UINT:
		case FORMAT_RG32_SINT:
		case FORMAT_RG32_FLOAT:
		case FORMAT_RGBA16_UNORM:
		case FORMAT_RGBA16_SNORM:
		case FORMAT_RGBA16_UINT:
		case FORMAT_RGBA16_SINT:
		case FORMAT_RGBA16_FLOAT: return 8;

		case FORMAT_RGBA32_UINT:
		case FORMAT_RGBA32_SINT:
		case FORMAT_RGBA32_FLOAT: return 16;

		// The 24 bit depth formats land here with FORMAT_NONE: their texels have no size anyone can name, which is also why WebGPU refuses to copy them
		default: return 0;
		}
	}
}

namespace GPU::detail {
	// WebGPU wants the rows of a copy 256 byte aligned, but takes a single row with no pitch at all, so tightly packed rows that aren't are copied one at a time.
	inline void copy_texture_rows(GpuCommandBuffer* cmd, WGPUBuffer buffer, uint64_t base, const GpuTexture* texture, bool to_texture) {
		auto [width, height, depth] = texture->descriptor.dimensions;
		uint32_t row_bytes = width * format_bytes(texture->descriptor.format);
		uint32_t first_layer = texture->range ? texture->range->start : 0;
		uint32_t layers = texture->range ? texture->range->end - texture->range->start
			: texture->descriptor.type == TEXTURE_3D ? depth : texture->descriptor.layerCount;
		bool aligned = row_bytes % 256 == 0;
		if(row_bytes == 0 || (!aligned && gpuFormatIsDepthStencilEXT(texture->descriptor.format))) {
			GPU::report(cmd, GPU_DIAGNOSTIC_UNSUPPORTED, "gpuCopyToTexture/gpuCopyFromTexture: WebGPU can't copy the 24 bit depth formats, and copies other depth or stencil textures only whole, so with rows a multiple of 256 bytes; nothing is copied");
			return;
		}

		for(uint32_t layer = 0; layer < (aligned ? 1 : layers); ++layer)
			for(uint32_t row = 0; row < (aligned ? 1 : height); ++row) {
				WGPUTexelCopyBufferInfo linear {
					.layout = {
						.offset = base + (uint64_t(layer) * height + row) * row_bytes,
						.bytesPerRow = aligned ? row_bytes : WGPU_COPY_STRIDE_UNDEFINED,
						.rowsPerImage = aligned ? height : WGPU_COPY_STRIDE_UNDEFINED,
					},
					.buffer = buffer,
				};
				WGPUTexelCopyTextureInfo image {
					.texture = texture->texture,
					.mipLevel = 0, // TODO: How do we allow more control over this?
					.origin = {0, row, first_layer + layer},
					// Combined formats copy their depth, as on Vulkan
					.aspect = gpuFormatIsDepthEXT(texture->descriptor.format) && gpuFormatIsStencilEXT(texture->descriptor.format) ? WGPUTextureAspect_DepthOnly : WGPUTextureAspect_All
				};
				WGPUExtent3D size {width, aligned ? height : 1, aligned ? layers : 1};
				if(to_texture) wgpuCommandEncoderCopyBufferToTexture(cmd->encoder, &linear, &image, &size);
				else wgpuCommandEncoderCopyTextureToBuffer(cmd->encoder, &image, &linear, &size);
			}
	}
}

void gpuCopyToTexture(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, GpuTexture* texture) {
	endCurrentPass(cmd); // A copy is not something a pass can hold, and a dispatch may have opened one
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_);
	auto [dest_range, dest_offset, dest_addr] = dest; auto [src_range, src_offset, src_addr] = src;

	assert(cmd->queue->gpu2textures[dest_addr] == texture);
	GPU::detail::copy_texture_rows(cmd, cmd->queue->monobuffers[src_range.buffer], src_range.start + src_offset, texture, true);
}

void gpuCopyFromTexture(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, const GpuTexture* texture) {
	endCurrentPass(cmd); // A copy is not something a pass can hold, and a dispatch may have opened one
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_);
	auto [dest_range, dest_offset, dest_addr] = dest; auto [src_range, src_offset, src_addr] = src;

	assert(cmd->queue->gpu2textures[src_addr] == texture);
	GPU::detail::copy_texture_rows(cmd, cmd->queue->monobuffers[dest_range.buffer], dest_range.start + dest_offset, texture, false);
}

namespace GPU::detail {
	// A monotexture is sized by its first texture, so a smaller one covers only the top left of its slot, never under half of it.
	inline uvec2 monotexture_slot_extent(const GpuTexture* texture, uint32_t mip) {
		return {
			std::max(wgpuTextureGetWidth(texture->texture) >> mip, 1u),
			std::max(wgpuTextureGetHeight(texture->texture) >> mip, 1u)
		};
	}

	// Creates the single subresource view a texture is attached to a render pass through. The view lives until the submission referencing it has finished.
	inline WGPUTextureView attachment_view(GpuCommandBuffer* cmd, const GpuTexture* texture, uint32_t mipLevel, uint32_t slice) {
		bool volume = texture->descriptor.type == TEXTURE_3D;

		WGPUTextureViewDescriptor d {
			.label = {"NoAPI Attachment", WGPU_STRLEN},
			.format = GPU::format2wgpu(texture->descriptor.format),
			.dimension = volume ? WGPUTextureViewDimension_3D : WGPUTextureViewDimension_2D,
			.baseMipLevel = mipLevel,
			.mipLevelCount = 1,
			// Textures backed by a monotexture live at an offset into its array layers
			.baseArrayLayer = volume ? 0 : (texture->range ? texture->range->start : 0) + slice,
			.arrayLayerCount = 1,
			.aspect = WGPUTextureAspect_All,
			.usage = WGPUTextureUsage_RenderAttachment,
		};
		auto view = wgpuTextureCreateView(texture->texture, &d);

		cmd->queue->code_pending_submission_finished.emplace_back([view] {
			wgpuTextureViewRelease(view);
		}, cmd->queue->next_submission_index);
		return view;
	}

	inline uvec2 attachment_extent(const GpuTexture* texture, uint32_t mipLevel) {
		return {
			std::max(texture->descriptor.dimensions.x >> mipLevel, 1u),
			std::max(texture->descriptor.dimensions.y >> mipLevel, 1u)
		};
	}

	// Both layout flavors compile this module; they differ only in sampler and texture types the WGSL never names.
	constexpr static std::string_view BLIT_WGSL_CODE = R"WGSL(
@group(0) @binding(0) var blit_source : texture_2d<f32>;
@group(0) @binding(1) var blit_sampler : sampler;
// xy puts the source's edge where the destination image ends; zw is that edge half a texel in, replicated across the gap.
@group(0) @binding(2) var<uniform> blit_uv_scale : vec4<f32>;

struct Varyings {
	@builtin(position) position : vec4<f32>,
	@location(0) uv : vec2<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) index : u32) -> Varyings {
	// An oversized triangle covering the whole viewport, its uvs run 0..1 across the covered area
	let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
	return Varyings(
		// WebGPU's clip space puts +Y at the top of the viewport, but v = 0 is the top texel row
		vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0),
		uv * blit_uv_scale.xy
	);
}

@fragment
fn fragment(varyings : Varyings) -> @location(0) vec4<f32> {
	// The view covers a single mip, so there is never another level to pick between
	return textureSampleLevel(blit_source, blit_sampler, min(varyings.uv, blit_uv_scale.zw), 0.0);
}
)WGSL";

	inline void ensure_blit_layouts(GpuQueue* queue, bool filtering) {
		if(queue->blit_bind_group_layouts[filtering]) return;

		std::array<WGPUBindGroupLayoutEntry, 3> entries = {
			WGPUBindGroupLayoutEntry{
				.binding = 0,
				.visibility = WGPUShaderStage_Fragment,
				.texture = {
					.sampleType = filtering ? WGPUTextureSampleType_Float : WGPUTextureSampleType_UnfilterableFloat,
					.viewDimension = WGPUTextureViewDimension_2D,
					.multisampled = false,
				}
			}, WGPUBindGroupLayoutEntry{
				.binding = 1,
				.visibility = WGPUShaderStage_Fragment,
				.sampler = {
					.type = filtering ? WGPUSamplerBindingType_Filtering : WGPUSamplerBindingType_NonFiltering,
				}
			}, WGPUBindGroupLayoutEntry{
				.binding = 2,
				// Both stages: the vertex half reads the scale, the fragment half reads the limit it pins the source coordinate to
				.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
				.buffer = {
					.type = WGPUBufferBindingType_Uniform,
					.minBindingSize = sizeof(float) * 4,
				}
			}
		};

		WGPUBindGroupLayoutDescriptor layout {
			.label = {"NoAPI Blit", WGPU_STRLEN},
			.entryCount = entries.size(),
			.entries = entries.data(),
		};
		queue->blit_bind_group_layouts[filtering] = wgpuDeviceCreateBindGroupLayout(queue->device, &layout);

		WGPUPipelineLayoutDescriptor pipeline_layout {
			.label = {"NoAPI Blit", WGPU_STRLEN},
			.bindGroupLayoutCount = 1,
			.bindGroupLayouts = &queue->blit_bind_group_layouts[filtering],
		};
		queue->blit_pipeline_layouts[filtering] = wgpuDeviceCreatePipelineLayout(queue->device, &pipeline_layout);

		WGPUSamplerDescriptor sampler {
			.label = {"NoAPI Blit", WGPU_STRLEN},
			.addressModeU = WGPUAddressMode_ClampToEdge,
			.addressModeV = WGPUAddressMode_ClampToEdge,
			.addressModeW = WGPUAddressMode_ClampToEdge,
			.magFilter = filtering ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest,
			.minFilter = filtering ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest,
			.mipmapFilter = WGPUMipmapFilterMode_Nearest,
			.lodMinClamp = 0,
			.lodMaxClamp = 1,
			.maxAnisotropy = 1,
		};
		queue->blit_samplers[filtering] = wgpuDeviceCreateSampler(queue->device, &sampler);
	}

	// The format is baked into the pipeline and the sampler type into its layout, so one is kept per (format, filtering).
	inline WGPURenderPipeline blit_pipeline(GpuQueue* queue, WGPUTextureFormat format, bool filtering) {
		auto key = static_cast<uint64_t>(format) << 1 | filtering;
		if(auto found = queue->blit_pipelines.find(key); found != queue->blit_pipelines.end())
			return found->second;

		ensure_blit_layouts(queue, filtering);

		if(!queue->blit_shader) {
			WGPUShaderSourceWGSL source {
				.chain = {.sType = WGPUSType_ShaderSourceWGSL},
				.code = {BLIT_WGSL_CODE.data(), BLIT_WGSL_CODE.size()}
			};
			WGPUShaderModuleDescriptor module {
				.nextInChain = &source.chain,
				.label = {"NoAPI Blit", WGPU_STRLEN},
			};
			queue->blit_shader = wgpuDeviceCreateShaderModule(queue->device, &module);
		}

		WGPUColorTargetState target {
			.format = format,
			.writeMask = WGPUColorWriteMask_All,
		};
		WGPUFragmentState fragment {
			.module = queue->blit_shader,
			.entryPoint = {"fragment", WGPU_STRLEN},
			.targetCount = 1,
			.targets = &target,
		};
		WGPURenderPipelineDescriptor d {
			.label = {"NoAPI Blit", WGPU_STRLEN},
			.layout = queue->blit_pipeline_layouts[filtering],
			.vertex = {
				.module = queue->blit_shader,
				.entryPoint = {"vertex", WGPU_STRLEN},
			},
			.primitive = {
				.topology = WGPUPrimitiveTopology_TriangleList,
				.stripIndexFormat = WGPUIndexFormat_Undefined,
				.frontFace = WGPUFrontFace_CCW,
				.cullMode = WGPUCullMode_None,
			},
			.multisample = {
				.count = 1,
				.mask = 0xFFFFFFFF,
			},
			.fragment = &fragment,
		};

		auto pipeline = wgpuDeviceCreateRenderPipeline(queue->device, &d);
		queue->blit_pipelines[key] = pipeline;
		return pipeline;
	}
}

void gpuBlitTextureEXT(GpuCommandBuffer* cmd, GpuTexture* destination, const GpuTexture* source,
		bool linear_filter,
		uint32_t destination_mip, uint32_t destination_slice,
		uint32_t source_mip, uint32_t source_slice) {
	assert(!cmd->render_pass && "A blit opens a render pass of its own, so it can't be recorded inside another one");
	assert((source->descriptor.usage & USAGE_SAMPLED) && "The blit source must have been created with USAGE_SAMPLED");
	assert((destination->descriptor.usage & USAGE_COLOR_ATTACHMENT) && "The blit destination must have been created with USAGE_COLOR_ATTACHMENT");
	assert(!gpuFormatIsDepthStencilEXT(source->descriptor.format) && !gpuFormatIsDepthStencilEXT(destination->descriptor.format) && "Depth/stencil textures can't be blitted");
	assert(source->descriptor.type != TEXTURE_3D && "A slice of a 3D texture can't be bound on its own, blit out of a 2D array instead");
	assert(source_mip < source->descriptor.mipCount && destination_mip < destination->descriptor.mipCount);

	endComputePass(cmd);

	// One mip of one layer, so the shader stays a plain texture_2d whatever the source is.
	WGPUTextureViewDescriptor view {
		.label = {"NoAPI Blit Source", WGPU_STRLEN},
		.format = GPU::format2wgpu(source->descriptor.format),
		.dimension = WGPUTextureViewDimension_2D,
		.baseMipLevel = source_mip,
		.mipLevelCount = 1,
		// Textures backed by a monotexture live at an offset into its array layers
		.baseArrayLayer = (source->range ? source->range->start : 0) + source_slice,
		.arrayLayerCount = 1,
		.aspect = WGPUTextureAspect_All,
		.usage = WGPUTextureUsage_TextureBinding,
	};
	auto source_view = wgpuTextureCreateView(source->texture, &view);
	cmd->queue->code_pending_submission_finished.emplace_back([source_view] {
		wgpuTextureViewRelease(source_view);
	}, cmd->queue->next_submission_index);

	// Building the pipeline is what brings the matching layout and sampler into existence
	bool filtering = linear_filter && gpuFormatIsFilterableEXT(source->descriptor.format);
	auto pipeline = GPU::detail::blit_pipeline(cmd->queue, GPU::format2wgpu(destination->descriptor.format), filtering);

	// The viewport covers the whole slot, so fragments past the image replicate the source's edge into the gap; each side needs its own slot ratio.
	auto extent = GPU::detail::attachment_extent(destination, destination_mip);
	auto slot = GPU::detail::monotexture_slot_extent(destination, destination_mip);
	auto source_extent = GPU::detail::attachment_extent(source, source_mip);
	auto source_slot = GPU::detail::monotexture_slot_extent(source, source_mip);

	float source_fraction[2] = {
		static_cast<float>(source_extent.x) / source_slot.x,
		static_cast<float>(source_extent.y) / source_slot.y,
	};
	float uv_scale[4] = {
		source_fraction[0] * slot.x / extent.x,
		source_fraction[1] * slot.y / extent.y,
		source_fraction[0] - 0.5f / source_slot.x,
		source_fraction[1] - 0.5f / source_slot.y,
	};
	WGPUBufferDescriptor scale_desc {
		.label = {"NoAPI Blit UV Scale", WGPU_STRLEN},
		.usage = WGPUBufferUsage_Uniform,
		.size = sizeof(uv_scale),
		.mappedAtCreation = true,
	};
	auto scale_buffer = wgpuDeviceCreateBuffer(cmd->queue->device, &scale_desc);
	memcpy(wgpuBufferGetMappedRange(scale_buffer, 0, scale_desc.size), uv_scale, sizeof(uv_scale));
	wgpuBufferUnmap(scale_buffer);
	cmd->queue->code_pending_submission_finished.emplace_back([scale_buffer] {
		wgpuBufferRelease(scale_buffer);
	}, cmd->queue->next_submission_index);

	std::array<WGPUBindGroupEntry, 3> entries = {
		WGPUBindGroupEntry{
			.binding = 0,
			.textureView = source_view,
		}, WGPUBindGroupEntry{
			.binding = 1,
			.sampler = cmd->queue->blit_samplers[filtering],
		}, WGPUBindGroupEntry{
			.binding = 2,
			.buffer = scale_buffer,
			.offset = 0,
			.size = sizeof(uv_scale),
		}
	};
	WGPUBindGroupDescriptor group_desc {
		.label = {"NoAPI Blit", WGPU_STRLEN},
		.layout = cmd->queue->blit_bind_group_layouts[filtering],
		.entryCount = entries.size(),
		.entries = entries.data(),
	};
	auto group = wgpuDeviceCreateBindGroup(cmd->queue->device, &group_desc);
	cmd->queue->code_pending_submission_finished.emplace_back([group] {
		wgpuBindGroupRelease(group);
	}, cmd->queue->next_submission_index);

	WGPURenderPassColorAttachment attachment {
		.view = GPU::detail::attachment_view(cmd, destination, destination_mip, destination_slice),
		// Only a view into a 3D texture picks the written slice here, everything else selected it as an array layer
		.depthSlice = destination->descriptor.type == TEXTURE_3D ? destination_slice : WGPU_DEPTH_SLICE_UNDEFINED,
		// The draw covers the whole mip, so there is never anything worth loading back in first
		.loadOp = WGPULoadOp_Clear,
		.storeOp = WGPUStoreOp_Store,
		.clearValue = {0, 0, 0, 0},
	};
	WGPURenderPassDescriptor pass_desc {
		.label = {"NoAPI Blit", WGPU_STRLEN},
		.colorAttachmentCount = 1,
		.colorAttachments = &attachment,
	};

	// A pass of its own, so the state the user had set on cmd->render_pass is undisturbed.
	auto pass = wgpuCommandEncoderBeginRenderPass(cmd->encoder, &pass_desc);
	wgpuRenderPassEncoderSetPipeline(pass, pipeline);
	wgpuRenderPassEncoderSetBindGroup(pass, 0, group, 0, nullptr);
	wgpuRenderPassEncoderSetViewport(pass, 0, 0, slot.x, slot.y, 0, 1);
	wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, slot.x, slot.y);
	wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
	wgpuRenderPassEncoderEnd(pass);
	wgpuRenderPassEncoderRelease(pass);
}

void gpuSetActiveTextureHeapPtr(GpuCommandBuffer* cmd, gpu* texture_heap) {
	// TODO: Should we add some validation to check that what is provided is a valid texture heap?
	auto [dest_range, dest_offset, dest_addr] = GPU::detail::closest_buffer(cmd->queue, texture_heap);
	dest_range.start += dest_offset;
	// dest_range.end -= dest_offset;

	// A pass can't bind a monobuffer both writable and as a heap, so the heap is snapshotted and shader writes go unseen.
	GPU::report(cmd, GPU_DIAGNOSTIC_EMULATED,
		"gpuSetActiveTextureHeapPtr: a WebGPU monobuffer cannot be bound as a descriptor heap, so the heap is copied on every call and writes a shader makes to it are not seen");
	endCurrentPass(cmd); // A copy can't be recorded with a pass open

	auto size = dest_range.size();
	WGPUBufferDescriptor d {
		.label = {"NoAPI Texture Heap", WGPU_STRLEN},
		.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
		.size = size,
	};
	auto snapshot = wgpuDeviceCreateBuffer(cmd->queue->device, &d);
	wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, cmd->queue->monobuffers[dest_range.buffer], dest_range.start, snapshot, 0, size);
	cmd->queue->code_pending_submission_finished.emplace_back([snapshot](){ wgpuBufferRelease(snapshot); }, cmd->queue->next_submission_index);

	cmd->active_texture_heap = snapshot;
	cmd->active_texture_heap_size = size;
}

// WebGPU orders passes and makes their writes visible itself, so a barrier has nothing to add.
void gpuBarrier(GpuCommandBuffer* cmd, STAGE before, STAGE after, HAZARD_FLAGS hazards) {}

namespace GPU {
	inline void push_to_monobuffer(GpuCommandBuffer* cmd, GpuQueue::MonobufferRange range, void* cpu) {
		auto size = range.size();
		WGPUBufferDescriptor d{
			.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_MapWrite,
			.size = size,
			.mappedAtCreation = true,
		};
		auto tmp = wgpuDeviceCreateBuffer(cmd->queue->device, &d);
		auto tmp_cpu = wgpuBufferGetMappedRange(tmp, 0, size);
		memcpy(tmp_cpu, cpu, size);
		wgpuBufferUnmap(tmp);

		wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, tmp, 0, cmd->queue->monobuffers[range.buffer], range.start, size);
		cmd->code_pending_submission_finished.emplace_back([tmp]() {
			wgpuBufferRelease(tmp);
		});
	}

	inline size_t push_to_monobuffer(GpuQueue* queue, GpuQueue::MonobufferRange range, void* cpu) {
		auto cmd = gpuStartCommandRecording(queue);
		push_to_monobuffer(cmd, range, cpu);
		return gpuSubmit(queue, {&cmd, 1});

	}
}

void gpuSignalAfter(GpuCommandBuffer* cmd, STAGE before, gpu* ptr, uint64_t value, SIGNAL signal) {
	// WGSL has only 32 bit atomics, so MAX and OR on a 64 bit counter can't be done for multiple producers.
	if(signal != SIGNAL_ATOMIC_SET) {
		GPU::report(cmd, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuSignalAfter: SIGNAL_ATOMIC_MAX and SIGNAL_ATOMIC_OR need a 64 bit atomic, which WGSL has no form of, so the counter is left alone");
		return;
	}

	// The counter is written by a copy, and the command encoder can't record one while a render pass is open
	if(cmd->render_pass) {
		GPU::report(cmd, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuSignalAfter: writing the counter needs a copy, which can't be recorded inside a render pass, so nothing is signalled");
		return;
	}

	// Ordered in the command stream, so the counter lands after whatever was recorded before it.
	endComputePass(cmd);
	auto counter = GPU::detail::closest_buffer(cmd->queue, ptr);
	if(!counter.base) return;

	auto range = counter.range;
	range.start += counter.offset;
	range.end = range.start + sizeof(value);
	GPU::push_to_monobuffer(cmd, range, &value);
}

void gpuWaitBefore(GpuCommandBuffer* cmd, STAGE after, gpu* ptr, uint64_t value, OP op, HAZARD_FLAGS hazards, uint64_t mask) {
	// WebGPU can't wait on memory and has no events, but orders passes anyway, so the dependency holds and only the split is lost.
	GPU::report(cmd, GPU_DIAGNOSTIC_EMULATED,
		"gpuWaitBefore: WebGPU has neither a wait on memory nor events, so nothing is split; the ordering the pair asks for is already implied by its pass boundaries");
}

void gpuSetPipeline(GpuCommandBuffer* cmd, const GpuPipeline* pipeline) {
	cmd->bound_pipeline = pipeline;

	if(std::holds_alternative<GpuPipeline::ComputeCache>(pipeline->cache)) {
		if(pipeline->reference_layout != cmd->queue->current_compute_pipeline_layout)
			update_compute_pipeline(cmd->queue, pipeline);

		endRenderPass(cmd);
		if(!cmd->compute_pass) cmd->compute_pass = wgpuCommandEncoderBeginComputePass(cmd->encoder, nullptr);

		wgpuComputePassEncoderSetPipeline(cmd->compute_pass, std::get<GpuPipeline::ComputeCache>(pipeline->cache).pipeline);
	} else {
		endComputePass(cmd);

		// A render pipeline variant is built and set by the draws, once the state it is drawn with is known.
	}
}

// Offsets match the generated GpuShaderData (std140 style); each is padded to the 16 byte multiple WGSL rounds a uniform block's size to.
struct ComputeShaderData {
	gpu* compute;
	uint64_t _padding = 0;
};

// Builds group 0: every monobuffer, the active texture heap, and a throwaway uniform holding the root data pointer(s) the shaders were invoked with
WGPUBindGroup createBufferBindGroup(GpuCommandBuffer* cmd, const void* shader_data, size_t shader_data_size, bool compute) {
	uint32_t binding = 0;
	std::vector<WGPUBindGroupEntry> group0;
	for(uint32_t i = 0; i < cmd->queue->monobuffers.size(); ++i)
		group0.push_back({
			.binding = binding++,
			.buffer = cmd->queue->monobuffers[i],
			.offset = 0,
			.size = cmd->queue->monobuffer_sizes[i]
		});

	// If there is no active texture heap bind the empty buffer
	if(cmd->active_texture_heap)
		group0.push_back({
			.binding = binding++,
			.buffer = cmd->active_texture_heap,
			.offset = 0,
			.size = cmd->active_texture_heap_size
		});
	else group0.push_back({
		.binding = binding++,
		.buffer = cmd->queue->empty_buffer,
		.offset = 0,
		.size = 4
	});

	{
		auto& sampler_set = cmd->sampler_set ? *cmd->sampler_set : *cmd->queue->default_sampler_set;
		group0.push_back({
			.binding = binding++,
			.buffer = sampler_set.lookup_buffer,
			.offset = 0,
			.size = GPU::detail::sampler_lookup_size * sizeof(uint32_t)
		});
	}

	{
		WGPUBufferDescriptor d {
			.label = {"NoAPI ShaderData", WGPU_STRLEN},
			.usage = WGPUBufferUsage_Uniform,
			.size = shader_data_size,
			.mappedAtCreation = true,
		};
		auto tmp = wgpuDeviceCreateBuffer(cmd->queue->device, &d);
		cmd->queue->code_pending_submission_finished.emplace_back([tmp] { wgpuBufferRelease(tmp); }, cmd->queue->next_submission_index);
		memcpy(wgpuBufferGetMappedRange(tmp, 0, d.size), shader_data, shader_data_size);
		wgpuBufferUnmap(tmp);

		group0.push_back({
			.binding = binding++,
			.buffer = tmp,
			.offset = 0,
			.size = d.size
		});
	}

	WGPUBindGroupDescriptor d {
		.layout = compute ? cmd->queue->current_compute_bind_group_layout0 : cmd->queue->current_graphics_bind_group_layout0,
		.entryCount = group0.size(),
		.entries = group0.data()
	};
	return wgpuDeviceCreateBindGroup(cmd->queue->device, &d);
}

// Binds every group needed by a dispatch or a draw (the texture groups are shared between the two)
void bindGroups(GpuCommandBuffer* cmd, const void* shader_data, size_t shader_data_size, bool compute) {
	auto group0 = createBufferBindGroup(cmd, shader_data, shader_data_size, compute);
	cmd->queue->code_pending_submission_finished.emplace_back([group0] { wgpuBindGroupRelease(group0); }, cmd->queue->next_submission_index);

	auto group3 = (cmd->sampler_set ? cmd->sampler_set : cmd->queue->default_sampler_set)->bind_group;

	if(compute) {
		wgpuComputePassEncoderSetBindGroup(cmd->compute_pass, 0, group0, 0, nullptr);
		wgpuComputePassEncoderSetBindGroup(cmd->compute_pass, 1, cmd->queue->current_compute_bind_group1, 0, nullptr);
		wgpuComputePassEncoderSetBindGroup(cmd->compute_pass, 2, cmd->queue->current_compute_bind_group2, 0, nullptr);
		wgpuComputePassEncoderSetBindGroup(cmd->compute_pass, 3, group3, 0, nullptr);
	} else {
		wgpuRenderPassEncoderSetBindGroup(cmd->render_pass, 0, group0, 0, nullptr);
		wgpuRenderPassEncoderSetBindGroup(cmd->render_pass, 1, cmd->queue->current_graphics_bind_group1, 0, nullptr);
		wgpuRenderPassEncoderSetBindGroup(cmd->render_pass, 2, cmd->queue->current_graphics_bind_group2, 0, nullptr);
		wgpuRenderPassEncoderSetBindGroup(cmd->render_pass, 3, group3, 0, nullptr);
	}
}

void bindComputeGroups(GpuCommandBuffer* cmd, gpu* data) {
	ComputeShaderData shader_data { .compute = data };
	bindGroups(cmd, &shader_data, sizeof(shader_data), true);
}

void gpuDispatch(GpuCommandBuffer* cmd, gpu* data, uvec3 grid_dimensions) {
	assert(cmd->bound_pipeline);
	// A copy or signal since gpuSetPipeline ended the pass, which Vulkan has no equivalent of, so it is reopened with the pipeline still bound
	if(!cmd->compute_pass) gpuSetPipeline(cmd, cmd->bound_pipeline);

	bindComputeGroups(cmd, data);

	wgpuComputePassEncoderDispatchWorkgroups(cmd->compute_pass, grid_dimensions.x, grid_dimensions.y, grid_dimensions.z);
}

void gpuDispatchIndirect(GpuCommandBuffer* cmd, gpu* data, gpu* grid_dimensions_gpu) {
	assert(cmd->bound_pipeline);
	assert(grid_dimensions_gpu);

	auto [grid_range, grid_offset, grid_addr] = GPU::detail::closest_buffer(cmd->queue, grid_dimensions_gpu);

	// A compute pass binds every monobuffer writable, and WebGPU won't read indirect arguments from a buffer written in the same pass
	endComputePass(cmd);
	WGPUBufferDescriptor d {
		.label = {"NoAPI Dispatch Arguments", WGPU_STRLEN},
		.usage = WGPUBufferUsage_Indirect | WGPUBufferUsage_CopyDst,
		.size = sizeof(uvec3),
	};
	auto arguments = wgpuDeviceCreateBuffer(cmd->queue->device, &d);
	wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, cmd->queue->monobuffers[grid_range.buffer], grid_range.start + grid_offset, arguments, 0, sizeof(uvec3));
	cmd->queue->code_pending_submission_finished.emplace_back([arguments](){ wgpuBufferRelease(arguments); }, cmd->queue->next_submission_index);

	gpuSetPipeline(cmd, cmd->bound_pipeline);
	bindComputeGroups(cmd, data);
	wgpuComputePassEncoderDispatchWorkgroupsIndirect(cmd->compute_pass, arguments, 0);
}

GpuPipeline* gpuCreateGraphicsPipeline(GpuQueue* queue, GpuByteSpan vertexIR, GpuByteSpan fragmentIR, const GpuRasterDesc& desc) {
	if(desc.supportDualSourceBlending && !queue->capabilities.dual_source_blending) {
		GPU::report(queue, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuCreateGraphicsPipeline: this device was created without dual-source-blending, so a pipeline with supportDualSourceBlending is not created");
		errno = WGPUErrorType_Validation;
		return nullptr;
	}

	auto out = (GpuPipeline*)queue->cpu_allocator(nullptr, sizeof(GpuPipeline));
	new(out) GpuPipeline{
		.reference_layout = nullptr,
		.cache = GpuPipeline::RenderCache {
			.vertexIR = {(char*)vertexIR.data(), (char*)(vertexIR.data() + vertexIR.size())},
			.fragmentIR = {(char*)fragmentIR.data(), (char*)(fragmentIR.data() + fragmentIR.size())},
			.color_targets = {desc.colorTargets.begin(), desc.colorTargets.end()},
			.descriptor = desc,
		}
	};

	// The descriptor only borrows its color targets, so point it at our copy of them
	auto& cache = std::get<GpuPipeline::RenderCache>(out->cache);
	cache.descriptor.colorTargets = cache.color_targets;

	// Compiled once now, though only a draw builds the pipeline, so a broken shader fails here as it does on Vulkan
	for(auto [IR, stage]: {std::pair{std::string_view(cache.vertexIR), GPU::shaders::SHADER_STAGE::VERTEX}, std::pair{std::string_view(cache.fragmentIR), GPU::shaders::SHADER_STAGE::FRAGMENT}}) {
		if(IR.empty()) continue;
		auto module = create_shader_module(queue, IR, stage);
		if(!module) {
			gpuFreePipeline(queue, out);
			return nullptr;
		}
		wgpuShaderModuleRelease(module);
	}

	// Built by the first draw, once the depth/stencil, blend and index state it bakes in are known.
	return out;
}

// WebGPU has no mesh stage in any form, so there is no degraded version to hand back; capabilities say so up front.
GpuPipeline* gpuCreateGraphicsMeshletPipeline(GpuQueue* queue, GpuByteSpan meshletIR, GpuByteSpan fragmentIR, const GpuRasterDesc& desc) {
	GPU::report(queue, GPU_DIAGNOSTIC_UNSUPPORTED,
		"gpuCreateGraphicsMeshletPipeline: WebGPU has no mesh shaders, so no pipeline is created");
	return nullptr;
}

GpuDepthStencilState* gpuCreateDepthStencilState(GpuQueue* queue, const GpuDepthStencilDesc& desc) {
	auto out = (GpuDepthStencilState*)queue->cpu_allocator(nullptr, sizeof(GpuDepthStencilState));
	new(out) GpuDepthStencilState{.descriptor = desc};
	return out;
}

GpuBlendState* gpuCreateBlendState(GpuQueue* queue, const GpuBlendDesc& desc) {
	auto out = (GpuBlendState*)queue->cpu_allocator(nullptr, sizeof(GpuBlendState));
	new(out) GpuBlendState{.descriptor = desc};
	return out;
}

void gpuFreeDepthStencilState(GpuQueue* queue, GpuDepthStencilState* state) {
	state->~GpuDepthStencilState();
	queue->cpu_allocator(state, 0);
}

void gpuFreeBlendState(GpuQueue* queue, GpuBlendState* state) {
	state->~GpuBlendState();
	queue->cpu_allocator(state, 0);
}

void gpuSetDepthStencilState(GpuCommandBuffer* cmd, const GpuDepthStencilState* state) {
	// WebGPU bakes all but the stencil reference into the pipeline, so this is recorded for bindRenderPipeline to match.
	cmd->depth_stencil = state ? std::optional{state->descriptor} : std::nullopt;
	if(state && state->descriptor.stencilFront.reference != state->descriptor.stencilBack.reference)
		GPU::report(cmd, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuSetDepthStencilState: WebGPU has one stencil reference for both faces, so back faces are tested against the front's");

	if(cmd->render_pass && cmd->depth_stencil)
		wgpuRenderPassEncoderSetStencilReference(cmd->render_pass, cmd->depth_stencil->stencilFront.reference);
}

void gpuSetBlendState(GpuCommandBuffer* cmd, const GpuBlendState* state) {
	// Blending is part of a WebGPU PSO, so this is recorded and applied by the next draw
	cmd->blend = state ? std::optional{state->descriptor} : std::nullopt;
}

void gpuSetViewportEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin, float depth_min, float depth_max) {
	assert(cmd->render_pass && "The viewport can only be set inside a render pass");
	wgpuRenderPassEncoderSetViewport(cmd->render_pass, origin.x, origin.y, extent.x, extent.y, depth_min, depth_max);
}

void gpuSetScissorRectEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin) {
	assert(cmd->render_pass && "The scissor rectangle can only be set inside a render pass");
	wgpuRenderPassEncoderSetScissorRect(cmd->render_pass, origin.x, origin.y, extent.x, extent.y);
}

namespace GPU {
	// WebGPU has no "don't care", but discarding/clearing expresses the same intent to a tiler
	inline WGPULoadOp load2wgpu(LOAD_OP op) {
		switch (op) {
		case LOAD_OP_LOAD: return WGPULoadOp_Load;
		case LOAD_OP_CLEAR:
		case LOAD_OP_DONT_CARE: return WGPULoadOp_Clear;
		}
		std::unreachable();
	}

	inline WGPUStoreOp store2wgpu(STORE_OP op) {
		switch (op) {
		case STORE_OP_STORE: return WGPUStoreOp_Store;
		case STORE_OP_DONT_CARE: return WGPUStoreOp_Discard;
		}
		std::unreachable();
	}
}

void gpuBeginRenderPass(GpuCommandBuffer* cmd, const GpuRenderPassDesc& desc) {
	endCurrentPass(cmd);

	uvec2 extent = {0, 0};

	std::vector<WGPURenderPassColorAttachment> colors; colors.reserve(desc.colorAttachments.size());
	for(auto& color: desc.colorAttachments) {
		bool volume = color.texture->descriptor.type == TEXTURE_3D;
		colors.push_back({
			.view = GPU::detail::attachment_view(cmd, color.texture, color.mipLevel, color.slice),
			// Only a view into a 3D texture picks the rendered slice here, everything else selects it as an array layer
			.depthSlice = volume ? color.slice : WGPU_DEPTH_SLICE_UNDEFINED,
			.resolveTarget = color.resolveTexture ? GPU::detail::attachment_view(cmd, color.resolveTexture, color.mipLevel, color.slice) : nullptr,
			.loadOp = GPU::load2wgpu(color.loadOp),
			.storeOp = GPU::store2wgpu(color.storeOp),
			.clearValue = {color.clearValue.r, color.clearValue.g, color.clearValue.b, color.clearValue.a},
		});

		if(extent.x == 0) extent = GPU::detail::attachment_extent(color.texture, color.mipLevel);
	}

	// WebGPU merges depth and stencil into one attachment, so when both are given they must name the same subresource.
	auto& merged = desc.depthAttachment ? desc.depthAttachment : desc.stencilAttachment;
	assert((!(desc.depthAttachment && desc.stencilAttachment)
		|| (desc.depthAttachment->texture == desc.stencilAttachment->texture
			&& desc.depthAttachment->mipLevel == desc.stencilAttachment->mipLevel
			&& desc.depthAttachment->slice == desc.stencilAttachment->slice))
		&& "WebGPU requires the depth and stencil attachments to be the same subresource");

	WGPURenderPassDepthStencilAttachment depth_stencil {};
	if(merged) {
		auto format = merged->texture->descriptor.format;
		depth_stencil.view = GPU::detail::attachment_view(cmd, merged->texture, merged->mipLevel, merged->slice);

		// Both aspects of the format need ops, whether or not the user described that aspect
		if(gpuFormatIsDepthEXT(format)) {
			auto& attachment = desc.depthAttachment ? *desc.depthAttachment : *merged;
			depth_stencil.depthLoadOp = desc.depthAttachment ? GPU::load2wgpu(attachment.loadOp) : WGPULoadOp_Load;
			depth_stencil.depthStoreOp = desc.depthAttachment ? GPU::store2wgpu(attachment.storeOp) : WGPUStoreOp_Store;
			depth_stencil.depthClearValue = static_cast<float>(attachment.clearValue);
		}
		if(gpuFormatIsStencilEXT(format)) {
			auto& attachment = desc.stencilAttachment ? *desc.stencilAttachment : *merged;
			depth_stencil.stencilLoadOp = desc.stencilAttachment ? GPU::load2wgpu(attachment.loadOp) : WGPULoadOp_Load;
			depth_stencil.stencilStoreOp = desc.stencilAttachment ? GPU::store2wgpu(attachment.storeOp) : WGPUStoreOp_Store;
			depth_stencil.stencilClearValue = static_cast<uint32_t>(attachment.clearValue);
		}

		if(extent.x == 0) extent = GPU::detail::attachment_extent(merged->texture, merged->mipLevel);
	}

	WGPURenderPassDescriptor d {
		.label = {"NoAPI Render Pass", WGPU_STRLEN},
		.colorAttachmentCount = colors.size(),
		.colorAttachments = colors.data(),
		.depthStencilAttachment = merged ? &depth_stencil : nullptr,
	};
	cmd->render_pass = wgpuCommandEncoderBeginRenderPass(cmd->encoder, &d);

	gpuSetViewportEXT(cmd, extent);
	gpuSetScissorRectEXT(cmd, extent);
}

void gpuEndRenderPass(GpuCommandBuffer* cmd, GpuOptionalRenderPassDesc desc) {
	// The descriptor is only needed by backends that transition images by hand, WebGPU tracks that itself
	endRenderPass(cmd);
}

namespace GPU {
	template<typename T>
	inline bool same(const std::optional<T>& a, const std::optional<T>& b) {
		if(a.has_value() != b.has_value()) return false;
		return !a.has_value() || *a == *b;
	}

	inline WGPUCompareFunction op2wgpu(OP op) {
		switch (op) {
		case OP_NEVER: return WGPUCompareFunction_Never;
		case OP_LESS: return WGPUCompareFunction_Less;
		case OP_EQUAL: return WGPUCompareFunction_Equal;
		case OP_LESS_EQUAL: return WGPUCompareFunction_LessEqual;
		case OP_GREATER: return WGPUCompareFunction_Greater;
		case OP_NOT_EQUAL: return WGPUCompareFunction_NotEqual;
		case OP_GREATER_EQUAL: return WGPUCompareFunction_GreaterEqual;
		case OP_ALWAYS: return WGPUCompareFunction_Always;
		}
		std::unreachable();
	}

	inline WGPUStencilOperation stencil2wgpu(STENCIL_OP op) {
		switch (op) {
		case STENCIL_OP_KEEP: return WGPUStencilOperation_Keep;
		case STENCIL_OP_ZERO: return WGPUStencilOperation_Zero;
		case STENCIL_OP_REPLACE: return WGPUStencilOperation_Replace;
		case STENCIL_OP_INCR_SAT: return WGPUStencilOperation_IncrementClamp;
		case STENCIL_OP_DECR_SAT: return WGPUStencilOperation_DecrementClamp;
		case STENCIL_OP_INVERT: return WGPUStencilOperation_Invert;
		case STENCIL_OP_INCR_WRAP: return WGPUStencilOperation_IncrementWrap;
		case STENCIL_OP_DECR_WRAP: return WGPUStencilOperation_DecrementWrap;
		}
		std::unreachable();
	}

	inline WGPUBlendOperation blend2wgpu(BLEND op) {
		switch (op) {
		case BLEND_ADD: return WGPUBlendOperation_Add;
		case BLEND_SUBTRACT: return WGPUBlendOperation_Subtract;
		case BLEND_REV_SUBTRACT: return WGPUBlendOperation_ReverseSubtract;
		case BLEND_MIN: return WGPUBlendOperation_Min;
		case BLEND_MAX: return WGPUBlendOperation_Max;
		}
		std::unreachable();
	}

	inline WGPUBlendFactor factor2wgpu(FACTOR factor) {
		switch (factor) {
		case FACTOR_ZERO: return WGPUBlendFactor_Zero;
		case FACTOR_ONE: return WGPUBlendFactor_One;
		case FACTOR_SRC_COLOR: return WGPUBlendFactor_Src;
		case FACTOR_ONE_MINUS_SRC_COLOR: return WGPUBlendFactor_OneMinusSrc;
		case FACTOR_DST_COLOR: return WGPUBlendFactor_Dst;
		case FACTOR_ONE_MINUS_DST_COLOR: return WGPUBlendFactor_OneMinusDst;
		case FACTOR_SRC_ALPHA: return WGPUBlendFactor_SrcAlpha;
		case FACTOR_ONE_MINUS_SRC_ALPHA: return WGPUBlendFactor_OneMinusSrcAlpha;
		case FACTOR_DST_ALPHA: return WGPUBlendFactor_DstAlpha;
		case FACTOR_ONE_MINUS_DST_ALPHA: return WGPUBlendFactor_OneMinusDstAlpha;
		case FACTOR_SRC1_COLOR: return WGPUBlendFactor_Src1;
		case FACTOR_ONE_MINUS_SRC1_COLOR: return WGPUBlendFactor_OneMinusSrc1;
		case FACTOR_SRC1_ALPHA: return WGPUBlendFactor_Src1Alpha;
		case FACTOR_ONE_MINUS_SRC1_ALPHA: return WGPUBlendFactor_OneMinusSrc1Alpha;
		}
		std::unreachable();
	}

	inline WGPUColorWriteMask mask2wgpu(uint8_t mask) {
		WGPUColorWriteMask out = WGPUColorWriteMask_None;
		if (mask & 0x1) out |= WGPUColorWriteMask_Red;
		if (mask & 0x2) out |= WGPUColorWriteMask_Green;
		if (mask & 0x4) out |= WGPUColorWriteMask_Blue;
		if (mask & 0x8) out |= WGPUColorWriteMask_Alpha;
		return out;
	}

	inline WGPUPrimitiveTopology topology2wgpu(TOPOLOGY topology) {
		switch (topology) {
		case TOPOLOGY_TRIANGLE_LIST: return WGPUPrimitiveTopology_TriangleList;
		case TOPOLOGY_TRIANGLE_STRIP: return WGPUPrimitiveTopology_TriangleStrip;
		}
		std::unreachable();
	}

	// We define counter clockwise triangles as front facing (matching the Vulkan backend)
	inline WGPUCullMode cull2wgpu(CULL cull) {
		switch (cull) {
		case CULL_NONE: return WGPUCullMode_None;
		case CULL_CCW: return WGPUCullMode_Front;
		case CULL_CW:
		case CULL_ALL: // Never built: WebGPU can't cull both faces, so the draws skip a CULL_ALL pipeline instead (see skips_draw)
			return WGPUCullMode_Back;
		}
		std::unreachable();
	}

	inline WGPUIndexFormat index2wgpu(INDEX_TYPE_EXT type) {
		switch (type) {
		case INDEX_TYPE_UINT8: // Never reached: the draws skip 8 bit indices, which WebGPU has no format for (see skips_draw)
		case INDEX_TYPE_UINT16: return WGPUIndexFormat_Uint16;
		case INDEX_TYPE_UINT32: return WGPUIndexFormat_Uint32;
		}
		std::unreachable();
	}
}

// WebGPU bakes the depth/stencil, blend and index state into the pipeline, so each variant records what it was built with.
void update_render_pipeline(GpuQueue* queue, const GpuPipeline* pipeline, const std::optional<GpuDepthStencilDesc>& depth_stencil, const std::optional<GpuBlendDesc>& blend, INDEX_TYPE_EXT index_type) {
	auto& cache = std::get<GpuPipeline::RenderCache>(pipeline->cache);
	if(cache.pipeline) wgpuRenderPipelineRelease(cache.pipeline);

	auto& desc = cache.descriptor;

	// A dynamically applied blend state takes precedence over one baked into the rasterizer description
	auto effective_blend = blend ? blend : std::optional<GpuBlendDesc>(desc.blendstate);
	// MIN and MAX ignore the factors, which WebGPU insists are then ONE
	constexpr static auto component = [](BLEND op, FACTOR src, FACTOR dst) {
		bool ignored = op == BLEND_MIN || op == BLEND_MAX;
		return WGPUBlendComponent{
			.operation = GPU::blend2wgpu(op),
			.srcFactor = GPU::factor2wgpu(ignored ? FACTOR_ONE : src),
			.dstFactor = GPU::factor2wgpu(ignored ? FACTOR_ONE : dst),
		};
	};
	auto chosen = effective_blend.value_or(GpuBlendDesc{});
	WGPUBlendState blend_state {
		.color = component(chosen.colorOp, chosen.srcColorFactor, chosen.dstColorFactor),
		.alpha = component(chosen.alphaOp, chosen.srcAlphaFactor, chosen.dstAlphaFactor),
	};

	std::vector<WGPUColorTargetState> targets; targets.reserve(desc.colorTargets.size());
	for(auto& target: desc.colorTargets) {
		// The target's baked write mask and the blend state's write mask both have to be satisfied
		uint8_t mask = effective_blend ? (target.writeMask & effective_blend->colorWriteMask) : target.writeMask;
		targets.push_back({
			.format = GPU::format2wgpu(target.format),
			.blend = effective_blend ? &blend_state : nullptr,
			.writeMask = GPU::mask2wgpu(mask),
		});
	}

	auto depth_stencil_format = desc.depthFormat != FORMAT_NONE ? desc.depthFormat : desc.stencilFormat;
	auto state = depth_stencil.value_or(GpuDepthStencilDesc{});
	auto has_depth = gpuFormatIsDepthEXT(depth_stencil_format);
	auto has_stencil = gpuFormatIsStencilEXT(depth_stencil_format);

	auto stencil2wgpu = [](const GpuStencil& stencil) {
		return WGPUStencilFaceState {
			.compare = GPU::op2wgpu(stencil.test),
			.failOp = GPU::stencil2wgpu(stencil.failOp),
			.depthFailOp = GPU::stencil2wgpu(stencil.depthFailOp),
			.passOp = GPU::stencil2wgpu(stencil.passOp),
		};
	};
	// WebGPU rejects any non default state for an aspect the attachment format doesn't have
	constexpr static WGPUStencilFaceState stencil_disabled {
		.compare = WGPUCompareFunction_Always,
		.failOp = WGPUStencilOperation_Keep,
		.depthFailOp = WGPUStencilOperation_Keep,
		.passOp = WGPUStencilOperation_Keep,
	};
	WGPUDepthStencilState depth_stencil_state {
		.format = GPU::format2wgpu(depth_stencil_format),
		.depthWriteEnabled = (has_depth && (state.depthMode & DEPTH_WRITE)) ? WGPUOptionalBool_True : WGPUOptionalBool_False,
		.depthCompare = (has_depth && (state.depthMode & DEPTH_READ)) ? GPU::op2wgpu(state.depthTest) : WGPUCompareFunction_Always,
		.stencilFront = has_stencil ? stencil2wgpu(state.stencilFront) : stencil_disabled,
		.stencilBack = has_stencil ? stencil2wgpu(state.stencilBack) : stencil_disabled,
		.stencilReadMask = has_stencil ? static_cast<uint32_t>(state.stencilReadMask) : 0xFFFFFFFFu,
		.stencilWriteMask = has_stencil ? static_cast<uint32_t>(state.stencilWriteMask) : 0xFFFFFFFFu,
		// WebGPU counts the constant bias in units of the smallest representable depth value
		.depthBias = static_cast<int32_t>(state.depthBias),
		.depthBiasSlopeScale = state.depthBiasSlopeFactor,
		.depthBiasClamp = state.depthBiasClamp,
	};

	auto vertex_module = create_shader_module(queue, cache.vertexIR, GPU::shaders::SHADER_STAGE::VERTEX);
	auto fragment_module = cache.fragmentIR.empty() ? nullptr
		: create_shader_module(queue, cache.fragmentIR, GPU::shaders::SHADER_STAGE::FRAGMENT, desc.supportDualSourceBlending);

	WGPUFragmentState fragment {
		.module = fragment_module,
		// Undefined rather than left alone: a zeroed WGPUStringView is the empty string, which Dawn then goes looking for as an entry point name
		.entryPoint = WGPU_STRING_VIEW_INIT,
		.targetCount = targets.size(),
		.targets = targets.data(),
	};
	WGPURenderPipelineDescriptor d {
		.layout = queue->current_graphics_pipeline_layout,
		.vertex = {
			.module = vertex_module,
			// Undefined picks the module's only entry point for the stage, so one source can serve both stages.
			.entryPoint = WGPU_STRING_VIEW_INIT,
		},
		.primitive = {
			.topology = GPU::topology2wgpu(desc.topology),
			// Strips need to know how the index buffer restarts primitives, lists must not name a format
			.stripIndexFormat = desc.topology == TOPOLOGY_TRIANGLE_STRIP ? GPU::index2wgpu(index_type) : WGPUIndexFormat_Undefined,
			.frontFace = WGPUFrontFace_CCW,
			.cullMode = GPU::cull2wgpu(desc.cull),
		},
		.depthStencil = depth_stencil_format == FORMAT_NONE ? nullptr : &depth_stencil_state,
		.multisample = {
			.count = desc.sampleCount,
			.mask = 0xFFFFFFFF,
			.alphaToCoverageEnabled = desc.alphaToCoverage,
		},
		.fragment = fragment_module ? &fragment : nullptr,
	};
	cache.pipeline = wgpuDeviceCreateRenderPipeline(queue->device, &d);

	pipeline->reference_layout = queue->current_graphics_pipeline_layout;
	cache.depth_stencil = depth_stencil;
	cache.blend = blend;
	cache.index_type = index_type;

	wgpuShaderModuleRelease(vertex_module);
	if(fragment_module) wgpuShaderModuleRelease(fragment_module);
}

// Rebuilds the bound pipeline's variant when the state or layout has moved on, then sets it.
void bindRenderPipeline(GpuCommandBuffer* cmd) {
	assert(cmd->render_pass && "A render pass must be started before drawing");
	assert(cmd->bound_pipeline && "A graphics pipeline must be set before drawing");
	assert(std::holds_alternative<GpuPipeline::RenderCache>(cmd->bound_pipeline->cache) && "A compute pipeline can't be drawn with");

	auto pipeline = cmd->bound_pipeline;
	auto& cache = std::get<GpuPipeline::RenderCache>(pipeline->cache);

	if(!cache.pipeline
	|| pipeline->reference_layout != cmd->queue->current_graphics_pipeline_layout
	|| !GPU::same(cache.depth_stencil, cmd->depth_stencil)
	|| !GPU::same(cache.blend, cmd->blend)
	// The index format is only baked into the pipeline when primitives are restarted by the indices
	|| (cache.descriptor.topology == TOPOLOGY_TRIANGLE_STRIP && cache.index_type != cmd->index_type))
		update_render_pipeline(cmd->queue, pipeline, cmd->depth_stencil, cmd->blend, cmd->index_type);

	if(cmd->bound_render_pipeline != cache.pipeline) {
		wgpuRenderPassEncoderSetPipeline(cmd->render_pass, cache.pipeline);
		cmd->bound_render_pipeline = cache.pipeline;
	}

	// The stencil reference is the one piece of depth/stencil state WebGPU leaves dynamic
	if(cmd->depth_stencil)
		wgpuRenderPassEncoderSetStencilReference(cmd->render_pass, cmd->depth_stencil->stencilFront.reference);
}

struct GraphicsShaderData {
	gpu* vertex;
	gpu* fragment;
	gpu* indices;
	uint64_t _padding = 0;
};

void bindGraphicsGroups(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* index_data) {
	GraphicsShaderData shader_data { .vertex = vertex_data, .fragment = fragment_data, .indices = index_data };
	bindGroups(cmd, &shader_data, sizeof(shader_data), false);
}

namespace GPU::detail {
	// Monobuffers carry Index usage, so indices are drawn from where they were allocated.
	inline void bind_index_buffer(GpuCommandBuffer* cmd, gpu* indices, INDEX_TYPE_EXT index_type) {
		auto [range, offset, address] = GPU::detail::closest_buffer(cmd->queue, indices);
		assert(range.start + offset < range.end && "The indices point past the end of their allocation");

		auto stride = index_type == INDEX_TYPE_UINT8 ? 1u : index_type == INDEX_TYPE_UINT16 ? 2u : 4u;
		auto start = range.start + offset;
		// A misaligned start would shift every index that follows, not just slow the draw.
		assert(start % stride == 0 && "The index pointer isn't aligned to the size of one index");

		// WebGPU rejects a bound size that isn't a whole number of indices.
		auto size = ((range.end - start) / stride) * stride;
		wgpuRenderPassEncoderSetIndexBuffer(cmd->render_pass, cmd->queue->monobuffers[range.buffer],
			GPU::index2wgpu(index_type), start, size);
	}
}

namespace GPU::detail {
	// CULL_ALL leaves a draw nothing to rasterize, which is exactly what skipping it does; 8 bit indices can't be drawn at all.
	inline bool skips_draw(GpuCommandBuffer* cmd, INDEX_TYPE_EXT index_type, const char* function) {
		if(index_type == INDEX_TYPE_UINT8) {
			GPU::report(cmd, GPU_DIAGNOSTIC_UNSUPPORTED, std::string(function) + ": WebGPU has no 8 bit indices, so nothing is drawn");
			return true;
		}
		auto render = cmd->bound_pipeline ? std::get_if<GpuPipeline::RenderCache>(&cmd->bound_pipeline->cache) : nullptr;
		return render && render->descriptor.cull == CULL_ALL;
	}
}

void gpuDrawIndexedInstanced(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* indices, uint32_t index_count, uint32_t instance_count, INDEX_TYPE_EXT index_type) {
	assert(cmd->render_pass);
	assert(indices);
	if(GPU::detail::skips_draw(cmd, index_type, "gpuDrawIndexedInstanced")) return;

	cmd->index_type = index_type;
	bindRenderPipeline(cmd);
	bindGraphicsGroups(cmd, vertex_data, fragment_data, indices);
	GPU::detail::bind_index_buffer(cmd, indices, index_type);

	wgpuRenderPassEncoderDrawIndexed(cmd->render_pass, index_count, instance_count, 0, 0, 0);
}

// Matches both VkDrawIndexedIndirectCommand and WebGPU's indexed indirect argument layout
struct GpuDrawIndexedIndirectCommand {
	uint32_t indexCount;
	uint32_t instanceCount;
	uint32_t firstIndex;
	int32_t baseVertex;
	uint32_t firstInstance;
};

void gpuDrawIndexedInstancedIndirect(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* indices, gpu* args, INDEX_TYPE_EXT index_type) {
	assert(cmd->render_pass);
	assert(indices);
	assert(args);
	if(GPU::detail::skips_draw(cmd, index_type, "gpuDrawIndexedInstancedIndirect")) return;

	cmd->index_type = index_type;
	bindRenderPipeline(cmd);
	bindGraphicsGroups(cmd, vertex_data, fragment_data, indices);
	GPU::detail::bind_index_buffer(cmd, indices, index_type);

	auto [range, offset, address] = GPU::detail::closest_buffer(cmd->queue, args);
	auto start = range.start + offset;
	auto count = (range.end - start) / sizeof(GpuDrawIndexedIndirectCommand);

	// No multi draw indirect without an extension, so each argument struct is its own draw.
	for(size_t i = 0; i < count; ++i)
		wgpuRenderPassEncoderDrawIndexedIndirect(cmd->render_pass, cmd->queue->monobuffers[range.buffer], start + i * sizeof(GpuDrawIndexedIndirectCommand));
}

// Reported rather than thrown: an exception cannot leave an extern "C" function.
void gpuDrawMeshlets(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, uvec3 dim) {
	GPU::report(cmd, GPU_DIAGNOSTIC_UNSUPPORTED,
		"gpuDrawMeshlets: WebGPU has no mesh shaders, so nothing is drawn");
}

void gpuDrawMeshletsIndirect(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, gpu* dim) {
	GPU::report(cmd, GPU_DIAGNOSTIC_UNSUPPORTED,
		"gpuDrawMeshletsIndirect: WebGPU has no mesh shaders, so nothing is drawn");
}

void gpuSetEnabledSamplersEXT(GpuCommandBuffer* cmd, GpuSamplerDescSpan enabled_samplers) {
	// Nothing is recorded: the set only decides which groups the next dispatch or draw binds.
	cmd->sampler_set = GPU::detail::ensure_sampler_set(cmd->queue, enabled_samplers);
}

namespace GPU::detail {
	// Views recorded from the texture keep it alive, so they stay valid until their submission releases them.
	inline void release_surface_texture(GpuSurface* surface) {
		if(!surface->acquired) return;

		wgpuTextureRelease(surface->current.texture);
		surface->current.texture = nullptr;
		surface->acquired = false;
	}
}

void gpuFreeSurfaceEXT(GpuQueue* queue, GpuSurface* surface) {
	GPU::detail::release_surface_texture(surface);
	wgpuSurfaceUnconfigure(surface->surface);
	// The WGPUSurface belongs to whoever owns the window, as the VkSurfaceKHR does on Vulkan.

	auto allocator = queue->cpu_allocator;
	surface->~GpuSurface();
	allocator(surface, 0);
}

namespace GPU {
	// Only needed where WebGPU picks the format, which today means a surface reporting its preferred formats. Inverting format2wgpu keeps the two from drifting apart.
	inline FORMAT wgpu2format(WGPUTextureFormat format) {
		for(int named = FORMAT_NONE + 1; named <= FORMAT_D32_FLOAT_S8_UINT; ++named)
			if(GPU::format2wgpu((FORMAT)named) == format) return (FORMAT)named;
		return FORMAT_NONE;
	}

	// PRESENT_MODE_BEST_AVAILABLE has no WebGPU spelling, so it is resolved against what the surface supports before this is reached (see GPU::detail::pick_present_mode)
	inline WGPUPresentMode present2wgpu(PRESENT_MODE mode) {
		switch (mode) {
		case PRESENT_MODE_IMMEDIATE: return WGPUPresentMode_Immediate;
		case PRESENT_MODE_FIFO_RELAXED: return WGPUPresentMode_FifoRelaxed;
		case PRESENT_MODE_MAILBOX: return WGPUPresentMode_Mailbox;
		default: return WGPUPresentMode_Fifo;
		}
	}

	// Best first, so presentModes[0] of gpuGetSurfaceCapabilitiesEXT is what BEST_AVAILABLE resolves to; immediate trails the tear free modes.
	constexpr static PRESENT_MODE PRESENT_MODE_PREFERENCE[] = {
		PRESENT_MODE_MAILBOX, PRESENT_MODE_FIFO_RELAXED, PRESENT_MODE_FIFO, PRESENT_MODE_IMMEDIATE
	};

	// The surface was configured with one of these, by pick_present_mode
	inline PRESENT_MODE wgpu2present(WGPUPresentMode mode) {
		auto named = std::ranges::find_if(PRESENT_MODE_PREFERENCE, [mode](PRESENT_MODE named) { return present2wgpu(named) == mode; });
		return named != std::ranges::end(PRESENT_MODE_PREFERENCE) ? *named : PRESENT_MODE_FIFO;
	}
}

namespace GPU::detail {
	// Otherwise the best supported, in GPU::PRESENT_MODE_PREFERENCE's order.
	inline WGPUPresentMode pick_present_mode(PRESENT_MODE requested, const WGPUSurfaceCapabilities& caps) {
		auto supported = [&caps](WGPUPresentMode mode) {
			auto modes = std::span(caps.presentModes, caps.presentModeCount);
			return std::ranges::find(modes, mode) != modes.end();
		};

		if(requested != PRESENT_MODE_BEST_AVAILABLE)
			if(auto wanted = GPU::present2wgpu(requested); supported(wanted))
				return wanted;

		// Fifo is in the list and guaranteed to be supported, so the fallback is never needed
		auto best = std::ranges::find_if(GPU::PRESENT_MODE_PREFERENCE, [&](PRESENT_MODE mode) { return supported(GPU::present2wgpu(mode)); });
		return best != std::ranges::end(GPU::PRESENT_MODE_PREFERENCE) ? GPU::present2wgpu(*best) : WGPUPresentMode_Fifo;
	}

	// FORMAT_NONE takes the surface's preference, BGRA8 on most platforms and the only one on some.
	inline WGPUTextureFormat pick_surface_format(FORMAT requested, const WGPUSurfaceCapabilities& caps) {
		if(requested != FORMAT_NONE) {
			auto wanted = GPU::format2wgpu(requested);
			for(size_t i = 0; i < caps.formatCount; ++i)
				if(caps.formats[i] == wanted) return wanted;
		}

		return caps.formatCount ? caps.formats[0] : WGPUTextureFormat_BGRA8Unorm;
	}

	inline WGPUCompositeAlphaMode pick_alpha_mode(bool opaque, const WGPUSurfaceCapabilities& caps) {
		auto wanted = opaque ? WGPUCompositeAlphaMode_Opaque : WGPUCompositeAlphaMode_Premultiplied;
		// Auto aliases the surface's first supported mode and is never listed as one itself
		auto modes = std::span(caps.alphaModes, caps.alphaModeCount);
		return std::ranges::find(modes, wanted) != modes.end() ? wanted : WGPUCompositeAlphaMode_Auto;
	}
}

void gpuSurfaceReconfigureEXT(GpuQueue* queue, GpuSurface* surface, const GpuSurfaceDescriptor& desc) {
	// Whatever is still acquired belongs to the configuration being replaced
	GPU::detail::release_surface_texture(surface);

	WGPUSurfaceCapabilities caps = {};
	if(wgpuSurfaceGetCapabilities(surface->surface, queue->adapter, &caps) != WGPUStatus_Success) {
		errno = WGPUErrorType_Unknown;
		return;
	}

	surface->descriptor = desc;
	auto& texture = surface->descriptor.texture;

	// One 2D image without mips or multisampling; resolve or blit into it instead.
	texture.type = TEXTURE_2D;
	texture.mipCount = 1;
	texture.sampleCount = 1;
	texture.layerCount = 1;
	texture.dimensions.z = 1;
	assert(texture.dimensions.x > 0 && texture.dimensions.y > 0 && "A surface can't be configured for a zero sized window");

	auto format = GPU::detail::pick_surface_format(texture.format, caps);
	auto present_mode = GPU::detail::pick_present_mode(surface->descriptor.presentMode, caps);
	auto alpha_mode = GPU::detail::pick_alpha_mode(surface->descriptor.opaque, caps);

	// RenderAttachment is universal; Sampled helps gpuBlitTextureEXT read a frame. emdawnwebgpu leaves usages zero, meaning unreported rather than none.
	auto supported_usages = caps.usages ? caps.usages : WGPUTextureUsage(WGPUTextureUsage_RenderAttachment);
	auto usage = (GPU::usage2wgpu(texture.usage) | WGPUTextureUsage_RenderAttachment) & supported_usages;

	WGPUSurfaceConfiguration config {
		.device = queue->device,
		.format = format,
		.usage = usage,
		.width = texture.dimensions.x,
		.height = texture.dimensions.y,
		.alphaMode = alpha_mode,
		.presentMode = present_mode,
	};
	wgpuSurfaceConfigure(surface->surface, &config);
	wgpuSurfaceCapabilitiesFreeMembers(caps);

	// What was granted rather than requested, so a pipeline can match the swapchain format.
	texture.format = GPU::wgpu2format(format);
	texture.usage = (TEXTURE_USAGE_FLAGS)(USAGE_COLOR_ATTACHMENT
		| ((usage & WGPUTextureUsage_TextureBinding) ? USAGE_SAMPLED : 0)
		| ((usage & WGPUTextureUsage_StorageBinding) ? USAGE_STORAGE : 0)
		| ((usage & WGPUTextureUsage_CopySrc) ? USAGE_TRANSFER_SRC : 0)
		| ((usage & WGPUTextureUsage_CopyDst) ? USAGE_TRANSFER_DST : 0));
	surface->descriptor.presentMode = GPU::wgpu2present(present_mode);
	surface->descriptor.opaque = alpha_mode == WGPUCompositeAlphaMode_Opaque
		|| (alpha_mode == WGPUCompositeAlphaMode_Auto && surface->descriptor.opaque);
}

GpuSurfaceCapabilities gpuGetSurfaceCapabilitiesEXT(GpuQueue* queue, GpuSurface* surface) {
	GpuSurfaceCapabilities out = {};

	WGPUSurfaceCapabilities caps = {};
	if(wgpuSurfaceGetCapabilities(surface->surface, queue->adapter, &caps) != WGPUStatus_Success) {
		errno = WGPUErrorType_Unknown;
		return out;
	}

	// WebGPU lists formats in preference order, so formats[0] is the FORMAT_NONE pick; formats can repeat, and the first sighting counts.
	for(size_t i = 0; i < caps.formatCount && out.formatCount < GPU_MAX_SURFACE_FORMATS; ++i) {
		auto format = GPU::wgpu2format(caps.formats[i]);
		// A format the rest of the API can't name is one it could never be configured with either
		if(format == FORMAT_NONE) continue;
		auto listed = std::span(out.formats, out.formatCount);
		if(std::ranges::find(listed, format) == listed.end())
			out.formats[out.formatCount++] = format;
	}

	for(auto mode: GPU::PRESENT_MODE_PREFERENCE) {
		if(out.presentModeCount >= GPU_MAX_SURFACE_PRESENT_MODES) break;

		auto wanted = GPU::present2wgpu(mode);
		for(size_t i = 0; i < caps.presentModeCount; ++i)
			if(caps.presentModes[i] == wanted) {
				out.presentModes[out.presentModeCount++] = mode;
				break;
			}
	}

	// Auto is whatever the surface does by default, which is not a promise that it blends
	for(size_t i = 0; i < caps.alphaModeCount; ++i)
		if(caps.alphaModes[i] == WGPUCompositeAlphaMode_Premultiplied
			|| caps.alphaModes[i] == WGPUCompositeAlphaMode_Unpremultiplied)
			out.supportsTransparency = true;

	wgpuSurfaceCapabilitiesFreeMembers(caps);
	return out;
}

GpuSurfaceDescriptor gpuSurfaceGetConfigurationEXT(const GpuSurface* surface) {
	return surface->descriptor;
}

const GpuTexture* gpuSurfaceNextTextureEXT(GpuQueue* queue, GpuSurface* surface) {
	if(surface->acquired) {
		GPU::report(queue, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuSurfaceNextTextureEXT: a texture is already acquired and not yet presented, so no new one is handed out; present the one you have first");
		errno = WGPUErrorType_Validation;
		return nullptr;
	}

	WGPUSurfaceTexture acquired = {};
	wgpuSurfaceGetCurrentTexture(surface->surface, &acquired);

	switch(acquired.status) {
	break; case WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal:
		// Nothing to report
	break; case WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal:
		// The texture no longer matches the window but can still be rendered into and presented. The caller decides whether that is worth a reconfiguration.
		errno = SURFACE_SUBOPTIMAL;
	break; default:
		// Outdated, Lost, Timeout and Error all leave nothing to render into
		errno = acquired.status;
		if(acquired.texture) wgpuTextureRelease(acquired.texture);
		return nullptr;
	}

	// The texture's own size, in case a platform ever rounds the configuration.
	surface->current = GpuTexture {
		.descriptor = surface->descriptor.texture,
		// Deliberately no monotexture range. Everything that consumes a texture through the descriptor heap therefore can't touch this one; gpuBlitTextureEXT is the bridge in both directions.
		.range = {},
		.texture = acquired.texture,
	};
	surface->current.descriptor.dimensions = {
		wgpuTextureGetWidth(acquired.texture),
		wgpuTextureGetHeight(acquired.texture),
		1
	};
	surface->acquired = true;
	return &surface->current;
}

void gpuSurfacePresentEXT(GpuQueue* queue, GpuSurface* surface, uint64_t wait_submission_index) {
	if(!surface->acquired) {
		GPU::report(queue, GPU_DIAGNOSTIC_UNSUPPORTED,
			"gpuSurfacePresentEXT: no texture is currently acquired, so nothing is presented; call gpuSurfaceNextTextureEXT first");
		return;
	}

	// One queue, presenting behind everything submitted, so there is nothing to wait for; the index is only checked.
	assert((wait_submission_index == NO_SUBMISSION_WAIT || wait_submission_index < queue->next_submission_index)
		&& "Presenting behind a submission that hasn't happened yet would deadlock on a backend that waits");

	GPU::detail::release_surface_texture(surface);
#ifndef __EMSCRIPTEN__
	// In the browser the canvas is presented when control returns to the event loop, and calling this is an error there
	if(wgpuSurfacePresent(surface->surface) != WGPUStatus_Success)
		errno = WGPUErrorType_Unknown;
#endif
}

const GpuSemaphore* gpuGetSubmissionSemaphoreEXT(GpuQueue* queue) {
	return (GpuSemaphore*)&queue->current_submission_timeline_semaphore;
}

void gpuWaitIdleEXT(GpuQueue* queue) {
	struct Wait { volatile bool done = false; } wait;

    wgpuQueueOnSubmittedWorkDone(
        queue->queue,
		WGPUQueueWorkDoneCallbackInfo {
			.mode = WGPUCallbackMode_AllowSpontaneous,
#ifdef __EMSCRIPTEN__
			.callback = [](WGPUQueueWorkDoneStatus status, WGPU_NULLABLE void* userdata1, WGPU_NULLABLE void* userdata2) {
				static_cast<Wait*>(userdata1)->done = true;
			},
#else
			.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView message, WGPU_NULLABLE void* userdata1, WGPU_NULLABLE void* userdata2) {
				static_cast<Wait*>(userdata1)->done = true;
			},
#endif
        	.userdata1 = &wait
		}
	);

    while (!wait.done) {
#ifdef __EMSCRIPTEN__
		emscripten_sleep(1); // yields back to the browser event loop
#else
		wgpuDeviceTick(queue->device);
#endif
    }
	GPU::process_pending_code(queue);
}

namespace GPU {
	inline void pull_from_monobuffer(GpuCommandBuffer* cmd, GpuQueue::MonobufferRange range, void* cpu) {
		auto size = range.size();

		WGPUBufferDescriptor d{
			.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
			.size = size,
		};
		auto tmp = wgpuDeviceCreateBuffer(cmd->queue->device, &d);

		wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, cmd->queue->monobuffers[range.buffer], range.start, tmp, 0, size);
		cmd->code_pending_submission_finished.emplace_back([queue = cmd->queue, tmp, cpu, size]() {
			GPU::wait_for_buffer_map(queue, tmp, WGPUMapMode_Read, 0, size);

			auto tmp_cpu = wgpuBufferGetConstMappedRange(tmp, 0, size);
			memcpy(cpu, tmp_cpu, size);

			wgpuBufferRelease(tmp);
		});
	}

	inline void pull_from_monobuffer(GpuQueue* queue, GpuQueue::MonobufferRange range, void* cpu) {
		auto cmd = gpuStartCommandRecording(queue);
		pull_from_monobuffer(cmd, range, cpu);
		auto submit_index = gpuSubmit(queue, {&cmd, 1});

		// When we pull it seems much more likely that we want the cpu memory updated before considering our work "done"
		GPU::semaphore_wait(queue, queue->current_submission_timeline_semaphore, submit_index);
		GPU::process_pending_code(queue);
	}
}

void gpuSyncMemoryEXT(GpuCommandBuffer* cmd, gpu* mem) {
	endCurrentPass(cmd);
	auto found = cmd->queue->allocations.find(mem);
	if(found == cmd->queue->allocations.end()) return;
	auto [range, cpu, memory_type] = found->second;

	switch (memory_type) {
	case MEMORY_GPU:
	case MEMORY_TEXTURE:
		break;
	// Nothing here can be bound as a descriptor heap in place, so this is MEMORY_DEFAULT under another name
	case MEMORY_DEFAULT:
	case MEMORY_DESCRIPTOR_HEAP:
		GPU::push_to_monobuffer(cmd, range, cpu);
		break;
	case MEMORY_READBACK:
	case MEMORY_TEXTURE_READBACK:
		GPU::pull_from_monobuffer(cmd, range, cpu);
		break;
	}
}

void gpuSyncMemoryImmediateEXT(GpuQueue* queue, gpu* mem) {
	auto cmd = gpuStartCommandRecording(queue);
	gpuSyncMemoryEXT(cmd, mem);
	auto submit_index = gpuSubmit(queue, {&cmd, 1});

	GPU::semaphore_wait(queue, queue->current_submission_timeline_semaphore, submit_index);
	GPU::process_pending_code(queue);
}
