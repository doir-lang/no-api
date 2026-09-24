#include "noapi.hpp"
#include "common.hpp"

#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <variant>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

const GpuQueue::MonotextureRange GpuQueue::MonotextureRange::INVALID = {static_cast<uint32_t>(-1), static_cast<uint32_t>(-1), static_cast<uint32_t>(-1)};

void GPU::default_::error_callback(void* queue, int type, std::string_view message) {
	std::cerr << "WGPU Device Error " << type << ": " << message << std::endl;
#ifdef _WIN32
	system("pause");
#endif
	exit(-1);
}

std::expected<GpuWebGPUDefault, std::string> gpuSetupDefaultWebGPUEXT(GPU::function_t<WGPUSurface(WGPUInstance)> surface_loader, void(*error_callback)(void* queue, int type, std::string_view message) /* = GPU::default_::error_callback */, bool prefer_high_power /* = true */){
	GpuWebGPUDefault out;

	{ // Instance
		WGPUInstanceDescriptor d {

		};
		out.instance = wgpuCreateInstance(&d);
	}
	if(!out.instance) return std::unexpected("Failld to create WebGPU instance");

	out.surface = surface_loader(out.instance);

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
				auto error_callback = *(void(**)(void* queue, int type, std::string_view message))userdata2;
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
	if(!out.adapter) return std::unexpected("Failed to create WebGPU adapter");

	out.limits = {}; // Emscripten apparently asserts that the limits have been zeroed out!
	wgpuAdapterGetLimits(out.adapter, &out.limits);

	{ // Device
		WGPUDeviceDescriptor device_desc = {
			.requiredLimits = &out.limits,
			.deviceLostCallbackInfo = {
				.mode = WGPUCallbackMode_AllowSpontaneous,
				.callback = +[](WGPUDevice const* device, WGPUDeviceLostReason reason, WGPUStringView message, void* userdata1, void* _) {
					if(reason == WGPUDeviceLostReason_Destroyed) return;

					auto callback = (void(*)(WGPUDevice const* device, WGPUErrorType type, std::string_view message))userdata1;
					callback(device, (WGPUErrorType)reason, {message.data, message.length});
				},
				.userdata1 = (void*)error_callback
			},
			.uncapturedErrorCallbackInfo = {
				.callback = +[](WGPUDevice const* device, WGPUErrorType type, WGPUStringView message, void* userdata1, void* _){
					auto callback = (void(*)(WGPUDevice const* device, WGPUErrorType type, std::string_view message))userdata1;
					callback(device, type, {message.data, message.length});
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
				auto& error_callback = *(GPU::function_t<void(WGPUDevice const* device, WGPUErrorType type, std::string_view message)>*)userdata2;
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
	if(!out.device) return std::unexpected("Failed to create WebGPU device");

	return out;
}

// Group 0 describes the monobuffers, the active texture heap, the sampler/texture metadata buffer,
// and the shader data uniform. Its layout never changes, but compute and graphics need separate ones
// since WebGPU forbids writable storage buffers in the vertex stage (so graphics binds the
// monobuffers read only).
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

// Group 3 is one binding per sampler slot. Its shape is fixed, so unlike the texture groups it is
// built once and shared by every sampler set.
WGPUBindGroupLayout create_sampler_bind_group_layout(GpuQueue* queue) {
	std::vector<WGPUBindGroupLayoutEntry> group3;
	for(uint32_t i = 0; i < gpu_sampler_slot_count; ++i)
		group3.push_back({
			.binding = i,
			.visibility = WGPUShaderStage_Compute | WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
			.sampler = {
				// Filtering accepts a nearest sampler too, and group 2 already declares every sampled
				// monotexture as a filterable float texture
				.type = WGPUSamplerBindingType_Filtering,
			}
		});

	WGPUBindGroupLayoutDescriptor d{
		.entryCount = static_cast<uint32_t>(group3.size()),
		.entries = group3.data(),
	};
	return wgpuDeviceCreateBindGroupLayout(queue->device, &d);
}

namespace GPU::detail {
	// Sampler descriptions pack into 16 bits but only ever reach max_packed(), so the lookup map
	// needs one word per value in [0, max_packed()]
	constexpr static uint32_t sampler_lookup_size = GpuSamplerDesc::max_packed() + 1;

	constexpr static WGPUAddressMode address2wgpu(ADDRESS_MODE mode) {
		switch (mode) {
		case ADDRESS_MODE_CLAMP: return WGPUAddressMode_ClampToEdge;
		case ADDRESS_MODE_MIRROR_REPEAT: return WGPUAddressMode_MirrorRepeat;
		case ADDRESS_MODE_REPEAT: return WGPUAddressMode_Repeat;
		}
		return WGPUAddressMode_Repeat;
	}

	constexpr static WGPUFilterMode filter2wgpu(FILTER filter) {
		return filter == FILTER_NEAREST ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear;
	}

	constexpr static WGPUMipmapFilterMode mip_filter2wgpu(FILTER filter) {
		return filter == FILTER_NEAREST ? WGPUMipmapFilterMode_Nearest : WGPUMipmapFilterMode_Linear;
	}

	// Looks up (or builds) the sampler set for a list of enabled samplers. Slot 0 is always the
	// default sampler, matching the Vulkan backend, so a packed description that was never enabled
	// still resolves to something valid.
	inline GpuQueue::SamplerSet* ensure_sampler_set(GpuQueue* queue, std::span<const GpuSamplerDesc> requested) {
		// One slot is spent on the default sampler, so the caller gets the rest
		assert(requested.size() < gpu_sampler_slot_count && "More samplers enabled at once than there are sampler slots");
		auto count = std::min<size_t>(requested.size(), gpu_sampler_slot_count - 1);

		std::vector<GpuSamplerDesc> enabled(count + 1, GpuSamplerDesc{});
		std::copy_n(requested.begin(), count, enabled.begin() + 1);

		if(auto found = queue->sampler_cache.find(enabled); found != queue->sampler_cache.end())
			return &found->second;

		auto& set = queue->sampler_cache[enabled];

		// Every slot has to be filled for the bind group to match the layout, so the ones the caller
		// didn't ask for get a copy of the default sampler
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
			lookup[enabled[i].pack()] = i;

		WGPUBufferDescriptor buffer {
			.label = {"NoAPI Sampler Map", WGPU_STRLEN},
			.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
			.size = lookup.size() * sizeof(uint32_t),
		};
		set.lookup_buffer = wgpuDeviceCreateBuffer(queue->device, &buffer);
		wgpuQueueWriteBuffer(queue->queue, set.lookup_buffer, 0, lookup.data(), buffer.size);

		return &set;
	}

	inline void free_sampler_set(GpuQueue::SamplerSet& set) {
		if(set.lookup_buffer) wgpuBufferRelease(set.lookup_buffer);
		if(set.bind_group) wgpuBindGroupRelease(set.bind_group);
		for(auto sampler: set.samplers)
			wgpuSamplerRelease(sampler);
	}
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

	//
	// Group 0: buffers (one flavor per pipeline type, neither of which is dynamic)
	//
	if(!queue->current_compute_bind_group_layout0)
		queue->current_compute_bind_group_layout0 = create_buffer_bind_group_layout(queue, true);
	if(!queue->current_graphics_bind_group_layout0)
		queue->current_graphics_bind_group_layout0 = create_buffer_bind_group_layout(queue, false);

	//
	// Group 1: storage textures
	//
	// Groups 1 and 2 are shared between the compute and graphics pipeline layouts (the bind groups
	// built from them get bound to both), so their visibility has to name every stage allowed to
	// touch them. Writable storage textures are illegal in the vertex stage, hence the asymmetry.
	//
	std::vector<WGPUBindGroupLayoutEntry> group1;
	std::vector<WGPUBindGroupEntry> group1entries;

	uint32_t binding = 0;

	for(auto const& [_cap, texture, view, desc] : queue->storage_monotextures) {
		group1.push_back({
			.binding = binding,
			.visibility = WGPUShaderStage_Compute | WGPUShaderStage_Fragment,
			.storageTexture = {
				.access = WGPUStorageTextureAccess_ReadWrite,
				.format = GPU::format2wgpu(desc.format),
				.viewDimension = GPU::texture_view2wgpu(desc.type),
			}
		});
		group1entries.push_back({
			.binding = binding++,
			.textureView = view,
		});
	}

	//
	// Group 2: sampled textures
	//
	std::vector<WGPUBindGroupLayoutEntry> group2;
	std::vector<WGPUBindGroupEntry> group2entries;

	binding = 0;

	for(auto const& [_cap, texture, view, desc] : queue->sampled_monotextures) {
		group2.push_back({
			.binding = binding,
			.visibility = WGPUShaderStage_Compute | WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
			.texture = {
				.sampleType = WGPUTextureSampleType_Float,
				.viewDimension = GPU::texture_view2wgpu(desc.type),
				.multisampled = desc.sampleCount > 1,
			}
		});
		group2entries.push_back({
			.binding = binding++,
			.textureView = view,
		});
	}

	if(queue->current_bind_group_layout1) queue->code_pending_submission_finished.emplace_back([l = queue->current_bind_group_layout1, gp = queue->current_bind_group1](){ 
		wgpuBindGroupLayoutRelease(l); 
		wgpuBindGroupRelease(gp); 
	}, queue->next_submission_index);
	std::tie(queue->current_bind_group_layout1, queue->current_bind_group1) = create_layout_and_group(group1, group1entries);

	if(queue->current_bind_group_layout2) queue->code_pending_submission_finished.emplace_back([l = queue->current_bind_group_layout2, gp = queue->current_bind_group2](){ 
		wgpuBindGroupLayoutRelease(l); 
		wgpuBindGroupRelease(gp); 
	}, queue->next_submission_index);
	std::tie(queue->current_bind_group_layout2, queue->current_bind_group2) = create_layout_and_group(group2, group2entries);

	//
	// Group 3: samplers
	//
	// Only its contents ever change, never its shape, so the layout outlives every sampler set
	//
	if(!queue->current_bind_group_layout3)
		queue->current_bind_group_layout3 = create_sampler_bind_group_layout(queue);

	auto create_pipeline_layout = [&](WGPUBindGroupLayout bgl0) {
		WGPUBindGroupLayout layouts[] = {
			bgl0,
			queue->current_bind_group_layout1,
			queue->current_bind_group_layout2,
			queue->current_bind_group_layout3
		};

		WGPUPipelineLayoutDescriptor pd{
			.bindGroupLayoutCount = 4,
			.bindGroupLayouts = layouts,
		};
		return wgpuDeviceCreatePipelineLayout(queue->device, &pd);
	};

	if(queue->current_compute_pipeline_layout) wgpuPipelineLayoutRelease(queue->current_compute_pipeline_layout);
	queue->current_compute_pipeline_layout = create_pipeline_layout(queue->current_compute_bind_group_layout0);

	if(queue->current_graphics_pipeline_layout) wgpuPipelineLayoutRelease(queue->current_graphics_pipeline_layout);
	queue->current_graphics_pipeline_layout = create_pipeline_layout(queue->current_graphics_bind_group_layout0);
}



std::string generate_binding_prologue(GpuQueue* queue, bool compute) {
	// Has to name the same view dimension GPU::texture_view2wgpu puts in the bind group layouts, or
	// the declaration won't match the view that gets bound. A plain 2D texture is still an array
	// view, since that is how several of them get packed into one monotexture.
	constexpr static auto texture_dimension_suffix = [](TEXTURE type) {
		switch (GPU::texture_view2wgpu(type)) {
		case WGPUTextureViewDimension_1D: return "1d";
		case WGPUTextureViewDimension_3D: return "3d";
		case WGPUTextureViewDimension_Cube: return "cube";
		case WGPUTextureViewDimension_CubeArray: return "cube_array";
		case WGPUTextureViewDimension_2DArray: return "2d_array";
		default: return "2d";
		}
	};
	constexpr static auto wgsl_format = [](FORMAT f) {
		switch (f) {
		case FORMAT_RGBA8_UNORM:
			return "rgba8unorm";
		case FORMAT_RGBA8_SRGB:
			return "rgba8snorm";
		case FORMAT_RGBA16_FLOAT:
			return "rgba16float";
		case FORMAT_RGBA32_FLOAT:
			return "rgba32float";
		// case FORMAT_RG11B10_FLOAT:
		// case FORMAT_RGB10_A2_UNORM:
		// case FORMAT_R8_UNORM:
		// case FORMAT_R16_FLOAT:
		// case FORMAT_R32_FLOAT:
		// case FORMAT_D16_UNORM:
		// case FORMAT_D24_UNORM_S8_UINT:
		// case FORMAT_D32_FLOAT:
		// case FORMAT_D32_FLOAT_S8_UINT:
		default:
			assert(false && "Unsupported storage texture format");
			return "rgba8unorm";
		}
	};

	// A texture only owns part of the monotexture holding it, so its [0, 1] coordinates have to be
	// scaled down by its share of the atlas before they are sampled. Cube coordinates are a direction
	// rather than a position, so they pass through untouched.
	constexpr static auto texture_sample_remap = [](TEXTURE type) {
		switch (GPU::texture_view2wgpu(type)) {
		case WGPUTextureViewDimension_1D: return "uv.x * f32(size.x) / f32(textureDimensions(sampled_tex_{0}, 0))";
		case WGPUTextureViewDimension_3D: return "uv * vec3<f32>(size) / vec3<f32>(textureDimensions(sampled_tex_{0}, 0))";
		case WGPUTextureViewDimension_Cube:
		case WGPUTextureViewDimension_CubeArray: return "uv";
		default: return "uv.xy * vec2<f32>(size.xy) / vec2<f32>(textureDimensions(sampled_tex_{0}, 0))";
		}
	};

	// textureSampleLevel's argument list depends on the view dimension too, so every per slot wrapper
	// spells its own call out. The dimensions that have no array index simply drop it.
	constexpr static auto texture_sample_arguments = [](TEXTURE type) {
		switch (GPU::texture_view2wgpu(type)) {
		case WGPUTextureViewDimension_2DArray:
		case WGPUTextureViewDimension_CubeArray: return "at, layer, mip";
		default: return "at, mip";
		}
	};

	// WebGPU forbids writable storage buffers in the vertex stage, so the rasterizer only ever gets
	// read only access to the monobuffers (matching what create_buffer_bind_group_layout declares)
	std::string out;
	uint32_t binding = 0;
	for(uint32_t i = 0; i < queue->monobuffers.size(); ++i)
		out += std::format("@group(0) @binding({}) var<storage, {}> mono{} : array<u32>;\n", binding++, compute ? "read_write" : "read", i);

	out += std::format("\n"
		"@group(0) @binding({}) var<storage> texture_heap : array<u32>;\n"
		"@group(0) @binding({}) var<storage> noapi_sampler_map : array<u32>;\n"
		"\n", binding, binding + 1);
	binding += 2;

	out += std::string(compute ?
		"struct GPUShaderData {\n"
		"	compute: vec2<u32>,\n"
		"}\n"
		: "struct GPUShaderData {\n"
		"	vertex: vec2<u32>,\n"
		"	fragment: vec2<u32>,\n"
		"}\n")
	+ std::format("@group(0) @binding({}) var<uniform> shader_data : GPUShaderData;\n\n", binding++);

	binding = 0;

	for (auto const& [_cap, texture, view, desc] : queue->storage_monotextures) {
		out += std::format("@group(1) @binding({}) var storage_tex_{} : texture_storage_{}<{}, read_write>;\n", binding, binding, texture_dimension_suffix(desc.type), wgsl_format(desc.format));
		++binding;
	}

	binding = 0;

	for (auto const& [_cap, texture, view, desc] : queue->sampled_monotextures) {
		out += std::format("@group(2) @binding({}) var sampled_tex_{} : texture_{}<f32>;\n", binding, binding, texture_dimension_suffix(desc.type));
		++binding;
	}

	//
	// Samplers, and the metadata buffer that makes them addressable
	//
	out += "\n";
	for (uint32_t i = 0; i < gpu_sampler_slot_count; ++i)
		out += std::format("@group(3) @binding({}) var noapi_sampler_{} : sampler;\n", i, i);

	auto storage_count = static_cast<uint32_t>(queue->storage_monotextures.size());
	auto sampled_count = static_cast<uint32_t>(queue->sampled_monotextures.size());

	out += std::format("\n"
		"const NOAPI_SAMPLER_SLOT_COUNT : u32 = {}u;\n"
		"const NOAPI_SAMPLER_LOOKUP_SIZE : u32 = {}u;\n"
		"const NOAPI_STORAGE_TEXTURE_COUNT : u32 = {}u;\n"
		"const NOAPI_SAMPLED_TEXTURE_COUNT : u32 = {}u;\n"
		"\n"
		"// Turns a packed GpuSamplerDesc into the slot holding it, or slot 0 (the default sampler)\n"
		"// when that description was never handed to gpuSetEnabledSamplersEXT\n"
		"fn noapi_sampler_slot(packed : u32) -> u32 {{\n"
		"	if (packed >= NOAPI_SAMPLER_LOOKUP_SIZE) {{ return 0u; }}\n"
		"	return noapi_sampler_map[packed];\n"
		"}}\n"
		"\n"
		"const NOAPI_MONOTEXTURE_STORAGE_BIT : u32 = 0x80000000u;\n"
		"const NOAPI_MONOTEXTURE_INDEX_MASK : u32 = 0x7FFFFFFFu;\n"
		"\n"
		"// One entry of the texture heap, as gpuTextureViewDescriptor lays it out: eight words, the\n"
		"// first of which packs three bytes. `size` is the texture's own, not the monotexture's --\n"
		"// WGSL can report the monotexture's itself, but not how much of it this texture owns.\n"
		"struct GPUTextureDescriptor {{\n"
		"	texture_type : u32,\n"
		"	base_mip : u32,\n"
		"	mip_count : u32,\n"
		"	size : vec3<u32>, // width, height, depth\n"
		"	monotexture : u32, // Top bit set means a storage monotexture, the rest is the index\n"
		"	first_layer : u32, // Absolute layer within that monotexture\n"
		"	end_layer : u32,\n"
		"}}\n"
		"\n"
		"fn noapi_texture_descriptor(heap_index : u32) -> GPUTextureDescriptor {{\n"
		"	let at = heap_index * 8u;\n"
		"	let packed = texture_heap[at];\n"
		"	return GPUTextureDescriptor(packed & 0xFFu, (packed >> 8u) & 0xFFu, (packed >> 16u) & 0xFFu,\n"
		"		vec3<u32>(texture_heap[at + 1u], texture_heap[at + 2u], texture_heap[at + 3u]),\n"
		"		texture_heap[at + 4u], texture_heap[at + 5u], texture_heap[at + 6u]);\n"
		"}}\n"
		"\n"
		"fn noapi_monotexture_index(descriptor : GPUTextureDescriptor) -> u32 {{\n"
		"	return descriptor.monotexture & NOAPI_MONOTEXTURE_INDEX_MASK;\n"
		"}}\n"
		"\n"
		"fn noapi_monotexture_is_storage(descriptor : GPUTextureDescriptor) -> bool {{\n"
		"	return (descriptor.monotexture & NOAPI_MONOTEXTURE_STORAGE_BIT) != 0u;\n"
		"}}\n"
		"\n",
		gpu_sampler_slot_count, GPU::detail::sampler_lookup_size, storage_count, sampled_count);

	// WGSL can't index an array of samplers or of textures, so the only way to pick either one at
	// runtime is a switch over every combination the queue currently has
	binding = 0;
	for (auto const& [_cap, texture, view, desc] : queue->sampled_monotextures) {
		auto arguments = texture_sample_arguments(desc.type);
		auto remap = std::vformat(texture_sample_remap(desc.type), std::make_format_args(binding));
		out += std::format("fn noapi_sample_tex_{}(slot : u32, size : vec3<u32>, uv : vec3<f32>, layer : u32, mip : f32) -> vec4<f32> {{\n"
			"	let at = {};\n"
			"	switch slot {{\n", binding, remap);
		for (uint32_t i = 1; i < gpu_sampler_slot_count; ++i)
			out += std::format("		case {}u: {{ return textureSampleLevel(sampled_tex_{}, noapi_sampler_{}, {}); }}\n", i, binding, i, arguments);
		out += std::format("		default: {{ return textureSampleLevel(sampled_tex_{}, noapi_sampler_0, {}); }}\n"
			"	}}\n"
			"}}\n"
			"\n", binding, arguments);
		++binding;
	}

	out += "// Samples monotexture `texture` on `layer`, through the sampler in `slot` (see\n"
		"// noapi_sampler_slot). `uv` runs 0..1 over a texture of `size`, which is scaled down onto\n"
		"// whatever part of the monotexture that texture occupies. `layer` and `mip` are absolute.\n"
		"fn noapi_sample_texture(texture : u32, slot : u32, size : vec3<u32>, uv : vec3<f32>, layer : u32, mip : f32) -> vec4<f32> {\n"
		"	switch texture {\n";
	for (uint32_t i = 0; i < sampled_count; ++i)
		out += std::format("		case {}u: {{ return noapi_sample_tex_{}(slot, size, uv, layer, mip); }}\n", i, i);
	out += "		default: { return vec4<f32>(0.0); }\n"
		"	}\n"
		"}\n"
		"\n"
		"// Samples a texture straight out of the heap. `uv` runs 0..1 over the texture, and `layer`\n"
		"// and `mip` are relative to the view the descriptor describes rather than to the monotexture.\n"
		"fn noapi_sample(heap_index : u32, slot : u32, uv : vec3<f32>, layer : u32, mip : f32) -> vec4<f32> {\n"
		"	let descriptor = noapi_texture_descriptor(heap_index);\n"
		"	let level = f32(descriptor.base_mip) + clamp(mip, 0.0, max(f32(descriptor.mip_count), 1.0) - 1.0);\n"
		"	let at = descriptor.first_layer + min(layer, descriptor.end_layer - descriptor.first_layer - 1u);\n"
		"	return noapi_sample_texture(noapi_monotexture_index(descriptor), slot, descriptor.size, uv, at, level);\n"
		"}\n";

	return out;
}

GpuQueue* gpuCreateQueue(WGPUAdapter adapter, WGPUDevice device, WGPULimits limits, CpuAllocatorFunc allocator /* = default_::gpu_allocator */) {
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

	// A command buffer that never calls gpuSetEnabledSamplersEXT still has to bind something, and
	// every set resolves an unknown description to slot 0 anyway, so the default is a set of one
	out->default_sampler_set = GPU::detail::ensure_sampler_set(out, {});
	return out;
}

void gpuFreeQueue(GpuQueue* queue) {
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

constexpr static uint64_t gpu_address_max = 0x1FFFFFFFFFFFFFFF; // (2^61 - 1) aka max number storable in 60 bits

// top 3 bits encode monobuffer, rest encodes address
gpu* gpuEncodeWebGPUAddressEXT(uint8_t monobuffer, uint64_t address) {
	assert(monobuffer < GpuQueue{}.monobuffers.size());
	assert(address <= gpu_address_max);

	uint64_t debug = uint64_t(monobuffer + 1) << 61 | address;
	return (gpu*)debug;
}
std::pair<uint8_t, uint64_t> gpuDecodeWebGPUAddressEXT(gpu* addr) {
	auto address = (uint64_t)addr;
	return {(address >> 61) - 1, address & gpu_address_max};
}

void* gpuMalloc(GpuQueue* queue, size_t bytes, size_t align /* = 16 */, MEMORY memory /* = MEMORY_DEFAULT */) {
	constexpr static auto align_up = [](size_t addr, size_t align) {
		return (addr + align - 1) & ~(align - 1);
	};
	constexpr static auto allocation_bookkeeping = [](GpuQueue* queue, uint8_t active_monobuffer, size_t gpu_address, size_t bytes, MEMORY memory) {
		auto cpu = queue->cpu_allocator(nullptr, bytes);\
		auto gpu = gpuEncodeWebGPUAddressEXT(active_monobuffer, gpu_address);
		queue->allocations[gpu] = {GpuQueue::MonobufferRange{active_monobuffer, static_cast<uint32_t>(gpu_address), static_cast<uint32_t>(gpu_address + bytes)}, cpu, memory};
		queue->cpu2gpu[cpu] = gpu;
		return cpu;
	};

	if(bytes > queue->limits.maxStorageBufferBindingSize) {
		errno = WGPUErrorType_OutOfMemory;
		return nullptr;
	}

	for(size_t i = 0; i < queue->buffer_freelist.size(); ++i) {
		auto& [active_buffer, start, end] = queue->buffer_freelist[i];
		// Strip any empty ranges from the list
		if(start == end) {
			queue->buffer_freelist.erase(queue->buffer_freelist.begin() + i);
			--i;
			continue;
		}

		// If we find a free space big enough for the allocation... use that
		auto aligned = align_up(start, align);
		if(end - aligned > bytes) {
			start = aligned + bytes;
			return allocation_bookkeeping(queue, active_buffer, aligned, bytes, memory);
		}

		// If each element overlaps with the previous one merge them
		// assumes that the freelist is sorted by buffer index followed by starting value
		if(i == 0) continue;

		auto [prev_buffer, prev_start, prev_end] = queue->buffer_freelist[i - 1];
		if(start > prev_start && start < prev_end && prev_buffer == active_buffer) {
			queue->buffer_freelist.erase(queue->buffer_freelist.begin() + (i - 1));
			start = prev_start;
		}
	}

	auto new_start = align_up(queue->active_monobuffer_size, align);
	auto new_end = new_start + bytes;
	auto new_active_monobuffer = queue->active_monobuffer;

	// Reallocate the monobuffer if necessary
	bool activate_next_monobuffer = false;
	if(new_end > queue->active_monobuffer_capacity) {
		if(queue->monobuffers[queue->active_monobuffer] != queue->empty_buffer)
			queue->code_pending_submission_finished.emplace_back([buffer = queue->monobuffers[queue->active_monobuffer]]() {
				wgpuBufferRelease(buffer);
			}, queue->next_submission_index);
		else queue->active_monobuffer_capacity = new_end;

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

		if(old_monobuffer != queue->empty_buffer) {
			auto cmd = wgpuDeviceCreateCommandEncoder(queue->device, nullptr);
			wgpuCommandEncoderCopyBufferToBuffer(cmd, old_monobuffer, 0, queue->monobuffers[queue->active_monobuffer], 0, queue->active_monobuffer_size);
			auto to_submit = wgpuCommandEncoderFinish(cmd, nullptr);
			wgpuQueueSubmit(queue->queue, 1, &to_submit);
			wgpuCommandEncoderRelease(cmd);
			wgpuCommandBufferRelease(to_submit);
		}

		if(activate_next_monobuffer) {
			// Add the rest of the old monobuffer to the freelist
			queue->buffer_freelist.emplace_back(queue->active_monobuffer, queue->active_monobuffer_size, queue->limits.maxStorageBufferBindingSize);

			++queue->active_monobuffer;
			if(queue->active_monobuffer >= queue->monobuffers.size()) { // Fail if monobuffer being activated is the 8th monobuffer
				errno = WGPUErrorType_OutOfMemory;
				return nullptr;
			}

			WGPUBufferDescriptor d {
				.label = {"NoAPI Monobuffer", WGPU_STRLEN},
				.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc | WGPUBufferUsage_Index | WGPUBufferUsage_Indirect,
				.size = bytes
			};
			queue->monobuffers[queue->active_monobuffer] = wgpuDeviceCreateBuffer(queue->device, &d);

			if(new_start == 0)
				queue->monobuffer_sizes[queue->active_monobuffer] = queue->active_monobuffer_size = 0;
			else queue->monobuffer_sizes[queue->active_monobuffer] = queue->active_monobuffer_size = bytes;
			queue->active_monobuffer_capacity = bytes;
		}
	}

	if(!activate_next_monobuffer) queue->monobuffer_sizes[queue->active_monobuffer] = queue->active_monobuffer_size = new_end;
	return allocation_bookkeeping(queue, new_active_monobuffer, new_start, new_end - new_start, memory);
}

gpu* gpuHostToDevicePointer(GpuQueue* queue, void* ptr) {
	if(queue->cpu2gpu.contains(ptr))
		return queue->cpu2gpu[ptr];
	return nullptr;
}

void* gpuDeviceToHostPointerEXT(GpuQueue* queue, gpu* ptr) {
	if(queue->allocations.contains(ptr))
		return std::get<void*>(queue->allocations[ptr]);
	return nullptr;
}

void gpuFree(GpuQueue* queue, void* ptr) {
	auto device = gpuHostToDevicePointer(queue, ptr);
	if(device) gpuFree(queue, device);
}
void gpuFree(GpuQueue* queue, gpu* ptr) {
	if(!queue->allocations.contains(ptr)) return;

	// TODO: Extra buffers go here
	if(queue->gpu2textures.contains(ptr)) {
		auto texture = queue->gpu2textures[ptr];
		if(texture->range) {
			queue->texture_freelist.emplace_back(*texture->range);
			// Sort the freelist so entries are grouped by monotexture (storage flag + index) and then by start,
			// mirroring buffer_freelist's sort so gpuCreateTexture's freelist search can merge adjacent free ranges
			std::sort(queue->texture_freelist.begin(), queue->texture_freelist.end(), [](const GpuQueue::MonotextureRange& a, const GpuQueue::MonotextureRange& b) -> bool {
				if(a.storage() != b.storage())
					return a.storage() < b.storage();
				if(a._index == b._index)
					return a.start < b.start;
				return a._index < b._index;
			});
		} else {
			wgpuTextureRelease(texture->texture);
		}

		queue->cpu_allocator(texture, 0);
	}

	auto [range, cpu, _memory_type] = queue->allocations[ptr];
	queue->cpu_allocator(cpu, 0);

	queue->allocations.erase(ptr);
	queue->cpu2gpu.erase(cpu);

	queue->buffer_freelist.push_back(range);
	// Sort the freelist so buffers are together, and then by start
	std::sort(queue->buffer_freelist.begin(), queue->buffer_freelist.end(), [](const GpuQueue::MonobufferRange& a, const GpuQueue::MonobufferRange& b) -> bool {
		if(a.buffer == b.buffer)
			return a.start < b.start;

		return a.buffer < b.buffer;
	});
}



GpuTextureSizeAlign gpuTextureSizeAlign(GpuQueue* queue, const GpuTextureDesc& desc) {
	return {16, 16}; // Waste the minimum amount of space dealing with an allocation
}

GpuTexture* gpuCreateTexture(GpuQueue* queue, const GpuTextureDesc& desc, gpu* memory) {
	// Only one texture can be associated with a memory
	if(queue->gpu2textures.contains(memory)) {
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
			if(queue->storage_monotextures.size() >= queue->limits.maxStorageTexturesPerShaderStage) {
				errno = WGPUErrorType_OutOfMemory;
				return nullptr;
			}
			if(queue->storage_monotextures.size() >= queue->limits.maxSampledTexturesPerShaderStage) {
				errno = WGPUErrorType_OutOfMemory;
				return nullptr;
			}
			assert(desc.type != TEXTURE_1D && "1D textures aren't supported by WebGPU");
			assert(desc.type != TEXTURE_3D || desc.layerCount == 1 && "3D textures can't be arrays in WebGPU");

			(*monotextures_lookup)[hash] = monotextures->size();
			auto& [size, texture, full_view, description] = monotextures->emplace_back();
			description = hash;
			size = 0;
			{
				WGPUTextureDescriptor d {
					.usage = GPU::usage2wgpu(desc.usage),
					.dimension = GPU::texture2wgpu(desc.type),
					.size = {desc.dimensions.x, desc.dimensions.y, desc.type == TEXTURE_3D ? desc.dimensions.z : desc.layerCount * 2},
					.format = GPU::format2wgpu(desc.format),
					.mipLevelCount = static_cast<uint32_t>(std::log2(std::max(desc.dimensions.x, desc.dimensions.y)) + 1),
					.sampleCount = desc.sampleCount,
				};
				texture = wgpuDeviceCreateTexture(queue->device, &d);
			}{
				WGPUTextureViewDescriptor d {
					.format = GPU::format2wgpu(desc.format),
					.dimension = GPU::texture_view2wgpu(desc.type),
					.baseMipLevel = 0,
					.mipLevelCount = static_cast<uint32_t>(std::log2(std::max(desc.dimensions.x, desc.dimensions.y)) + 1), // TODO: Does this need a -1
					.baseArrayLayer = 0,
					.arrayLayerCount = desc.type == TEXTURE_3D ? 1 : desc.layerCount * 2,
					.aspect = WGPUTextureAspect_All, // TODO: Do we need to deal with depthonly or stencilonly?
					.usage = GPU::usage2wgpu(desc.usage),
				};
				full_view = wgpuTextureCreateView(texture, &d);
			}

			// We have a new monotexture the pipeline layouts should be aware of
			update_pipeline_layouts(queue);
		}

		auto texture_index = monotextures_lookup->at(hash);
		auto& [size, texture, full_view, _description] = monotextures->at(texture_index);

		for(size_t i = 0; i < queue->texture_freelist.size(); ++i) {
			auto& free = queue->texture_freelist[i];
			// Strip any empty ranges from the list
			if(free.start == free.end) {
				queue->texture_freelist.erase(queue->texture_freelist.begin() + i);
				--i;
				continue;
			}

			// Only consider free ranges belonging to this exact monotexture atlas
			if(free.storage() != is_storage_texture || free.index() != texture_index) continue;

			// If we find a free space big enough for the allocation... use that
			if(free.end - free.start >= desc.layerCount) {
				range = {._index = free._index, .start = free.start, .end = free.start + desc.layerCount};
				free.start += desc.layerCount;
				break;
			}

			// If each element overlaps with the previous one merge them
			// assumes that the freelist is sorted by storage then index then starting value
			if(i == 0) continue;

			auto prev = queue->texture_freelist[i - 1];
			if(free.start > prev.start && free.start < prev.end && prev._index == free._index) {
				queue->texture_freelist.erase(queue->texture_freelist.begin() + (i - 1));
				free.start = prev.start;
			}
		}

		if(!range) {
			auto capacity = desc.type == TEXTURE_3D ? 1 : wgpuTextureGetDepthOrArrayLayers(texture);
			if(size + desc.layerCount > capacity) {
				{
					queue->code_pending_submission_finished.emplace_back([texture]() {
						wgpuTextureRelease(texture);
					}, queue->next_submission_index);

					WGPUTextureDescriptor d {
						.usage = GPU::usage2wgpu(desc.usage),
						.dimension = GPU::texture2wgpu(desc.type),
						.size = {desc.dimensions.x, desc.dimensions.y, desc.type == TEXTURE_3D ? desc.dimensions.z : size * 2},
						.format = GPU::format2wgpu(desc.format),
						.mipLevelCount = static_cast<uint32_t>(std::log2(std::max(desc.dimensions.x, desc.dimensions.y)) + 1),
						.sampleCount = desc.sampleCount,
					};
					texture = wgpuDeviceCreateTexture(queue->device, &d);
				}{
					queue->code_pending_submission_finished.emplace_back([full_view]() {
						wgpuTextureViewRelease(full_view);
					}, queue->next_submission_index);

					WGPUTextureViewDescriptor d {
						.format = GPU::format2wgpu(desc.format),
						.dimension = GPU::texture_view2wgpu(desc.type),
						.baseMipLevel = 0,
						.mipLevelCount = static_cast<uint32_t>(std::log2(std::max(desc.dimensions.x, desc.dimensions.y)) + 1), // TODO: Does this need a -1
						.baseArrayLayer = 0,
						.arrayLayerCount = static_cast<uint32_t>(size * 2),
						.aspect = WGPUTextureAspect_All, // TODO: Do we need to deal with depthonly or stencilonly?
						.usage = GPU::usage2wgpu(desc.usage),
					};
					full_view = wgpuTextureCreateView(texture, &d);
				}
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
			.mipLevelCount = static_cast<uint32_t>(std::log2(std::max(desc.dimensions.x, desc.dimensions.y)) + 1),
			.sampleCount = desc.sampleCount,
		};
		out->texture = wgpuDeviceCreateTexture(queue->device, &d);
		// TODO: Texture view?
	}

	queue->gpu2textures[memory] = out;
	return out;
}

GpuTextureDescriptor gpuTextureViewDescriptor(GpuQueue* queue, const GpuTexture* texture, const GpuViewDesc& desc) {
	GpuTextureDescriptorImpl out {
		.type = static_cast<uint8_t>(texture->descriptor.type),
		.baseMip = desc.baseMip,
		.mipCount = static_cast<uint8_t>(desc.mipCount == ALL_MIPS ? texture->descriptor.mipCount - desc.baseMip : desc.mipCount),
		.width = texture->descriptor.dimensions.x,
		.height = texture->descriptor.dimensions.y,
		.depth = texture->descriptor.dimensions.z,
		.range = texture->range ? *texture->range : GpuQueue::MonotextureRange::INVALID
	};
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
	return gpuTextureViewDescriptor(queue, texture, desc);
}

// Splices the bindings generated for the queue's current monobuffer/monotexture set into the
// provided IR (replacing every @generated_noapi_bindings marker) and compiles the result
WGPUShaderModule create_shader_module(GpuQueue* queue, std::string_view IR, bool compute) {
	constexpr std::string_view marker = "@generated_noapi_bindings";
	std::string wgsl_source = {IR.begin(), IR.end()};
	auto generated_bindings = generate_binding_prologue(queue, compute);
	std::size_t pos = 0;
	while ((pos = wgsl_source.find(marker, pos)) != std::string::npos) {
		wgsl_source.replace(pos, marker.size(), generated_bindings);
		pos += generated_bindings.size();
	}

	WGPUShaderSourceWGSL source {
		.chain {.sType = WGPUSType_ShaderSourceWGSL },
		.code = {wgsl_source.data(), wgsl_source.size()}
	};
	WGPUShaderModuleDescriptor module {
		.nextInChain = &source.chain
	};
	return wgpuDeviceCreateShaderModule(queue->device, &module);
}

void update_compute_pipeline(GpuQueue* queue, const GpuPipeline* pipeline) {
	auto& cache = std::get<GpuPipeline::ComputeCache>(pipeline->cache);
	if(cache.pipeline) wgpuComputePipelineRelease(cache.pipeline);

	WGPUComputePipelineDescriptor d {
		.layout = queue->current_compute_pipeline_layout,
		.compute = {
			.module = create_shader_module(queue, cache.IR, true),
			.entryPoint = {"main", WGPU_STRLEN},
		}
	};
	cache.pipeline = wgpuDeviceCreateComputePipeline(queue->device, &d);
	pipeline->reference_layout = queue->current_compute_pipeline_layout;

	wgpuShaderModuleRelease(d.compute.module);
}

// Rebuilds the cached variant of a graphics pipeline against the queue's current pipeline layout
// and the depth/stencil, blend, and index state that is about to be drawn with. WebGPU bakes all
// of them into the PSO, so every one of them is recorded alongside the pipeline for comparison.
void update_render_pipeline(GpuQueue* queue, const GpuPipeline* pipeline, const std::optional<GpuDepthStencilDesc>& depth_stencil, const std::optional<GpuBlendDesc>& blend, INDEX_TYPE_EXT index_type) {
	auto& cache = std::get<GpuPipeline::RenderCache>(pipeline->cache);
	if(cache.pipeline) wgpuRenderPipelineRelease(cache.pipeline);

	auto& desc = cache.descriptor;

	// A dynamically applied blend state takes precedence over one baked into the rasterizer description
	auto effective_blend = blend ? blend : desc.blendstate;
	WGPUBlendState blend_state {
		.color = {
			.operation = GPU::blend2wgpu(effective_blend.value_or(GpuBlendDesc{}).colorOp),
			.srcFactor = GPU::factor2wgpu(effective_blend.value_or(GpuBlendDesc{}).srcColorFactor),
			.dstFactor = GPU::factor2wgpu(effective_blend.value_or(GpuBlendDesc{}).dstColorFactor),
		},
		.alpha = {
			.operation = GPU::blend2wgpu(effective_blend.value_or(GpuBlendDesc{}).alphaOp),
			.srcFactor = GPU::factor2wgpu(effective_blend.value_or(GpuBlendDesc{}).srcAlphaFactor),
			.dstFactor = GPU::factor2wgpu(effective_blend.value_or(GpuBlendDesc{}).dstAlphaFactor),
		},
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
	auto has_depth = gpuFormatIsDepth(depth_stencil_format);
	auto has_stencil = GPU::format_has_stencil(depth_stencil_format);

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

	auto vertex_module = create_shader_module(queue, cache.vertexIR, false);
	auto fragment_module = cache.fragmentIR.empty() ? nullptr : create_shader_module(queue, cache.fragmentIR, false);

	WGPUFragmentState fragment {
		.module = fragment_module,
		// Undefined rather than left alone: a zeroed WGPUStringView is the empty string, which Dawn
		// then goes looking for as an entry point name
		.entryPoint = WGPU_STRING_VIEW_INIT,
		.targetCount = targets.size(),
		.targets = targets.data(),
	};
	WGPURenderPipelineDescriptor d {
		.layout = queue->current_graphics_pipeline_layout,
		.vertex = {
			.module = vertex_module,
			// Leaving the entry point undefined picks the module's only entry point for the stage,
			// which lets a single blob holding both shaders be handed to us for both stages
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

GpuPipeline* gpuCreateComputePipeline(GpuQueue* queue, std::span<const std::byte> computeIR) {
	auto out = (GpuPipeline*)queue->cpu_allocator(nullptr, sizeof(GpuPipeline));
	new(out) GpuPipeline{
		.reference_layout = nullptr,
		.cache = GpuPipeline::ComputeCache {
			.IR = {(char*)computeIR.data(), (char*)(computeIR.data() + computeIR.size())},
		}
	};

	update_compute_pipeline(queue, out);
	return out;
}

GpuPipeline* gpuCreateGraphicsPipeline(GpuQueue* queue, std::span<const std::byte> vertexIR, std::span<const std::byte> fragmentIR, const GpuRasterDesc& desc) {
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

	// Unlike a compute pipeline we can't build anything yet: the depth/stencil state, the blend
	// state, and the index format are all baked into a WebGPU PSO but none of them are bound until
	// a draw is recorded. The first draw builds the initial variant.
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

GpuCommandBuffer* gpuStartCommandRecording(GpuQueue* queue) {
	auto out = (GpuCommandBuffer*)queue->cpu_allocator(nullptr, sizeof(GpuCommandBuffer));
	new(out) GpuCommandBuffer{
		.queue = queue,
		.encoder = wgpuDeviceCreateCommandEncoder(queue->device, nullptr)
	};

	GPU::process_pending_code(queue);

	return out;
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

void gpuFreeCommandBuffer(GpuCommandBuffer* cmd) {
	wgpuCommandEncoderRelease(cmd->encoder);

	auto allocator = cmd->queue->cpu_allocator;
	cmd->~GpuCommandBuffer();
	allocator(cmd, 0);
}

uint64_t gpuSubmitNoFree(GpuQueue* queue, std::span<GpuCommandBuffer*> command_buffers, GpuSemaphore* semaphore /* = nullptr */, uint64_t signal_value /* = 0 */) {
	std::vector<WGPUCommandBuffer> buffers; buffers.reserve(command_buffers.size() + 1);
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

	buffers.emplace_back(wgpuCommandEncoderFinish(cmd.encoder, nullptr));
	wgpuCommandEncoderRelease(cmd.encoder);

	wgpuQueueSubmit(queue->queue, buffers.size(), buffers.data());

	for(auto buffer: command_buffers)
		for(auto code: buffer->code_pending_submission_finished)
			queue->code_pending_submission_finished.emplace_back(code, queue->next_submission_index);

	return queue->next_submission_index++;
}

uint64_t gpuSubmit(GpuQueue* queue, std::span<GpuCommandBuffer*> command_buffers, GpuSemaphore* semaphore /* = nullptr */, uint64_t signal_value /* = 0 */) {
	auto submission = gpuSubmitNoFree(queue, command_buffers, semaphore, signal_value);
	for(auto cmd: command_buffers)
		gpuFreeCommandBuffer(cmd);
	return submission;
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

void gpuSyncMemoryEXT(GpuCommandBuffer* cmd, gpu* mem) {
	endCurrentPass(cmd);
	auto [range, cpu, memory_type] = cmd->queue->allocations[mem];

	switch (memory_type) {
	case MEMORY_DEFAULT:
	case MEMORY_GPU:
	case MEMORY_TEXTURE:
		GPU::push_to_monobuffer(cmd, range, cpu);

	break; case MEMORY_READBACK:
	case MEMORY_TEXTURE_READBACK:
		GPU::pull_from_monobuffer(cmd, range, cpu);
	}
}

void gpuSyncMemoryEXT(GpuQueue* queue, gpu* mem) {
	auto cmd = gpuStartCommandRecording(queue);
	gpuSyncMemoryEXT(cmd, mem);
	auto submit_index = gpuSubmit(queue, {&cmd, 1});

	GPU::semaphore_wait(queue, queue->current_submission_timeline_semaphore, submit_index);
	GPU::process_pending_code(queue);
}

const GpuSemaphore* gpuGetSubmissionSemaphoreEXT(GpuQueue* queue) {
	return (GpuSemaphore*)&queue->current_submission_timeline_semaphore;
}

GpuSemaphore* gpuCreateSemaphore(GpuQueue* queue, uint64_t initial_value) {
	auto out = (GpuSemaphore*)queue->cpu_allocator(nullptr, sizeof(GpuSemaphore));

	*out = GPU::semaphore_initialize(queue, initial_value);
	return out;
}

uint64_t gpuWaitSemaphore(GpuQueue* queue, const GpuSemaphore* semaphore, uint64_t value, uint64_t timeout /* = UINT64_MAX */) {
	if(value == GPU_GET_VALUE)
		return GPU::semaphore_value(queue, *semaphore);

	GPU::semaphore_wait(queue, *semaphore, value, std::chrono::nanoseconds(timeout));
	GPU::process_pending_code(queue);
	return value;
}

void gpuFreeSemaphore(GpuQueue* queue, GpuSemaphore* semaphore) {
	GPU::semaphore_destroy(*semaphore);
	queue->cpu_allocator(semaphore, 0);
}




namespace GPU::detail {
	inline std::tuple<GpuQueue::MonobufferRange, ptrdiff_t, gpu*> closest_buffer(GpuQueue* queue, gpu* addr, bool no_offsets) {
		gpu* closest = gpuEncodeWebGPUAddressEXT(queue->monobuffers.size() - 1, gpu_address_max - 1);
		if(no_offsets)
			closest = addr;
		else for(auto [key, _]: queue->allocations) {
			if(closest - addr > size_t(key - addr))
				closest = key;
		}
		return {std::get<GpuQueue::MonobufferRange>(queue->allocations[closest]), closest - addr, closest};
	}

	inline std::array<std::tuple<GpuQueue::MonobufferRange, ptrdiff_t, gpu*>, 2> closest_buffer(GpuQueue* queue, gpu* addrA, gpu* addrB, bool no_offsets) {
		auto last_monobuffer = queue->monobuffers.size() - 1;
		gpu* closestA = gpuEncodeWebGPUAddressEXT(last_monobuffer, gpu_address_max - 1), *closestB = gpuEncodeWebGPUAddressEXT(last_monobuffer, gpu_address_max - 1); // TODO: There are probably edge cases around setting these to zero!
		if(no_offsets) {
			closestA = addrA;
			closestB = addrB;
		} else for(auto [key, _]: queue->allocations) {
			if(closestA - addrA > size_t(key - addrA))
				closestA = key;
			if(closestB - addrB > size_t(key - addrB))
				closestB = key;
		}
		return {
			std::tuple<GpuQueue::MonobufferRange, ptrdiff_t, gpu*>{std::get<GpuQueue::MonobufferRange>(queue->allocations[closestA]), closestA - addrA, closestA},
			std::tuple<GpuQueue::MonobufferRange, ptrdiff_t, gpu*>{std::get<GpuQueue::MonobufferRange>(queue->allocations[closestB]), closestB - addrB, closestB}
		};
	}
}

namespace GPU::detail {
	// Bytes per texel for the uncompressed, single plane formats the API exposes
	inline uint32_t format_bytes(FORMAT format) {
		switch (format) {
		case FORMAT_R8_UNORM: return 1;
		case FORMAT_R16_FLOAT:
		case FORMAT_D16_UNORM: return 2;
		case FORMAT_RGBA8_UNORM:
		case FORMAT_RGBA8_SRGB:
		case FORMAT_RG11B10_FLOAT:
		case FORMAT_RGB10_A2_UNORM:
		case FORMAT_R32_FLOAT:
		case FORMAT_D24_UNORM_S8_UINT:
		case FORMAT_D32_FLOAT: return 4;
		case FORMAT_RGBA16_FLOAT:
		case FORMAT_D32_FLOAT_S8_UINT: return 8;
		case FORMAT_RGBA32_FLOAT: return 16;
		default: return 0;
		}
	}

	// The extent of the monotexture slot a texture sits in, at one mip level. A monotexture is
	// sized by the first texture to land in its bucket and the bucket keys on power of two rounded
	// dimensions, so a later, smaller texture only covers the top left corner of its slot. Both
	// round up to the same power of two, so a slot is never as much as twice the image.
	inline uvec2 monotexture_slot_extent(const GpuTexture* texture, uint32_t mip) {
		return {
			std::max(wgpuTextureGetWidth(texture->texture) >> mip, 1u),
			std::max(wgpuTextureGetHeight(texture->texture) >> mip, 1u)
		};
	}

	// Replicates a freshly uploaded texture's right and bottom edges across the rest of its slot.
	// Whatever the gap holds is otherwise undefined, and it gets sampled the moment normalized
	// coordinates run past the image or the sampler filters across its last row or column.
	//
	// The replication has to come out of the same staging buffer the upload did: WebGPU rejects a
	// texture to texture copy whose source and destination are the same mip of the same layer, so
	// the texture can't be used to widen itself.
	//
	// Naming bytesPerRow at all forces it to be 256 byte aligned, so every copy that can get away
	// with leaving it undefined does — that is any copy of a single row of a single layer, which
	// covers all of this but the one strip that has to walk down the image.
	inline void pad_monotexture_slot(GpuCommandBuffer* cmd, const GpuTexture* texture,
			WGPUBuffer buffer, uint64_t base, uint32_t layer, uint32_t layers) {
		const uint32_t width = texture->descriptor.dimensions.x, height = texture->descriptor.dimensions.y;
		auto slot = monotexture_slot_extent(texture, 0);
		const uint32_t gap_x = slot.x > width ? slot.x - width : 0;
		const uint32_t gap_y = slot.y > height ? slot.y - height : 0;
		if(!gap_x && !gap_y) return;

		const uint32_t bytes = format_bytes(texture->descriptor.format);
		const uint64_t row = uint64_t(width) * bytes;
		const uint64_t image = row * height; // What one array layer of the staging data spans

		WGPUTexelCopyTextureInfo destination {
			.texture = texture->texture,
			.mipLevel = 0,
			.aspect = WGPUTextureAspect_All,
		};

		// The last column, stretched across every column the image doesn't reach. This is the only
		// strip that spans rows, so it is the only one carrying the upload's row alignment.
		for(uint32_t x = width; x < slot.x; ++x) {
			WGPUTexelCopyBufferInfo source {
				.layout = {
					.offset = base + uint64_t(width - 1) * bytes,
					.bytesPerRow = static_cast<uint32_t>(row),
					.rowsPerImage = height,
				},
				.buffer = buffer,
			};
			destination.origin = {x, 0, layer};
			WGPUExtent3D size {1, height, layers};
			wgpuCommandEncoderCopyBufferToTexture(cmd->encoder, &source, &destination, &size);
		}

		// The last row, stretched down every row the image doesn't reach. Each layer keeps its own
		// last row, so they are written one at a time rather than strided over in a single copy.
		for(uint32_t l = 0; l < layers; ++l)
			for(uint32_t y = height; y < slot.y; ++y) {
				WGPUTexelCopyBufferInfo source {
					.layout = {
						.offset = base + l * image + (height - 1) * row,
						.bytesPerRow = WGPU_COPY_STRIDE_UNDEFINED,
						.rowsPerImage = WGPU_COPY_STRIDE_UNDEFINED,
					},
					.buffer = buffer,
				};
				destination.origin = {0, y, layer + l};
				WGPUExtent3D size {width, 1, 1};
				wgpuCommandEncoderCopyBufferToTexture(cmd->encoder, &source, &destination, &size);
			}

		if(!gap_x || !gap_y) return;

		// Where the two gaps meet every texel is the image's bottom right one. Copying it out of the
		// staging buffer directly would cost a command per texel, so each layer's corner texel is
		// first smeared across a scratch row that its remaining rows fill from in one command each.
		const bool alignable = bytes % 4 == 0 && (base + (height - 1) * row + uint64_t(width - 1) * bytes) % 4 == 0 && image % 4 == 0;
		if(alignable) {
			WGPUBufferDescriptor scratch_desc {
				.label = {"NoAPI Monotexture Corner", WGPU_STRLEN},
				.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst,
				.size = uint64_t(gap_x) * bytes * layers,
			};
			auto scratch = wgpuDeviceCreateBuffer(cmd->queue->device, &scratch_desc);
			cmd->queue->code_pending_submission_finished.emplace_back([scratch] {
				wgpuBufferRelease(scratch);
			}, cmd->queue->next_submission_index);

			for(uint32_t l = 0; l < layers; ++l) {
				auto corner = base + l * image + (height - 1) * row + uint64_t(width - 1) * bytes;
				for(uint32_t i = 0; i < gap_x; ++i)
					wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, buffer, corner,
						scratch, (uint64_t(l) * gap_x + i) * bytes, bytes);
			}

			for(uint32_t l = 0; l < layers; ++l)
				for(uint32_t y = height; y < slot.y; ++y) {
					WGPUTexelCopyBufferInfo source {
						.layout = {
							.offset = uint64_t(l) * gap_x * bytes,
							.bytesPerRow = WGPU_COPY_STRIDE_UNDEFINED,
							.rowsPerImage = WGPU_COPY_STRIDE_UNDEFINED,
						},
						.buffer = scratch,
					};
					destination.origin = {width, y, layer + l};
					WGPUExtent3D size {gap_x, 1, 1};
					wgpuCommandEncoderCopyBufferToTexture(cmd->encoder, &source, &destination, &size);
				}
		} else {
			// A buffer to buffer copy has to be 4 byte aligned, which the narrow formats can't
			// promise, so those fall back to writing the corner a texel at a time
			for(uint32_t l = 0; l < layers; ++l) {
				auto corner = base + l * image + (height - 1) * row + uint64_t(width - 1) * bytes;
				for(uint32_t y = height; y < slot.y; ++y)
					for(uint32_t x = width; x < slot.x; ++x) {
						WGPUTexelCopyBufferInfo source {
							.layout = {
								.offset = corner,
								.bytesPerRow = WGPU_COPY_STRIDE_UNDEFINED,
								.rowsPerImage = WGPU_COPY_STRIDE_UNDEFINED,
							},
							.buffer = buffer,
						};
						destination.origin = {x, y, layer + l};
						WGPUExtent3D size {1, 1, 1};
						wgpuCommandEncoderCopyBufferToTexture(cmd->encoder, &source, &destination, &size);
					}
			}
		}
	}
}

void gpuMemCpy(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, size_t bytes, bool no_offsets /* = false */) {
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_, no_offsets);
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

// TODO: Untested!
void gpuCopyToTexture(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, GpuTexture* texture, bool no_offsets /* = false */) {
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_, no_offsets);
	auto [dest_range, dest_offset, dest_addr] = dest; auto [src_range, src_offset, src_addr] = src;

	assert(cmd->queue->gpu2textures[dest_addr] == texture);

	auto base = static_cast<uint64_t>(src_range.start + src_offset);
	WGPUTexelCopyBufferInfo source {
		.layout = {
			.offset = base,
			.bytesPerRow = texture->descriptor.dimensions.x * GPU::detail::format_bytes(texture->descriptor.format),
			.rowsPerImage = texture->descriptor.dimensions.y,
		},
		.buffer = cmd->queue->monobuffers[src_range.buffer],
	};
	WGPUTexelCopyTextureInfo destination = {
		.texture = texture->texture,
		.mipLevel = 0, // TODO: How do we allow more control over this?
		.origin = {0, 0, 0},
		.aspect = WGPUTextureAspect_All
	};
	if(texture->range) destination.origin = {0, 0, texture->range->start};
	WGPUExtent3D size {
		.width = texture->descriptor.dimensions.x,
		.height = texture->descriptor.dimensions.y,
		.depthOrArrayLayers = texture->range ? texture->range->end - texture->range->start : texture->descriptor.dimensions.z
	};
	wgpuCommandEncoderCopyBufferToTexture(cmd->encoder, &source, &destination, &size);

	// The texture may not fill the slot it was handed in its monotexture, and the gap would
	// otherwise be sampled as undefined data the moment a coordinate ran past the image
	if(texture->range)
		GPU::detail::pad_monotexture_slot(cmd, texture, source.buffer, base, destination.origin.z, size.depthOrArrayLayers);
}

// TODO: Untested!
void gpuCopyFromTexture(GpuCommandBuffer* cmd, gpu* dest_, gpu* src_, const GpuTexture* texture, bool no_offsets /* = false */) {
	auto [dest, src] = GPU::detail::closest_buffer(cmd->queue, dest_, src_, no_offsets);
	auto [dest_range, dest_offset, dest_addr] = dest; auto [src_range, src_offset, src_addr] = src;

	assert(cmd->queue->gpu2textures[src_addr] == texture);

	WGPUTexelCopyTextureInfo source = {
		.texture = texture->texture,
		.mipLevel = 0, // TODO: How do we allow more control over this?
		.origin = {0, 0, 0},
		.aspect = WGPUTextureAspect_All
	};
	if(texture->range) source.origin = {0, 0, texture->range->start};
	WGPUTexelCopyBufferInfo destination {
		.layout = {
			.offset = static_cast<uint64_t>(src_range.start + src_offset),
			.bytesPerRow = texture->descriptor.dimensions.x * texture->descriptor.format,
			.rowsPerImage = texture->descriptor.dimensions.y,
		},
		.buffer = cmd->queue->monobuffers[src_range.buffer],
	};
	WGPUExtent3D size {
		.width = texture->descriptor.dimensions.x,
		.height = texture->descriptor.dimensions.y,
		.depthOrArrayLayers = texture->range ? texture->range->start - texture->range->end : texture->descriptor.dimensions.z
	};
	wgpuCommandEncoderCopyTextureToBuffer(cmd->encoder, &source, &destination, &size);
}



void gpuSetActiveTextureHeapPtr(GpuCommandBuffer* cmd, gpu* texture_heap, bool no_offsets /* = false */) {
	// TODO: Should we add some validation to check that what is provided is a valid texture heap?
	auto [dest_range, dest_offset, dest_addr] = GPU::detail::closest_buffer(cmd->queue, texture_heap, no_offsets);
	dest_range.start += dest_offset;
	// dest_range.end -= dest_offset;

	// The heap lives in a monobuffer, and group 0 binds every monobuffer as writable storage for
	// compute. WebGPU refuses to have one buffer bound writable and read only in the same pass, so
	// binding the heap out of the monobuffer directly would fail validation on the first dispatch.
	//
	// Copying instead of keeping a shadow buffer around: the heap is written through a raw mapped
	// pointer, so there is no way to tell whether it changed since last time and a persistent buffer
	// would have to re-copy on every activation anyway. A copy recorded here is also ordered in the
	// command stream, so it picks up whatever the commands before it wrote into the heap, and two
	// command buffers in flight with different heaps each get their own.
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



void gpuSetEnabledSamplersEXT(GpuCommandBuffer* cmd, std::span<GpuSamplerDesc> enabled_samplers) {
	// Nothing is recorded into the encoder: the set only decides which group 3 (and which copy of the
	// metadata buffer in group 0) the next dispatch or draw binds
	cmd->sampler_set = GPU::detail::ensure_sampler_set(cmd->queue, enabled_samplers);
}

void gpuBarrier(GpuCommandBuffer* cmd, STAGE before, STAGE after, HAZARD_FLAGS hazards /* = (HAZARD_FLAGS)0 */) {
	// TODO: I believe webgpu handles these internally... is there any reason to not make them noops?
}
void gpuSignalAfter(GpuCommandBuffer* cmd, STAGE before, gpu* ptr, uint64_t value, SIGNAL signal) {
	// TODO: I believe webgpu handles these internally... is there any reason to not make them noops?
}
void gpuWaitBefore(GpuCommandBuffer* cmd, STAGE after, gpu* ptr, uint64_t value, OP op, HAZARD_FLAGS hazards /* = (HAZARD_FLAGS)0 */, uint64_t mask /* = ~uint64_t(0) */) {
	// TODO: I believe webgpu handles these internally... is there any reason to not make them noops?
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

		// A render pipeline can't be built (let alone set) until we know what is being drawn with it,
		// so the variant matching the bound state is compiled and set by the draw commands instead.
	}
}

// Makes sure the cached variant of the bound graphics pipeline matches the state it is about to be
// drawn with, rebuilding it when the bound elements or the pipeline layout have moved on, and then
// sets it on the render pass
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

struct ComputeShaderData {
	gpu* compute;
};

struct GraphicsShaderData {
	gpu* vertex;
	gpu* fragment;
};

// Builds group 0: every monobuffer, the active texture heap, and a throwaway uniform holding the
// root data pointer(s) the shaders were invoked with
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

WGPUBindGroup createBufferBindGroup(GpuCommandBuffer* cmd, gpu* data, bool no_offsets) {
	ComputeShaderData shader_data { .compute = data };
	return createBufferBindGroup(cmd, &shader_data, sizeof(shader_data), true);
}

// Binds every group needed by a dispatch or a draw (the texture groups are shared between the two)
void bindGroups(GpuCommandBuffer* cmd, const void* shader_data, size_t shader_data_size, bool compute) {
	auto group0 = createBufferBindGroup(cmd, shader_data, shader_data_size, compute);
	cmd->queue->code_pending_submission_finished.emplace_back([group0] { wgpuBindGroupRelease(group0); }, cmd->queue->next_submission_index);

	auto group3 = (cmd->sampler_set ? cmd->sampler_set : cmd->queue->default_sampler_set)->bind_group;

	if(compute) {
		wgpuComputePassEncoderSetBindGroup(cmd->compute_pass, 0, group0, 0, nullptr);
		wgpuComputePassEncoderSetBindGroup(cmd->compute_pass, 1, cmd->queue->current_bind_group1, 0, nullptr);
		wgpuComputePassEncoderSetBindGroup(cmd->compute_pass, 2, cmd->queue->current_bind_group2, 0, nullptr);
		wgpuComputePassEncoderSetBindGroup(cmd->compute_pass, 3, group3, 0, nullptr);
	} else {
		wgpuRenderPassEncoderSetBindGroup(cmd->render_pass, 0, group0, 0, nullptr);
		wgpuRenderPassEncoderSetBindGroup(cmd->render_pass, 1, cmd->queue->current_bind_group1, 0, nullptr);
		wgpuRenderPassEncoderSetBindGroup(cmd->render_pass, 2, cmd->queue->current_bind_group2, 0, nullptr);
		wgpuRenderPassEncoderSetBindGroup(cmd->render_pass, 3, group3, 0, nullptr);
	}
}

void bindComputeGroups(GpuCommandBuffer* cmd, gpu* data) {
	ComputeShaderData shader_data { .compute = data };
	bindGroups(cmd, &shader_data, sizeof(shader_data), true);
}

void bindGraphicsGroups(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data) {
	GraphicsShaderData shader_data { .vertex = vertex_data, .fragment = fragment_data };
	bindGroups(cmd, &shader_data, sizeof(shader_data), false);
}

void gpuDispatch(GpuCommandBuffer* cmd, gpu* data, uvec3 grid_dimensions, bool no_offsets /* = false */) {
	assert(cmd->compute_pass);

	bindComputeGroups(cmd, data);

	wgpuComputePassEncoderDispatchWorkgroups(cmd->compute_pass, grid_dimensions.x, grid_dimensions.y, grid_dimensions.z);
}
void gpuDispatchIndirect(GpuCommandBuffer* cmd, gpu* data, gpu* grid_dimensions_gpu, bool no_offsets /* = false */) {
	assert(cmd->compute_pass);
	assert(grid_dimensions_gpu);

	auto [grid_range, grid_offset, grid_addr] = GPU::detail::closest_buffer(cmd->queue, grid_dimensions_gpu, no_offsets);

	bindComputeGroups(cmd, data);

	wgpuComputePassEncoderDispatchWorkgroupsIndirect(cmd->compute_pass, cmd->queue->monobuffers[grid_range.buffer], grid_range.start + grid_offset);
}


void gpuSetDepthStencilState(GpuCommandBuffer* cmd, const GpuDepthStencilState* state) {
	// WebGPU bakes everything except the stencil reference into the pipeline, so all we can do here
	// is record the state; bindRenderPipeline rebuilds the bound pipeline when it no longer matches
	cmd->depth_stencil = state ? std::optional{state->descriptor} : std::nullopt;

	if(cmd->render_pass && cmd->depth_stencil)
		wgpuRenderPassEncoderSetStencilReference(cmd->render_pass, cmd->depth_stencil->stencilFront.reference);
}

void gpuSetBlendState(GpuCommandBuffer* cmd, const GpuBlendState* state) {
	// Blending is part of a WebGPU PSO, so this is recorded and applied by the next draw
	cmd->blend = state ? std::optional{state->descriptor} : std::nullopt;
}

void gpuSetViewportEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin /* = {0, 0} */, float depth_min /* = 0 */, float depth_max /* = 1 */) {
	assert(cmd->render_pass && "The viewport can only be set inside a render pass");
	wgpuRenderPassEncoderSetViewport(cmd->render_pass, origin.x, origin.y, extent.x, extent.y, depth_min, depth_max);
}

void gpuSetScissorRectEXT(GpuCommandBuffer* cmd, uvec2 extent, ivec2 origin /* = {0, 0} */) {
	assert(cmd->render_pass && "The scissor rectangle can only be set inside a render pass");
	wgpuRenderPassEncoderSetScissorRect(cmd->render_pass, origin.x, origin.y, extent.x, extent.y);
}



namespace GPU::detail {
	// Creates the single subresource view a texture is attached to a render pass through. The view
	// lives until the submission referencing it has finished.
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

	// Depth and stencil are separate attachments in our API but WebGPU merges them into one, so when
	// both are provided they have to name the same subresource
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
		if(gpuFormatIsDepth(format)) {
			auto& attachment = desc.depthAttachment ? *desc.depthAttachment : *merged;
			depth_stencil.depthLoadOp = desc.depthAttachment ? GPU::load2wgpu(attachment.loadOp) : WGPULoadOp_Load;
			depth_stencil.depthStoreOp = desc.depthAttachment ? GPU::store2wgpu(attachment.storeOp) : WGPUStoreOp_Store;
			depth_stencil.depthClearValue = static_cast<float>(attachment.clearValue);
		}
		if(GPU::format_has_stencil(format)) {
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

void gpuEndRenderPass(GpuCommandBuffer* cmd, std::optional<const GpuRenderPassDesc> desc /* = {} */) {
	// The descriptor is only needed by backends that transition images by hand, WebGPU tracks that itself
	endRenderPass(cmd);
}



namespace GPU::detail {
	// The blit's fullscreen triangle. Both bind group layout flavors compile this same module; they
	// only differ in the sampler and texture types they declare, which the WGSL never names.
	constexpr static std::string_view BLIT_WGSL_CODE = R"WGSL(
@group(0) @binding(0) var blit_source : texture_2d<f32>;
@group(0) @binding(1) var blit_sampler : sampler;
// xy = the viewport's size over the destination image's, so that uv reaches 1 exactly at the
// image's edge even when the viewport was opened up to cover the rest of a monotexture slot
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
	return textureSampleLevel(blit_source, blit_sampler, varyings.uv, 0.0);
}
)WGSL";

	// A WebGPU sampler can only interpolate between texels of a filterable format, and the 32 bit
	// float ones only become filterable with an optional feature we never ask the device for
	inline bool format_is_filterable(FORMAT format) {
		return !(format == FORMAT_R32_FLOAT || format == FORMAT_RGBA32_FLOAT);
	}

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
				.visibility = WGPUShaderStage_Vertex,
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

	// The destination format is baked into a WebGPU PSO and the sampler type into its layout, so one
	// pipeline is kept per (format, filtering) pair for the lifetime of the queue
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
		bool linear_filter /* = true */,
		uint32_t destination_mip /* = 0 */, uint32_t destination_slice /* = 0 */,
		uint32_t source_mip /* = 0 */, uint32_t source_slice /* = 0 */) {
	assert(!cmd->render_pass && "A blit opens a render pass of its own, so it can't be recorded inside another one");
	assert((source->descriptor.usage & USAGE_SAMPLED) && "The blit source must have been created with USAGE_SAMPLED");
	assert((destination->descriptor.usage & USAGE_COLOR_ATTACHMENT) && "The blit destination must have been created with USAGE_COLOR_ATTACHMENT");
	assert(!gpuFormatIsDepth(source->descriptor.format) && !gpuFormatIsDepth(destination->descriptor.format) && "Depth/stencil textures can't be blitted");
	assert(source->descriptor.type != TEXTURE_3D && "A slice of a 3D texture can't be bound on its own, blit out of a 2D array instead");
	assert(source_mip < source->descriptor.mipCount && destination_mip < destination->descriptor.mipCount);

	endComputePass(cmd);

	// The source is only ever read through a single mip of a single layer, which is what lets the
	// shader stay a plain texture_2d no matter what the texture it came from actually is
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
	bool filtering = linear_filter && GPU::detail::format_is_filterable(source->descriptor.format);
	auto pipeline = GPU::detail::blit_pipeline(cmd->queue, GPU::format2wgpu(destination->descriptor.format), filtering);

	// The viewport is opened up to the whole monotexture slot so that fragments exist past the
	// destination image, and the uv scale keeps uv = 1 sitting exactly on the image's edge. Those
	// outer fragments therefore sample above 1, where the blit sampler's clamped addressing
	// replicates the edge texels into the gap, covering it in the same draw.
	auto extent = GPU::detail::attachment_extent(destination, destination_mip);
	auto slot = GPU::detail::monotexture_slot_extent(destination, destination_mip);

	float uv_scale[4] = {
		static_cast<float>(slot.x) / extent.x,
		static_cast<float>(slot.y) / extent.y,
		0, 0,
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

	// A pass of its own rather than cmd->render_pass, so nothing the command buffer had bound for
	// the user's rendering is disturbed (a WebGPU pass owns every piece of state set on it)
	auto pass = wgpuCommandEncoderBeginRenderPass(cmd->encoder, &pass_desc);
	wgpuRenderPassEncoderSetPipeline(pass, pipeline);
	wgpuRenderPassEncoderSetBindGroup(pass, 0, group, 0, nullptr);
	wgpuRenderPassEncoderSetViewport(pass, 0, 0, slot.x, slot.y, 0, 1);
	wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, slot.x, slot.y);
	wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
	wgpuRenderPassEncoderEnd(pass);
	wgpuRenderPassEncoderRelease(pass);
}



// Matches both VkDrawIndexedIndirectCommand and WebGPU's indexed indirect argument layout
struct GpuDrawIndexedIndirectCommand {
	uint32_t indexCount;
	uint32_t instanceCount;
	uint32_t firstIndex;
	int32_t baseVertex;
	uint32_t firstInstance;
};

namespace GPU::detail {
	// The monobuffers are created with Index usage, so the indices are drawn straight out of the one
	// they were allocated in; no shadow copy (and thus nothing for no_index_buffer_changes to skip)
	inline void bind_index_buffer(GpuCommandBuffer* cmd, gpu* indices, INDEX_TYPE_EXT index_type, bool no_offsets) {
		auto [range, offset, address] = GPU::detail::closest_buffer(cmd->queue, indices, no_offsets);
		assert(range.start + offset < range.end && "The indices point past the end of their allocation");

		wgpuRenderPassEncoderSetIndexBuffer(cmd->render_pass, cmd->queue->monobuffers[range.buffer],
			GPU::index2wgpu(index_type), range.start + offset, range.end - (range.start + offset));
	}
}

void gpuDrawIndexedInstanced(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* indices, uint32_t index_count, uint32_t instance_count, INDEX_TYPE_EXT index_type /* = INDEX_TYPE_UINT32 */, bool no_offsets /* = false */, bool no_index_buffer_changes /* = false */) {
	assert(cmd->render_pass);
	assert(indices);

	cmd->index_type = index_type;
	bindRenderPipeline(cmd);
	bindGraphicsGroups(cmd, vertex_data, fragment_data);
	GPU::detail::bind_index_buffer(cmd, indices, index_type, no_offsets);

	wgpuRenderPassEncoderDrawIndexed(cmd->render_pass, index_count, instance_count, 0, 0, 0);
}

void gpuDrawIndexedInstancedIndirect(GpuCommandBuffer* cmd, gpu* vertex_data, gpu* fragment_data, gpu* indices, gpu* args, INDEX_TYPE_EXT index_type /* = INDEX_TYPE_UINT32 */, bool no_offsets /* = false */, bool no_index_buffer_changes /* = false */) {
	assert(cmd->render_pass);
	assert(indices);
	assert(args);

	cmd->index_type = index_type;
	bindRenderPipeline(cmd);
	bindGraphicsGroups(cmd, vertex_data, fragment_data);
	GPU::detail::bind_index_buffer(cmd, indices, index_type, no_offsets);

	auto [range, offset, address] = GPU::detail::closest_buffer(cmd->queue, args, no_offsets);
	auto start = range.start + offset;
	auto count = (range.end - start) / sizeof(GpuDrawIndexedIndirectCommand);

	// WebGPU has no multi draw indirect (outside of an extension), so every argument struct between
	// the provided pointer and the end of its allocation becomes its own indirect draw
	for(size_t i = 0; i < count; ++i)
		wgpuRenderPassEncoderDrawIndexedIndirect(cmd->render_pass, cmd->queue->monobuffers[range.buffer], start + i * sizeof(GpuDrawIndexedIndirectCommand));
}

void gpuDrawMeshlets(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, uvec3 dim) {
	throw std::runtime_error("WebGPU doesn't support mesh shaders!");
}

void gpuDrawMeshletsIndirect(GpuCommandBuffer* cmd, gpu* meshlet_data, gpu* fragment_data, gpu* dim, bool no_offsets /* = false */) {
	throw std::runtime_error("WebGPU doesn't support mesh shaders!");
}
