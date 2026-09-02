#include "noapi.hpp"
#include "common.hpp"

#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>
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

void update_pipeline_layouts(GpuQueue* queue, bool compute) {
	auto visibility = compute ? WGPUShaderStage_Compute : WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;

	auto create_layout = [&](std::vector<WGPUBindGroupLayoutEntry>& entries) {
		WGPUBindGroupLayoutDescriptor d{
			.entryCount = static_cast<uint32_t>(entries.size()),
			.entries = entries.data(),
		};

		return wgpuDeviceCreateBindGroupLayout(queue->device, &d);
	};

	auto& bg0 = compute ? queue->current_compute_bind_group_layout0 : queue->current_graphics_bind_group_layout0;

	//
	// Group 0: buffers
	//
	uint32_t binding = 0;

	if(!bg0) { // The layout of buffers isn't dynamic so if it already exists we don't need to generate it again!
		std::vector<WGPUBindGroupLayoutEntry> group0;

		for(uint32_t i = 0; i < 6; ++i)
			group0.push_back({
				.binding = binding++,
				.visibility = visibility,
				.buffer = {
					.type = compute ? WGPUBufferBindingType_Storage : WGPUBufferBindingType_ReadOnlyStorage,
					.hasDynamicOffset = false,
					.minBindingSize = 0,
				}
			});

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

		bg0 = create_layout(group0);
	}

	//
	// Group 1: storage textures
	//
	std::vector<WGPUBindGroupLayoutEntry> group1;

	binding = 0;

	for(auto const& [_cap, texture, view, desc] : queue->storage_monotextures)
		group1.push_back({
			.binding = binding++,
			.visibility = visibility,
			.storageTexture = {
				.access = WGPUStorageTextureAccess_ReadWrite,
				.format = GPU::format2wgpu(desc.format),
				.viewDimension = GPU::texture_view2wgpu(desc.type),
			}
		});

	//
	// Group 2: sampled textures
	//
	std::vector<WGPUBindGroupLayoutEntry> group2;

	binding = 0;

	for(auto const& [_cap, texture, view, desc] : queue->sampled_monotextures)
		group2.push_back({
			.binding = binding++,
			.visibility = visibility,
			.texture = {
				.sampleType = WGPUTextureSampleType_Float,
				.viewDimension = GPU::texture_view2wgpu(desc.type),
				.multisampled = desc.sampleCount > 1,
			}
		});

	auto& bg1 = compute ? queue->current_compute_bind_group_layout1 : queue->current_graphics_bind_group_layout1;
	if(bg1) queue->code_pending_submission_finished.emplace_back([l = bg1](){ wgpuBindGroupLayoutRelease(l); }, queue->next_submission_index);
	bg1 = create_layout(group1);

	auto& bg2 = compute ? queue->current_compute_bind_group_layout2 : queue->current_graphics_bind_group_layout2;
	if(bg2) queue->code_pending_submission_finished.emplace_back([l = bg2](){ wgpuBindGroupLayoutRelease(l); }, queue->next_submission_index);
	bg2 = create_layout(group2);

	WGPUBindGroupLayout layouts[] = {
		bg0,
		bg1,
		bg2
	};
	auto& pipeline_layout = compute ? queue->current_compute_pipeline_layout : queue->current_graphics_pipeline_layout;
	if(pipeline_layout) wgpuPipelineLayoutRelease(pipeline_layout);

	WGPUPipelineLayoutDescriptor pd{
		.bindGroupLayoutCount = 3,
		.bindGroupLayouts = layouts,
	};
	pipeline_layout = wgpuDeviceCreatePipelineLayout(queue->device, &pd);
}
void update_pipeline_layouts(GpuQueue* queue) {
	update_pipeline_layouts(queue, true);
	update_pipeline_layouts(queue, false);
}



std::string generate_binding_prologue(GpuQueue* queue, bool compute) {
	constexpr static auto texture_dimension_suffix = [](TEXTURE type) {
		switch (type) {
		case TEXTURE_1D: return "1d";
		// case TEXTURE_2D: return "2d";
		case TEXTURE_3D: return "3d";
		case TEXTURE_CUBE: return "cube";
		case TEXTURE_2D_ARRAY: return "2d_array";
		case TEXTURE_CUBE_ARRAY: return "cube_array";
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

	std::string out = "@group(0) @binding(0) var<storage, read_write> mono0 : array<u32>;\n"
	"@group(0) @binding(1) var<storage, read_write> mono1 : array<u32>;\n"
	"@group(0) @binding(2) var<storage, read_write> mono2 : array<u32>;\n"
	"@group(0) @binding(3) var<storage, read_write> mono3 : array<u32>;\n"
	"@group(0) @binding(4) var<storage, read_write> mono4 : array<u32>;\n"
	"@group(0) @binding(5) var<storage, read_write> mono5 : array<u32>;\n"
	"\n"
	"@group(0) @binding(6) var<storage> texture_heap : array<u32>;\n"
	"struct GPUComputeData {\n"
	"	upload_buffer: vec2<u32>,\n"
	"	download_buffer: vec2<u32>\n"
	"}\n"
	"@group(0) @binding(7) var<uniform> shader_data : GPUComputeData;\n";

	uint32_t binding = 0;

	for (auto const& [_cap, texture, view, desc] : queue->storage_monotextures) {
		out += std::format("@group(1) @binding({}) var storage_tex_{} : texture_storage_{}<{}, read_write>;\n", binding, binding, texture_dimension_suffix(desc.type), wgsl_format(desc.format));
		++binding;
	}

	binding = 0;

	for (auto const& [_cap, texture, view, desc] : queue->sampled_monotextures) {
		out += std::format("@group(2) @binding({}) var sampled_tex_{} : texture_{}<f32>;\n", binding, binding, texture_dimension_suffix(desc.type));
		++binding;
	}

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

	out->queue = wgpuDeviceGetQueue(device);

	out->current_submission_timeline_semaphore = GPU::semaphore_initialize(out, out->next_submission_index - 1);

	WGPUBufferDescriptor d {
		.label = {"NoAPI Empty Monobuffer", WGPU_STRLEN},
		.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
		.size = 1
	};
	out->empty_monobuffer = wgpuDeviceCreateBuffer(out->device, &d);
	std::fill(out->monobuffers.begin(), out->monobuffers.end(), out->empty_monobuffer);

	update_pipeline_layouts(out);
	return out;
}

void gpuFreeQueue(GpuQueue* queue) {
	GPU::semaphore_destroy(queue->current_submission_timeline_semaphore);

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

	auto new_start = align_up(queue->monobuffer_size, align);
	auto new_end = new_start + bytes;
	auto new_active_monobuffer = queue->active_monobuffer;

	// Reallocate the monobuffer if necessary
	bool activate_next_monobuffer = false;
	if(new_end > queue->monobuffer_capacity) {
		if(queue->monobuffers[queue->active_monobuffer] != queue->empty_monobuffer)
			queue->code_pending_submission_finished.emplace_back([buffer = queue->monobuffers[queue->active_monobuffer]]() {
				wgpuBufferRelease(buffer);
			}, queue->next_submission_index);
		else queue->monobuffer_capacity = new_end;

		queue->monobuffer_capacity = std::max(queue->monobuffer_capacity * 2, new_end); // Doubles the size of the capacity each time
		if(queue->monobuffer_capacity > queue->limits.maxStorageBufferBindingSize) {
			queue->monobuffer_capacity = queue->limits.maxStorageBufferBindingSize;
			activate_next_monobuffer = true;
		}
		if(new_end > queue->limits.maxStorageBufferBindingSize) {
			new_start = 0;
			new_end = bytes;
			++new_active_monobuffer;
		}

		WGPUBufferDescriptor d {
			.label = {"NoAPI Monobuffer", WGPU_STRLEN},
			.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
			.size = queue->monobuffer_capacity
		};
		auto old_monobuffer = queue->monobuffers[queue->active_monobuffer];
		queue->monobuffers[queue->active_monobuffer] = wgpuDeviceCreateBuffer(queue->device, &d);

		if(old_monobuffer != queue->empty_monobuffer) {
			auto cmd = wgpuDeviceCreateCommandEncoder(queue->device, nullptr);
			wgpuCommandEncoderCopyBufferToBuffer(cmd, old_monobuffer, 0, queue->monobuffers[queue->active_monobuffer], 0, queue->monobuffer_size);
			auto to_submit = wgpuCommandEncoderFinish(cmd, nullptr);
			wgpuQueueSubmit(queue->queue, 1, &to_submit);
			wgpuCommandEncoderRelease(cmd);
			wgpuCommandBufferRelease(to_submit);
		}

		if(activate_next_monobuffer) {
			// Add the rest of the old monobuffer to the freelist
			queue->buffer_freelist.emplace_back(queue->active_monobuffer, queue->monobuffer_size, queue->limits.maxStorageBufferBindingSize);

			++queue->active_monobuffer;
			if(queue->active_monobuffer >= queue->monobuffers.size()) { // Fail if monobuffer being activated is the 8th monobuffer
				errno = WGPUErrorType_OutOfMemory;
				return nullptr;
			}

			WGPUBufferDescriptor d {
				.label = {"NoAPI Monobuffer", WGPU_STRLEN},
				.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
				.size = bytes
			};
			queue->monobuffers[queue->active_monobuffer] = wgpuDeviceCreateBuffer(queue->device, &d);

			if(new_start == 0)
				queue->monobuffer_size = 0;
			else queue->monobuffer_size = bytes;
			queue->monobuffer_capacity = bytes;
		}
	}

	if(!activate_next_monobuffer) queue->monobuffer_size = new_end;
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
		out->texture = std::get<WGPUTexture>((*monotextures)[range->_index]);
		// TODO: Texture view?
	} else { // If we are creating a purely bound texture rather than a lookup texture we are fine
		WGPUTextureDescriptor d {
			.usage = GPU::usage2wgpu(desc.usage),
			.dimension = GPU::texture2wgpu(desc.type),
			.size = {desc.dimensions.x, desc.dimensions.y, desc.type == TEXTURE_3D ? desc.dimensions.z : desc.layerCount},
			.format = GPU::format2wgpu(desc.format),
			.mipLevelCount = static_cast<uint32_t>(std::log2(std::max(desc.dimensions.x, desc.dimensions.y)) + 1),
		};
		out->texture = wgpuDeviceCreateTexture(queue->device, &d);
		// TODO: Texture view?
	}

	queue->gpu2textures[memory] = out;
	return out;
}

GpuTextureDescriptor gpuTextureViewDescriptor(GpuQueue* queue, const GpuTexture* texture, const GpuViewDesc& desc) {
	GpuTextureDescriptorImpl out {
		.type = texture->descriptor.type,
		.width = texture->descriptor.dimensions.x,
		.height = texture->descriptor.dimensions.y,
		.baseMip = desc.baseMip,
		.mipCount = desc.mipCount == ALL_MIPS ? texture->descriptor.mipCount - desc.baseMip : desc.mipCount,
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

void update_compute_pipeline(GpuQueue* queue, GpuPipeline* pipeline) {
	auto& cache = std::get<GpuPipeline::ComputeCache>(pipeline->cache);
	// if(cache.module) wgpuShaderModuleRelease(cache.module);
	if(cache.pipeline) wgpuComputePipelineRelease(cache.pipeline);

	constexpr std::string_view marker = "@generated_noapi_bindings";
	std::string wgsl_source = cache.IR;
	auto generated_bindings = generate_binding_prologue(queue, true);
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
	WGPUComputePipelineDescriptor d {
		.layout = pipeline->reference_layout = queue->current_compute_pipeline_layout,
		.compute = {
			.module = wgpuDeviceCreateShaderModule(queue->device, &module),
			.entryPoint = {"main", WGPU_STRLEN},
		}
	};
	cache.pipeline = wgpuDeviceCreateComputePipeline(queue->device, &d);

	wgpuShaderModuleRelease(d.compute.module);
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

void gpuFreePipeline(GpuQueue* queue, GpuPipeline* pipeline) {
	if(std::holds_alternative<GpuPipeline::ComputeCache>(pipeline->cache)) {
		auto& cache = std::get<GpuPipeline::ComputeCache>(pipeline->cache);
		// wgpuShaderModuleRelease(cache.module);
		wgpuComputePipelineRelease(cache.pipeline);
	} else {
		// TODO: implement graphics side
	}

	pipeline->~GpuPipeline();
	queue->cpu_allocator(pipeline, 0);
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

uint64_t gpuSubmitNoFree(GpuQueue* queue, std::span<GpuCommandBuffer*> command_buffers, GpuSemaphore* semaphore /* = nullptr */, uint64_t signal_value /* = 0 */) {
	std::vector<WGPUCommandBuffer> buffers; buffers.reserve(command_buffers.size() + 1);
	for(auto buffer: command_buffers)
		buffers.emplace_back(wgpuCommandEncoderFinish(buffer->encoder, nullptr));

	GpuCommandBuffer cmd {
		.queue = queue,
		.encoder = wgpuDeviceCreateCommandEncoder(queue->device, nullptr),
	};
	GPU::semaphore_gpu_increment(&cmd, queue->current_submission_timeline_semaphore);

	// TODO: GpuSemaphore* semaphore /* = nullptr */ stuff

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

void gpuWaitIdleEXT(GpuQueue* queue);

void gpuSyncMemoryEXT(GpuCommandBuffer* cmd, gpu* mem) {
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