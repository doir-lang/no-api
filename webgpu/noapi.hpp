#pragma once

#include "../surface.hpp"
#include "../sync.hpp"
#include "../samplers.hpp"
#include "../allocator.hpp"

#include <expected>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
// #include <map>
#include <array>
#include <variant>
#include <vector>
#include <bit>

#include <webgpu/webgpu.h>

namespace GPU {
#ifdef __cpp_lib_function_ref
	template<typename T>
	using function_t = std::function_ref<T>;
#else
	template<typename T>
	using function_t = std::function<T>;
#endif

	namespace default_ {
		void error_callback(void* queue, int type, std::string_view message);
	}
}

struct GpuWebGPUDefault {
	WGPUInstance instance;
	WGPUAdapter adapter;
	WGPULimits limits;
	WGPUDevice device;
	WGPUSurface surface;
};
std::expected<GpuWebGPUDefault, std::string> gpuSetupDefaultWebGPUEXT(GPU::function_t<WGPUSurface(WGPUInstance)> surface_loader, void(*error_callback)(void* queue, int type, std::string_view message) = GPU::default_::error_callback, bool prefer_high_power = true);

// WebGPU can't index an array of samplers, so every slot gets its own binding and shaders pick
// between them with a switch over the slot the lookup map handed back. Shaders are compiled against
// however many slots exist, so this is the 16 maxSamplersPerShaderStage guarantees rather than
// whatever a particular device offers.
constexpr static uint32_t gpu_sampler_slot_count = 16;

gpu* gpuEncodeWebGPUAddressEXT(uint8_t monobuffer, uint64_t address);
std::pair<uint8_t, uint64_t> gpuDecodeWebGPUAddressEXT(gpu* addr);\

struct GpuSemaphore {
	WGPUBuffer buffer;			// 8 bytes, Storage | CopySrc | CopyDst
	WGPUBuffer readback_buffer;	// 8 bytes, MapRead | CopyDst
	WGPUBuffer upload_buffer;	// 8 bytes, Uniform | CopyDst (input to cs_set)
	WGPUBindGroup bind_group;	// binds {buffer, valueUniformBuffer}
};

struct GpuQueue {
	WGPUAdapter adapter;
	WGPUDevice device;
	WGPULimits limits;
	WGPUQueue queue;
	CpuAllocatorFunc cpu_allocator;

	WGPUBindGroupLayout semaphore_bind_group_layout = nullptr;
	WGPUComputePipeline semaphore_increment_pipeline = nullptr;
	WGPUComputePipeline semaphore_set_pipeline = nullptr;
	WGPUComputePipeline semaphore_set_max_pipeline = nullptr;

	size_t next_submission_index = 1;
	// The highest submission index the queue has reported finished, kept up to date by the
	// wgpuQueueOnSubmittedWorkDone callback gpuSubmitNoFree registers. Reclaiming deferred deletes
	// reads this instead of the timeline semaphore, which would mean a blocking readback (see
	// GPU::process_pending_code).
	size_t last_finished_submission = 0;
	GpuSemaphore current_submission_timeline_semaphore;

	// Group 0 has to fit inside maxStorageBuffersPerShaderStage, whose guaranteed minimum is 8.
	// Five of those slots are monobuffers, the sixth is the active texture heap, and the seventh is
	// the sampler lookup map (see SamplerSet::lookup_buffer).
	size_t active_monobuffer_size = 0, active_monobuffer_capacity = 0;
	WGPUBuffer empty_buffer = nullptr;
	std::array<WGPUBuffer, 5> monobuffers;
	std::array<size_t, 5> monobuffer_sizes;
	// static_assert(GpuQueue{}.monobuffers.size() == GpuQueue{}.monobuffer_sizes.size());
	uint8_t active_monobuffer = 0;

	struct MonobufferRange {
		uint8_t buffer;
		uint32_t start, end;
		size_t size() const { return end - start; }
	};
	std::unordered_map<gpu*, std::tuple<MonobufferRange, void*, MEMORY>> allocations;
	std::unordered_map<void*, gpu*> cpu2gpu;
	std::unordered_map<gpu*, GpuTexture*> gpu2textures;
	std::vector<MonobufferRange> buffer_freelist;

	std::vector<std::pair<std::function<void()>, size_t>> code_pending_submission_finished;

	struct TextureHash {
		TEXTURE type = TEXTURE_2D; ///< Dimensionality and view type.
		uvec3 dimensions = {1, 1, 1}; ///< Dimensions in texels.
		uint32_t sampleCount = 1; ///< MSAA sample count.
		FORMAT format = FORMAT_NONE; ///< Texel format.

		static TextureHash from_descriptor(const GpuTextureDesc& desc) {
			auto out = GpuQueue::TextureHash{desc.type, desc.dimensions, desc.sampleCount, desc.format};
			out.dimensions.x = std::bit_ceil(desc.dimensions.x);
			out.dimensions.y = std::bit_ceil(desc.dimensions.y);
			out.dimensions.z = std::bit_ceil(desc.dimensions.z);
			return out;
		}

		bool operator==(const TextureHash& o) const {
			return type == o.type && dimensions.x == o.dimensions.x && dimensions.y == o.dimensions.y && dimensions.z == o.dimensions.z && sampleCount == o.sampleCount && format == o.format;
		}

		struct Hasher {
			uint64_t operator()(const TextureHash& t) const {
				return std::hash<size_t>{}(t.type) ^ std::hash<size_t>{}(t.dimensions.x) ^ std::hash<size_t>{}(t.dimensions.y) ^ std::hash<size_t>{}(t.dimensions.z)
					^ std::hash<size_t>{}(t.sampleCount) ^ std::hash<size_t>{}(t.format);
			}
		};
		// struct Comparator {
		// 	bool operator()(const TextureHash& a, const TextureHash& b) const {
		// 		if (a.type != b.type)
		// 			return a.type < b.type;

		// 		if (a.dimensions.x != b.dimensions.x)
		// 			return a.dimensions.x < b.dimensions.x;

		// 		if (a.dimensions.y != b.dimensions.y)
		// 			return a.dimensions.y < b.dimensions.y;

		// 		if (a.dimensions.z != b.dimensions.z)
		// 			return a.dimensions.z < b.dimensions.z;

		// 		if (a.sampleCount != b.sampleCount)
		// 			return a.sampleCount < b.sampleCount;

		// 		return a.format < b.format;
		// 	}
		// };
	};
	std::vector<std::tuple<uint32_t, WGPUTexture, WGPUTextureView, TextureHash>> storage_monotextures;
	std::unordered_map<TextureHash, size_t, TextureHash::Hasher> storage_monotextures_lookup;
	std::vector<std::tuple<uint32_t, WGPUTexture, WGPUTextureView, TextureHash>> sampled_monotextures;
	std::unordered_map<TextureHash, size_t, TextureHash::Hasher> sampled_monotextures_lookup;

	struct MonotextureRange {
		const static MonotextureRange INVALID; // = {-1, -1, -1}
		static constexpr uint32_t STORAGE_BIT = 0x80000000u;
		static constexpr uint32_t INDEX_MASK  = 0x7FFFFFFFu;

		uint32_t _index; // top bit stores storage (versus sampled) flag
		uint32_t start, end;

		bool storage() const {
			return (_index & STORAGE_BIT) != 0;
		}

		void set_storage(bool storage) {
			if (storage)
				_index |= STORAGE_BIT;
			else
				_index &= INDEX_MASK;
		}

		uint32_t index() const {
			return _index & INDEX_MASK;
		}

		void set_index(uint32_t value) {
			_index = (_index & STORAGE_BIT) | (value & INDEX_MASK);
		}
	};
	std::vector<MonotextureRange> texture_freelist;


	WGPUBindGroupLayout current_bind_group_layout1 = nullptr; // 1 == storage textures
	WGPUBindGroup current_bind_group1 = nullptr;
	WGPUBindGroupLayout current_bind_group_layout2 = nullptr; // 2 == sampled textures
	WGPUBindGroup current_bind_group2 = nullptr;
	WGPUBindGroupLayout current_bind_group_layout3 = nullptr; // 3 == samplers

	// Everything gpuSetEnabledSamplersEXT produces for one list of enabled samplers. It is cached per
	// queue (and pointed at by the command buffers that enabled it) so that alternating between two
	// sampler sets doesn't rebuild either of them.
	struct SamplerSet {
		std::vector<WGPUSampler> samplers; // exactly gpu_sampler_slot_count of them, padded with the default
		WGPUBindGroup bind_group = nullptr; // group 3

		// Maps a packed GpuSamplerDesc onto the slot holding that sampler, or onto 0 (the default
		// sampler) when it was never enabled. A pure function of which samplers are enabled, so it is
		// a fixed size and gets written once, when the set is built.
		WGPUBuffer lookup_buffer = nullptr;
	};
	std::unordered_map<std::vector<GpuSamplerDesc>, SamplerSet, GpuSamplerDescListHash> sampler_cache;
	SamplerSet* default_sampler_set = nullptr; // What a command buffer gets when it never enables any

	WGPUBindGroupLayout current_compute_bind_group_layout0 = nullptr; // 0 == buffers
	WGPUPipelineLayout current_compute_pipeline_layout = nullptr;

	WGPUBindGroupLayout current_graphics_bind_group_layout0 = nullptr; // 0 == buffers
	WGPUPipelineLayout current_graphics_pipeline_layout = nullptr;

	// gpuBlitTextureEXT's internal pipeline. It binds nothing the rest of the API knows about
	// (just the source texture, a sampler, and the sampled rectangle), so it gets its own layouts
	// and is built lazily the first time a blit is recorded. Everything indexed by "filtering"
	// exists in a filtering and a non-filtering flavor, since WebGPU refuses to let a filtering
	// sampler touch a texture whose format it can't interpolate.
	WGPUShaderModule blit_shader = nullptr;
	std::array<WGPUBindGroupLayout, 2> blit_bind_group_layouts = {};
	std::array<WGPUPipelineLayout, 2> blit_pipeline_layouts = {};
	std::array<WGPUSampler, 2> blit_samplers = {};
	std::unordered_map<uint64_t, WGPURenderPipeline> blit_pipelines; // keyed by destination format and filtering
};

GpuQueue* gpuCreateQueue(WGPUAdapter adapter, WGPUDevice device, WGPULimits limits, CpuAllocatorFunc allocator = default_::cpu_allocator);
inline GpuQueue* gpuCreateQueue(GpuWebGPUDefault def, CpuAllocatorFunc allocator = default_::cpu_allocator) {
	return gpuCreateQueue(def.adapter, def.device, def.limits, allocator);
}

struct GpuCommandBuffer {
	GpuQueue* queue;

	// A snapshot of the active texture heap rather than the monobuffer range it lives in: group 0
	// binds every monobuffer writable for compute, and WebGPU won't have a buffer bound writable and
	// read only in the same pass. See gpuSetActiveTextureHeapPtr.
	WGPUBuffer active_texture_heap = nullptr;
	size_t active_texture_heap_size = 0;

	const GpuQueue::SamplerSet* sampler_set = nullptr; // Null means the queue's default_sampler_set
	WGPUCommandEncoder encoder;
	WGPUComputePassEncoder compute_pass = nullptr;
	WGPURenderPassEncoder render_pass = nullptr;

	// WebGPU bakes the depth/stencil state, the blend state, and (for strip topologies) the index
	// format into the pipeline, so all of them are tracked here and compared against what the bound
	// pipeline's cached variant was built with. Whenever they disagree the pipeline gets rebuilt.
	const GpuPipeline* bound_pipeline = nullptr;
	WGPURenderPipeline bound_render_pipeline = nullptr; // Whatever is currently set on render_pass
	std::optional<GpuDepthStencilDesc> depth_stencil = {};
	std::optional<GpuBlendDesc> blend = {};
	INDEX_TYPE_EXT index_type = INDEX_TYPE_UINT32;

	std::vector<std::function<void()>> code_pending_submission_finished;
};

struct GpuDepthStencilState {
	GpuDepthStencilDesc descriptor;
};

struct GpuBlendState {
	GpuBlendDesc descriptor;
};

struct GpuTexture {
	GpuTextureDesc descriptor;
	std::optional<GpuQueue::MonotextureRange> range = {};
	WGPUTexture texture;
};

// The 256 bits behind a GpuTextureDescriptor. Everything a shader needs to sample the texture:
// which monotexture holds it, which layers of it are the view's, how big the texture itself is (the
// monotexture is bigger, and WGSL can report its size on its own), and which mips are in range.
//
// type, baseMip and mipCount share a word. None of them needs more than a byte, and the space buys
// depth, without which a 3D view couldn't scale its coordinates.
struct GpuTextureDescriptorImpl {
	uint8_t type = TEXTURE_2D; ///< Dimensionality and view type (a TEXTURE).
	uint8_t baseMip = 0; ///< First mip level the view covers.
	uint8_t mipCount = 1; ///< Number of mip levels it covers.
	uint8_t _reserved0 = 0;
	uint32_t width = 1, height = 1, depth = 1; ///< The texture's own dimensions in texels.
	GpuQueue::MonotextureRange range;
	uint32_t _reserved1 = 0;
};
static_assert(sizeof(GpuTextureDescriptorImpl) == sizeof(GpuTextureDescriptor), "GPU Texture Descriptors of The Wrong Size");

// Statuses gpuSurfaceNextTextureEXT leaves in errno. A suboptimal texture is still perfectly usable
// (it just no longer matches the window), an out of date one means the configuration has to be redone.
constexpr static int SURFACE_SUBOPTIMAL = WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal;
constexpr static int SURFACE_OUT_OF_DATE = WGPUSurfaceGetCurrentTextureStatus_Outdated;

struct GpuPipeline {
	struct ComputeCache {
		std::string IR;
		mutable WGPUComputePipeline pipeline;
	};

	struct RenderCache {
		std::string vertexIR, fragmentIR;
		std::vector<GpuColorTarget> color_targets; // Backing storage for descriptor.colorTargets
		GpuRasterDesc descriptor;

		// The dynamic state the currently cached variant was built against (nothing is cached until pipeline is set)
		mutable std::optional<GpuDepthStencilDesc> depth_stencil = {};
		mutable std::optional<GpuBlendDesc> blend = {};
		mutable INDEX_TYPE_EXT index_type = INDEX_TYPE_UINT32; // Only relevant for strip topologies
		mutable WGPURenderPipeline pipeline = nullptr;
	};

	mutable WGPUPipelineLayout reference_layout; // The layout that the currently cached variant of this pipeline is built against
	std::variant<ComputeCache, RenderCache> cache;
};

struct GpuSurface {
	WGPUSurface surface;

	// What wgpuSurfaceConfigure was actually given, which is not necessarily what the user asked for:
	// the format, present mode, usage and alpha mode are all narrowed to what the surface supports so
	// that gpuSurfaceGetConfigurationEXT reports the truth rather than the request.
	GpuSurfaceDescriptor descriptor;

	// The texture wgpuSurfaceGetCurrentTexture handed back, wrapped for the rest of the API. Its range
	// stays empty: WebGPU allocates presentable textures itself, so there is no way to place one
	// inside a monotexture and therefore no way to name it from a GpuTextureDescriptor. It can be
	// attached to a render pass (or blitted to and from) and nothing else.
	GpuTexture current = {};
	bool acquired = false;
};

GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, WGPUSurface surface, const GpuSurfaceDescriptor& desc);
inline GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, GpuWebGPUDefault def, const GpuSurfaceDescriptor& desc) {
	return gpuCreateSurfaceEXT(queue, def.surface, desc);
}