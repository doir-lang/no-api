#pragma once

// The backend's C interface: the setup helper, the queue and surface constructors, the
// address helpers and the constants. This file adds the C++ conveniences on top of it,
// along with the definitions of the objects behind the API's opaque handles.
#include "noapi.h"

#include <expected>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <map>
#include <set>
#include <array>
#include <variant>
#include <vector>
#include <bit>

namespace GPU {
#ifdef __cpp_lib_function_ref
	template<typename T>
	using function_t = std::function_ref<T>;
#else
	template<typename T>
	using function_t = std::function<T>;
#endif

	namespace default_ {
		// The C hook, under the name the rest of the C++ code knows it by
		inline constexpr GpuErrorCallbackEXT error_callback = gpuDefaultErrorCallbackEXT;
	}
}

/**
 * gpuSetupDefaultWebGPUEXT – C++ flavour of the setup helper: takes any callable (a
 * capturing lambda, say) as the surface loader and reports failure as the string a
 * std::expected carries, rather than through out parameters.
 *
 * @param surface_loader Creates the presentation surface from the new instance.
 * @param error_callback Where device and validation errors are reported.
 * @param prefer_high_power Ask for the discrete GPU rather than the integrated one.
 */
inline std::expected<GpuWebGPUDefault, std::string> gpuSetupDefaultWebGPUEXT(
	GPU::function_t<WGPUSurface(WGPUInstance)> surface_loader,
	GpuErrorCallbackEXT error_callback = gpuDefaultErrorCallbackEXT,
	bool prefer_high_power = true
) {
	// The loader is handed across the C boundary as a plain function and the callable it
	// stands for rides along as the userdata
	auto load_surface = +[](WGPUInstance instance, void* userdata) -> WGPUSurface {
		return (*(GPU::function_t<WGPUSurface(WGPUInstance)>*)userdata)(instance);
	};

	GpuWebGPUDefault out = {};
	char error[512] = {};
	if(!gpuSetupDefaultWebGPUEXT(load_surface, &surface_loader, error_callback, prefer_high_power, &out, error, sizeof(error)))
		return std::unexpected(std::string(error));
	return out;
}

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

	// What this backend can do, for the parts of the API it can't provide. Filled in by
	// gpuCreateQueue and handed out by gpuGetCapabilitiesEXT.
	GpuCapabilities capabilities = {};

	// Where unsupported and emulated calls are reported, and the messages already reported.
	// Deduplicated because these fire from per draw calls, and one report per draw would bury the
	// first one.
	GpuDiagnosticCallbackEXT diagnostic_callback = nullptr;
	void* diagnostic_userdata = nullptr;
	std::set<std::string> reported_diagnostics;

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
	// Ordered rather than hashed: the lookup that matters is "which allocation holds this address",
	// which an ordered map answers with one upper_bound instead of a scan over every live
	// allocation. An address carries its monobuffer in its top bits, so integer order is
	// (monobuffer, offset) order (see GPU::detail::closest_buffer).
	std::map<gpu*, std::tuple<MonobufferRange, void*, MEMORY>> allocations;
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
	// The single mip level views the current group 1 binds the storage monotextures through, which
	// are not the whole chain views above; see update_pipeline_layouts. Released with that group.
	std::vector<WGPUTextureView> current_storage_views;

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
	// The monotextures every texture has been freed out of, keyed the way a MonotextureRange's
	// _index is (storage flag in the top bit, index in the rest). One of these holds a 1x1 stand in
	// rather than an atlas and no hash points at it any more, so gpuCreateTexture takes its slot
	// over for the next monotexture that needs one; see collapse_monotexture_if_empty.
	std::set<uint32_t> collapsed_monotextures;


	// 1 == storage textures and 2 == sampled ones, each in a compute flavor and a graphics one:
	// no pass may have a texture bound as both at once, so the storage bindings are only ever in
	// the first and the sampled views of those same monotextures only in the second. See
	// update_pipeline_layouts.
	WGPUBindGroupLayout current_compute_bind_group_layout1 = nullptr;
	WGPUBindGroup current_compute_bind_group1 = nullptr;
	WGPUBindGroupLayout current_graphics_bind_group_layout1 = nullptr;
	WGPUBindGroup current_graphics_bind_group1 = nullptr;
	WGPUBindGroupLayout current_compute_bind_group_layout2 = nullptr;
	WGPUBindGroup current_compute_bind_group2 = nullptr;
	WGPUBindGroupLayout current_graphics_bind_group_layout2 = nullptr;
	WGPUBindGroup current_graphics_bind_group2 = nullptr;
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

/**
 * GPU_UV_UNCLAMPED – The uvMax a texture that fills its monotexture slot gets: far enough past 1
 * that the shader's clamp never bites, so the sampler's own addressing mode still reaches the
 * whole range and a REPEAT sampler tiles the way it was asked to.
 */
constexpr static float GPU_UV_UNCLAMPED = 3.0e38f;

/**
 * The 512 bits behind a GpuTextureDescriptor: everything a shader needs to sample the texture.
 *
 * uvScale and uvMax are what make the monotexture atlas invisible to a shader. A monotexture is
 * sized by the first texture to land in its bucket, and the bucket keys on power of two rounded
 * dimensions, so a smaller texture only covers the top left corner of its slot: uvScale maps the
 * texture's own 0..1 onto that corner, and uvMax is the coordinate past which a bilinear tap would
 * start reaching into the slot's leftovers. Both are computed once here rather than recovered in
 * the shader, which is what lets gpuBackendSample skip a textureDimensions() and a divide per
 * sample -- and uvMax is why nothing has to fill those leftovers in the first place.
 *
 * width/height/depth are the texture's own dimensions. Nothing in the sample path reads them any
 * more now that uvScale carries the ratio, but they are what a heap entry means, and there is room.
 */
struct GpuTextureDescriptorImpl {
	uint32_t type = TEXTURE_2D; ///< Dimensionality and view type (a TEXTURE).
	uint32_t baseMip = 0; ///< First mip level the view covers.
	uint32_t mipCount = 1; ///< Number of mip levels it covers.
	uint32_t _reserved = 0;
	uint32_t width = 1, height = 1, depth = 1; ///< The texture's own dimensions in texels.
	GpuQueue::MonotextureRange range;
	float uvScale[3] = {1, 1, 1}; ///< The texture's share of its monotexture, per axis.
	float uvMax[3] = {GPU_UV_UNCLAMPED, GPU_UV_UNCLAMPED, GPU_UV_UNCLAMPED}; ///< Half a texel inside that share.
};
static_assert(sizeof(GpuTextureDescriptorImpl) == sizeof(GpuTextureDescriptor), "GPU Texture Descriptors of The Wrong Size");

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

inline GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, GpuWebGPUDefault def, const GpuSurfaceDescriptor& desc) {
	return gpuCreateSurfaceEXT(queue, def.surface, desc);
}