#pragma once

// The C interface plus the C++ conveniences, and the definitions behind the API's opaque handles.
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
		inline constexpr GpuErrorCallbackEXT error_callback = gpuDefaultErrorCallbackEXT;
	}
}

/**
 * gpuSetupDefaultWebGPUEXT – C++ flavour of the setup helper: takes any callable (a capturing lambda, say) as the surface loader and reports failure as the string a std::expected carries, rather than through out parameters.
 *
 * @code{.cpp}
 * auto webgpu = gpuSetupDefaultWebGPUEXT([](WGPUInstance) -> WGPUSurface { return nullptr; });
 * assert(webgpu);
 * wgpuDeviceRelease(webgpu->device);
 * wgpuAdapterRelease(webgpu->adapter);
 * wgpuInstanceRelease(webgpu->instance);
 * @endcode
 *
 * @param surface_loader Creates the presentation surface from the new instance. Returning no surface sets up a headless device that can still render into textures.
 * @param error_callback Where device and validation errors are reported.
 * @param prefer_high_power Ask for the discrete GPU rather than the integrated one.
 * @return The created objects, or why they could not be created (in which case nothing was).
 */
inline std::expected<GpuWebGPUDefault, std::string> gpuSetupDefaultWebGPUEXT(
	GPU::function_t<WGPUSurface(WGPUInstance)> surface_loader,
	GpuErrorCallbackEXT error_callback = gpuDefaultErrorCallbackEXT,
	bool prefer_high_power = true
) {
	// The callable crosses the C boundary as the userdata of a plain function
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
	WGPUBindGroup bind_group;	// binds {buffer, upload_buffer}
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

	GpuCapabilities capabilities = {};

	// Messages are deduplicated because they fire per draw, and one report per draw would bury the first.
	GpuDiagnosticCallbackEXT diagnostic_callback = nullptr;
	void* diagnostic_userdata = nullptr;
	std::set<std::string> reported_diagnostics;

	size_t next_submission_index = 1;
	// Kept current by wgpuQueueOnSubmittedWorkDone, so reclaiming deferred deletes needs no blocking readback (see GPU::process_pending_code).
	size_t last_finished_submission = 0;
	GpuSemaphore current_submission_timeline_semaphore;

	// Group 0 must fit the guaranteed 8 storage buffers: five monobuffers, the texture heap, and the sampler map (see SamplerSet::lookup_buffer).
	size_t active_monobuffer_size = 0, active_monobuffer_capacity = 0;
	WGPUBuffer empty_buffer = nullptr;
	std::array<WGPUBuffer, 5> monobuffers;
	std::array<size_t, 5> monobuffer_sizes;
	uint8_t active_monobuffer = 0;

	struct MonobufferRange {
		uint8_t buffer;
		uint32_t start, end;
		size_t size() const { return end - start; }
	};
	// Ordered, so "which allocation holds this address" is one upper_bound; an address's top bits are its monobuffer (see GPU::detail::closest_buffer).
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
		TEXTURE_USAGE_FLAGS usage = (TEXTURE_USAGE_FLAGS)0; ///< What the atlas has to be created for.

		// Dimensions round up to a power of two: textures up to that size share the atlas, each in the top left of its layers.
		static TextureHash from_descriptor(const GpuTextureDesc& desc) {
			auto out = GpuQueue::TextureHash{desc.type, desc.dimensions, desc.sampleCount, desc.format, desc.usage};
			out.dimensions.x = std::bit_ceil(desc.dimensions.x);
			out.dimensions.y = std::bit_ceil(desc.dimensions.y);
			out.dimensions.z = std::bit_ceil(desc.dimensions.z);
			return out;
		}

		bool operator==(const TextureHash& o) const {
			return type == o.type && dimensions.x == o.dimensions.x && dimensions.y == o.dimensions.y && dimensions.z == o.dimensions.z && sampleCount == o.sampleCount && format == o.format && usage == o.usage;
		}

		struct Hasher {
			uint64_t operator()(const TextureHash& t) const {
				return std::hash<size_t>{}(t.type) ^ std::hash<size_t>{}(t.dimensions.x) ^ std::hash<size_t>{}(t.dimensions.y) ^ std::hash<size_t>{}(t.dimensions.z)
					^ std::hash<size_t>{}(t.sampleCount) ^ std::hash<size_t>{}(t.format) ^ std::hash<size_t>{}(t.usage);
			}
		};
	};
	std::vector<std::tuple<uint32_t, WGPUTexture, WGPUTextureView, TextureHash>> storage_monotextures;
	std::unordered_map<TextureHash, size_t, TextureHash::Hasher> storage_monotextures_lookup;
	std::vector<std::tuple<uint32_t, WGPUTexture, WGPUTextureView, TextureHash>> sampled_monotextures;
	std::unordered_map<TextureHash, size_t, TextureHash::Hasher> sampled_monotextures_lookup;
	// Single mip views the current group 1 binds storage monotextures through, released with that group; see update_pipeline_layouts.
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
	// Monotextures emptied down to a 1x1 stand in, keyed like MonotextureRange::_index; gpuCreateTexture reuses their slots (see collapse_monotexture_if_empty).
	std::set<uint32_t> collapsed_monotextures;

	// Group 1 binds storage textures, group 2 sampled ones; a pass can't bind a texture as both, hence compute and graphics flavors (see update_pipeline_layouts).
	WGPUBindGroupLayout current_compute_bind_group_layout1 = nullptr;
	WGPUBindGroup current_compute_bind_group1 = nullptr;
	WGPUBindGroupLayout current_graphics_bind_group_layout1 = nullptr;
	WGPUBindGroup current_graphics_bind_group1 = nullptr;
	WGPUBindGroupLayout current_compute_bind_group_layout2 = nullptr;
	WGPUBindGroup current_compute_bind_group2 = nullptr;
	WGPUBindGroupLayout current_graphics_bind_group_layout2 = nullptr;
	WGPUBindGroup current_graphics_bind_group2 = nullptr;
	WGPUBindGroupLayout current_bind_group_layout3 = nullptr; // 3 == samplers

	// Everything one list of enabled samplers needs, cached per queue so alternating between two sets rebuilds neither.
	struct SamplerSet {
		std::vector<WGPUSampler> samplers; // exactly gpu_sampler_slot_count of them, padded with the default
		WGPUBindGroup bind_group = nullptr; // group 3

		// Maps a packed GpuSamplerDesc to its slot, or to 0 (the default sampler) when it was never enabled.
		WGPUBuffer lookup_buffer = nullptr;
	};
	std::unordered_map<std::vector<GpuSamplerDesc>, SamplerSet, GpuSamplerDescListHash> sampler_cache;
	SamplerSet* default_sampler_set = nullptr; // What a command buffer gets when it never enables any

	WGPUBindGroupLayout current_compute_bind_group_layout0 = nullptr; // 0 == buffers
	WGPUPipelineLayout current_compute_pipeline_layout = nullptr;

	WGPUBindGroupLayout current_graphics_bind_group_layout0 = nullptr; // 0 == buffers
	WGPUPipelineLayout current_graphics_pipeline_layout = nullptr;

	// gpuBlitTextureEXT's own layouts, built on the first blit; indexed by "filtering", since a filtering sampler can't touch an unfilterable format.
	WGPUShaderModule blit_shader = nullptr;
	std::array<WGPUBindGroupLayout, 2> blit_bind_group_layouts = {};
	std::array<WGPUPipelineLayout, 2> blit_pipeline_layouts = {};
	std::array<WGPUSampler, 2> blit_samplers = {};
	std::unordered_map<uint64_t, WGPURenderPipeline> blit_pipelines; // keyed by destination format and filtering
};

/**
 * gpuCreateQueue – Wrap the objects gpuSetupDefaultWebGPUEXT created as a GpuQueue.
 *
 * @code{.cpp}
 * GpuQueue* second = gpuCreateQueue(backend);
 * assert(second);
 * gpuFreeQueue(second);
 * @endcode
 *
 * @param def The objects to wrap.
 * @param allocator Host allocation hook.
 * @return The queue; see the C flavour in noapi.h.
 */
inline GpuQueue* gpuCreateQueue(GpuWebGPUDefault def, CpuAllocatorFunc allocator = default_::cpu_allocator) {
	return gpuCreateQueue(def.adapter, def.device, def.limits, allocator);
}

struct GpuCommandBuffer {
	GpuQueue* queue;

	// A snapshot of the heap: compute binds every monobuffer writable, and a pass can't bind one buffer both ways (see gpuSetActiveTextureHeapPtr).
	WGPUBuffer active_texture_heap = nullptr;
	size_t active_texture_heap_size = 0;

	const GpuQueue::SamplerSet* sampler_set = nullptr; // Null means the queue's default_sampler_set
	WGPUCommandEncoder encoder;
	WGPUComputePassEncoder compute_pass = nullptr;
	WGPURenderPassEncoder render_pass = nullptr;

	// WebGPU bakes these into the pipeline, so a draw rebuilds the bound pipeline when they disagree with what it was built with.
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
 * GPU_UV_UNCLAMPED – The uvMax a texture that fills its monotexture slot gets: far enough past 1 that the shader's clamp never bites, so the sampler's own addressing mode still reaches the whole range and a REPEAT sampler tiles the way it was asked to.
 */
constexpr static float GPU_UV_UNCLAMPED = 3.0e38f;

/**
 * GpuTextureDescriptorImpl – The 512 bits behind a GpuTextureDescriptor: everything a shader needs to sample the texture.
 *
 * uvScale and uvMax are what make the monotexture atlas invisible to a shader. A monotexture is sized by the first texture to land in its bucket, and the bucket keys on power of two rounded dimensions, so a smaller texture only covers the top left corner of its slot: uvScale maps the texture's own 0..1 onto that corner, and uvMax is the coordinate past which a bilinear tap would start reaching into the slot's leftovers. Both are computed once here rather than recovered in the shader, which is what lets gpuBackendSample skip a textureDimensions() and a divide per sample -- and uvMax is why nothing has to fill those leftovers in the first place.
 *
 * width/height/depth are the texture's own dimensions. Nothing in the sample path reads them any more now that uvScale carries the ratio, but they are what a heap entry means, and there is room.
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

		// The dynamic state the cached variant was built with; nothing is built until the first draw
		mutable std::optional<GpuDepthStencilDesc> depth_stencil = {};
		mutable std::optional<GpuBlendDesc> blend = {};
		mutable INDEX_TYPE_EXT index_type = INDEX_TYPE_UINT32; // Only relevant for strip topologies
		mutable WGPURenderPipeline pipeline = nullptr;
	};

	mutable WGPUPipelineLayout reference_layout; // The layout the cached variant was built against
	std::variant<ComputeCache, RenderCache> cache;
};

struct GpuSurface {
	WGPUSurface surface;

	// What wgpuSurfaceConfigure was actually given, narrowed to what the surface supports, so gpuSurfaceGetConfigurationEXT reports the truth.
	GpuSurfaceDescriptor descriptor;

	// WebGPU allocates presentable textures itself, so this has no monotexture range: it can be attached or blitted, nothing else.
	GpuTexture current = {};
	bool acquired = false;
};

/**
 * gpuCreateSurfaceEXT – Configure the surface gpuSetupDefaultWebGPUEXT created for presentation and wrap it as a GpuSurface.
 *
 * @code{.cpp}
 * auto surface = gpuCreateSurfaceEXT(queue, backend, {.texture = {.dimensions = {640, 480, 1}, .usage = USAGE_COLOR_ATTACHMENT}});
 * assert((surface != nullptr) == (backend.surface != nullptr));
 * if(surface) gpuFreeSurfaceEXT(queue, surface);
 * @endcode
 *
 * @param queue Queue the surface presents from.
 * @param def The objects holding the surface.
 * @param desc Requested configuration; what was actually granted is reported by gpuSurfaceGetConfigurationEXT.
 * @return The surface, or nullptr with errno set if the setup was headless.
 */
inline GpuSurface* gpuCreateSurfaceEXT(GpuQueue* queue, GpuWebGPUDefault def, const GpuSurfaceDescriptor& desc) {
	return gpuCreateSurfaceEXT(queue, def.surface, desc);
}
