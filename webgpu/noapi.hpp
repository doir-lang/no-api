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
	GpuSemaphore current_submission_timeline_semaphore;

	size_t monobuffer_size = 0, monobuffer_capacity = 0;
	WGPUBuffer empty_monobuffer = nullptr;
	std::array<WGPUBuffer, 6> monobuffers;
	uint8_t active_monobuffer = 0;

	struct MonobufferRange {
		uint8_t buffer;
		uint32_t start, end;
		size_t size() { return end - start; }
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


	WGPUBindGroupLayout current_compute_bind_group_layout0 = nullptr; // 0 == buffers
	WGPUBindGroupLayout current_compute_bind_group_layout1 = nullptr; // 1 == storage textures
	WGPUBindGroupLayout current_compute_bind_group_layout2 = nullptr; // 2 == sampled textures
	WGPUPipelineLayout current_compute_pipeline_layout = nullptr;

	WGPUBindGroupLayout current_graphics_bind_group_layout0 = nullptr; // 0 == buffers
	WGPUBindGroupLayout current_graphics_bind_group_layout1 = nullptr; // 1 == storage textures
	WGPUBindGroupLayout current_graphics_bind_group_layout2 = nullptr; // 2 == sampled textures
	WGPUPipelineLayout current_graphics_pipeline_layout = nullptr;
};

GpuQueue* gpuCreateQueue(WGPUAdapter adapter, WGPUDevice device, WGPULimits limits, CpuAllocatorFunc allocator = default_::cpu_allocator);
inline GpuQueue* gpuCreateQueue(GpuWebGPUDefault def, CpuAllocatorFunc allocator = default_::cpu_allocator) {
	return gpuCreateQueue(def.adapter, def.device, def.limits, allocator);
}

struct GpuCommandBuffer {
	GpuQueue* queue;
	WGPUCommandEncoder encoder;
	// WGPURenderPassEncoder render_pass;

	std::vector<std::function<void()>> code_pending_submission_finished;
};

struct GpuTexture {
	GpuTextureDesc descriptor;
	std::optional<GpuQueue::MonotextureRange> range = {};
	WGPUTexture texture;
};

struct GpuTextureDescriptorImpl {
	TEXTURE type = TEXTURE_2D; ///< Dimensionality and view type.
	uint32_t width = 1, height = 1; ///< Dimensions in texels.
	uint32_t baseMip = 0;
	uint32_t mipCount = 1;
	GpuQueue::MonotextureRange range;
};
static_assert(sizeof(GpuTextureDescriptorImpl) == sizeof(GpuTextureDescriptor), "GPU Texture Descriptors of The Wrong Size");

struct GpuPipeline {
	struct ComputeCache {
		std::string IR;
		WGPUComputePipeline pipeline;
	};

	struct RenderCache {
		WGPURenderPipeline pipeline;
	};

	WGPUPipelineLayout reference_layout; // The layout that the currently cached variant of this pipeline is built against
	std::variant<ComputeCache, RenderCache> cache;
};