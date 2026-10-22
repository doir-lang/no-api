#pragma once

// ---------------------------------------------------------------------------
// Sampler Extension
// ---------------------------------------------------------------------------

#include "graphics.h"

#ifdef __cplusplus
	#include <cstdint>
	#include <vector>
#endif

/**
 * ADDRESS_MODE – Texture coordinate wrapping mode applied when sampling outside the [0, 1] UV range.
 */
typedef enum ADDRESS_MODE {
	ADDRESS_MODE_CLAMP, ///< Clamp to the edge texel.
	ADDRESS_MODE_MIRROR_REPEAT, ///< Mirror the texture at every integer boundary.
	ADDRESS_MODE_REPEAT, ///< Wrap/tile the texture.
} ADDRESS_MODE;

/**
 * FILTER – Texel filtering mode used for magnification, minification, and mip-level selection.
 */
typedef enum FILTER {
	FILTER_NEAREST, ///< Point sampling (no interpolation).
	FILTER_LINEAR ///< Bilinear/trilinear interpolation.
} FILTER;

/**
 * GpuSamplerDesc – Description of a texture sampler's addressing and filtering behavior.
 *
 * Unlike Vulkan/DX12, samplers are not created as separate API objects. Instead a small, fixed set of samplers is declared up front via gpuSetEnabledSamplersEXT, and shaders select between them using the packed 16-bit value returned by gpuSamplerDescPackEXT. This avoids per-material sampler object management while still covering the full addressing/filtering combination space.
 */
typedef struct GpuSamplerDesc {
	ADDRESS_MODE address_mode_u NOAPI_DEFAULT(ADDRESS_MODE_REPEAT); ///< Wrap mode along the U (horizontal) axis.
	ADDRESS_MODE address_mode_v NOAPI_DEFAULT(ADDRESS_MODE_REPEAT); ///< Wrap mode along the V (vertical) axis.
	ADDRESS_MODE address_mode_w NOAPI_DEFAULT(ADDRESS_MODE_REPEAT); ///< Wrap mode along the W (depth/3D) axis.
	FILTER mag_filter NOAPI_DEFAULT(FILTER_LINEAR); ///< Filter used when magnifying (zoomed in).
	FILTER min_filter NOAPI_DEFAULT(FILTER_LINEAR); ///< Filter used when minifying (zoomed out).
	FILTER mip_filter NOAPI_DEFAULT(FILTER_LINEAR); ///< Filter used when interpolating between mip levels.
} GpuSamplerDesc;

/**
 * GPU_SAMPLER_DESC_DEFAULT – The defaults above as an initializer, for C. Keep in sync with GpuSamplerDesc.
 */
#define GPU_SAMPLER_DESC_DEFAULT { \
	ADDRESS_MODE_REPEAT, ADDRESS_MODE_REPEAT, ADDRESS_MODE_REPEAT, \
	FILTER_LINEAR, FILTER_LINEAR, FILTER_LINEAR \
}

NOAPI_EXTERN_C_BEGIN

/**
 * gpuSamplerDescPackEXT – Encode a sampler description into a compact 16-bit value: a stable, hashable key for matching against the set of samplers enabled via gpuSetEnabledSamplersEXT.
 *
 * @code
 * GpuSamplerDesc linear = GPU_SAMPLER_DESC_DEFAULT, nearest = GPU_SAMPLER_DESC_DEFAULT;
 * nearest.mag_filter = FILTER_NEAREST;
 * assert(gpuSamplerDescPackEXT(linear) != gpuSamplerDescPackEXT(nearest));
 * @endcode
 *
 * @param desc Sampler description to pack.
 */
NOAPI_CONSTEXPR uint16_t gpuSamplerDescPackEXT(GpuSamplerDesc desc) NOAPI_NOEXCEPT {
	return (uint16_t)((uint16_t)(desc.address_mode_u)
		| ((uint16_t)(desc.address_mode_v) << 2)
		| ((uint16_t)(desc.address_mode_w) << 4)
		| ((uint16_t)(desc.mag_filter) << 6)
		| ((uint16_t)(desc.min_filter) << 7)
		| ((uint16_t)(desc.mip_filter) << 8));
}

/**
 * gpuSamplerDescMaxPackedEXT – Largest value gpuSamplerDescPackEXT can produce, useful for sizing lookup tables indexed by it (which need gpuSamplerDescMaxPackedEXT() + 1 entries).
 *
 * Derived from the packing's bit layout — three 2 bit address modes followed by three 1 bit filters — rather than from the largest enumerators that happen to exist today, so that adding an addressing or filtering mode cannot silently undersize a table.
 *
 * @code
 * uint8_t* slots = (uint8_t*)calloc((size_t)gpuSamplerDescMaxPackedEXT() + 1, 1);
 * GpuSamplerDesc desc = GPU_SAMPLER_DESC_DEFAULT;
 * slots[gpuSamplerDescPackEXT(desc)] = 1;
 * free(slots);
 * @endcode
 */
NOAPI_CONSTEXPR uint16_t gpuSamplerDescMaxPackedEXT(void) NOAPI_NOEXCEPT {
	return (uint16_t)0x1ff; // 1'1'1'11'11'11
}

NOAPI_EXTERN_C_END

#ifdef __cplusplus
/**
 * std::hash<GpuSamplerDesc> – Hashes a GpuSamplerDesc via its packed representation, allowing it to be used as a key in unordered associative containers.
 */
template<>
struct std::hash<GpuSamplerDesc> {
	/**
	 * operator() – Computes the hash of \p desc from its packed representation.
	 *
	 * @param desc Sampler description to hash.
	 */
	size_t operator()(const GpuSamplerDesc& desc) const noexcept {
		return std::hash<uint16_t>{}(gpuSamplerDescPackEXT(desc));
	}
};

/**
 * operator== – Compares two GpuSamplerDesc values by their packed representation.
 *
 * @param a First sampler description to compare.
 * @param b Second sampler description to compare.
 */
constexpr bool operator==(const GpuSamplerDesc& a, const GpuSamplerDesc& b) noexcept {
	return gpuSamplerDescPackEXT(a) == gpuSamplerDescPackEXT(b);
}

// The bit widths gpuSamplerDescMaxPackedEXT assumes; a new mode past these has to widen the packing first.
static_assert(ADDRESS_MODE_REPEAT <= 0b11, "the packing gives each address mode two bits");
static_assert(FILTER_LINEAR <= 0b1, "the packing gives each filter one bit");

/**
 * GpuSamplerDescListHash – Hashes the list of samplers enabled by gpuSetEnabledSamplersEXT so that it can be used as a cache key.
 *
 * Order matters: a description's position in the list is the sampler slot it lands in, so two orderings of the same descriptions are different sets. Lives here, named, rather than as a std::hash specialization in each backend's header — those would be two different definitions of one symbol, which is an ODR violation the moment both backends end up in the same binary.
 */
struct GpuSamplerDescListHash {
	size_t operator()(const std::vector<GpuSamplerDesc>& descs) const noexcept {
		size_t out = descs.size();
		for(auto& desc: descs)
			out = out * 31 + gpuSamplerDescPackEXT(desc);
		return out;
	}
};
#endif

/**
 * GpuSamplerDescSpan – A read only list of sampler descriptions: std::span<const GpuSamplerDesc> in C++, a {ptr, count} pair in C.
 */
NOAPI_SPAN_TYPE(GpuSamplerDescSpan, const GpuSamplerDesc);

NOAPI_EXTERN_C_BEGIN

/**
 * gpuSetEnabledSamplersEXT – Declare the fixed set of samplers available to shaders for subsequent draw/dispatch commands in this command buffer.
 *
 * Shaders select a sampler by its description (gpuGetSamplerIndex in the `noapi` module) rather than binding a separate sampler object, avoiding per-material sampler management. A description that was not enabled gets the default sampler, linear and repeating.
 *
 * @code{.cpp}
 * GpuTextureDesc desc = {.dimensions = {64, 64, 1}, .format = FORMAT_RGBA8_UNORM, .usage = TEXTURE_USAGE_FLAGS(USAGE_SAMPLED | USAGE_TRANSFER_DST)};
 * auto memory = (gpu*)gpuMalloc(queue, gpuTextureSizeAlign(queue, desc));
 * auto texture = gpuCreateTexture(queue, desc, memory);
 * auto texels = gpuMalloc<uint32_t>(queue, 64 * 64);
 * for(uint32_t i = 0; i < 64 * 64; ++i) texels[i] = i % 64 < 32 ? 0xff0000ff : 0xff00ff00; // Red on the left, green on the right
 * auto texels_gpu = gpuHostToDevicePointer(queue, texels);
 * auto heap = gpuMalloc<GpuTextureDescriptor>(queue, 1, MEMORY_DESCRIPTOR_HEAP);
 * heap[0] = gpuTextureViewDescriptor(queue, texture, {});
 * auto heap_gpu = gpuHostToDevicePointer(queue, heap);
 *
 * auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(R"slang(
 * 	import noapi;
 * 	[shader("compute")] [numthreads(1, 1, 1)]
 * 	void main() {
 * 		let clamp = GpuSamplerDesc(ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, FILTER_NEAREST, FILTER_NEAREST, FILTER_NEAREST);
 * 		gpuComputeData<float4>()[0] = gpuSample2D(0, gpuGetSamplerIndex(clamp), float2(1.25, 0.5));
 * 	}
 * )slang"));
 * auto color = gpuMalloc<float>(queue, 4, MEMORY_READBACK);
 * auto color_gpu = gpuHostToDevicePointer(queue, color);
 *
 * GpuSamplerDesc clamp = {ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, ADDRESS_MODE_CLAMP, FILTER_NEAREST, FILTER_NEAREST, FILTER_NEAREST};
 * auto cmd = gpuStartCommandRecording(queue);
 * gpuSyncMemoryEXT(cmd, texels_gpu);
 * gpuCopyToTexture(cmd, memory, texels_gpu, texture);
 * gpuBarrier(cmd, STAGE_TRANSFER, STAGE_COMPUTE);
 * gpuSyncMemoryEXT(cmd, heap_gpu);
 * gpuSetActiveTextureHeapPtr(cmd, heap_gpu);
 * gpuSetEnabledSamplersEXT(cmd, {&clamp, 1});
 * gpuSetPipeline(cmd, pipeline);
 * gpuDispatch(cmd, color_gpu, {1, 1, 1});
 * gpuSyncMemoryEXT(cmd, color_gpu);
 * gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), gpuSubmit(queue, {&cmd, 1}));
 * assert(color[0] == 0.0f && color[1] == 1.0f); // Clamped to the green edge, where repeating would have wrapped round to red
 *
 * gpuFreePipeline(queue, pipeline);
 * gpuFree(queue, color);
 * gpuFree(queue, heap);
 * gpuFree(queue, texels);
 * gpuFree(queue, memory);
 * @endcode
 *
 * @note On WebGPU at most gpu_sampler_slot_count - 1 samplers can be enabled at once, since the default sampler takes a slot too.
 *
 * @param cmd Command buffer to record into.
 * @param enabled_samplers List of sampler descriptions to enable.
 */
void gpuSetEnabledSamplersEXT(GpuCommandBuffer* cmd, GpuSamplerDescSpan enabled_samplers);

NOAPI_EXTERN_C_END
