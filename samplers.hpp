#pragma once

// ---------------------------------------------------------------------------
// Sampler Extension
// ---------------------------------------------------------------------------

#include "graphics.hpp"
#include <cstdint>
#include <vector>

/**
 * ADDRESS_MODE – Texture coordinate wrapping mode applied when sampling
 * outside the [0, 1] UV range.
 */
enum ADDRESS_MODE {
	ADDRESS_MODE_CLAMP, ///< Clamp to the edge texel.
	ADDRESS_MODE_MIRROR_REPEAT, ///< Mirror the texture at every integer boundary.
	ADDRESS_MODE_REPEAT, ///< Wrap/tile the texture.
};

/**
 * FILTER – Texel filtering mode used for magnification, minification, and
 * mip-level selection.
 */
enum FILTER {
	FILTER_NEAREST, ///< Point sampling (no interpolation).
	FILTER_LINEAR ///< Bilinear/trilinear interpolation.
};

/**
 * GpuSamplerDesc – Description of a texture sampler's addressing and filtering
 * behavior.
 *
 * Unlike Vulkan/DX12, samplers are not created as separate API objects. Instead
 * a small, fixed set of samplers is declared up front via
 * gpuSetEnabledSamplersEXT, and shaders select between them using the packed
 * 16-bit value returned by pack(). This avoids per-material sampler object
 * management while still covering the full addressing/filtering combination
 * space.
 */
struct GpuSamplerDesc {
	ADDRESS_MODE address_mode_u = ADDRESS_MODE_REPEAT; ///< Wrap mode along the U (horizontal) axis.
	ADDRESS_MODE address_mode_v = ADDRESS_MODE_REPEAT; ///< Wrap mode along the V (vertical) axis.
	ADDRESS_MODE address_mode_w = ADDRESS_MODE_REPEAT; ///< Wrap mode along the W (depth/3D) axis.
	FILTER mag_filter = FILTER_LINEAR; ///< Filter used when magnifying (zoomed in).
	FILTER min_filter = FILTER_LINEAR; ///< Filter used when minifying (zoomed out).
	FILTER mip_filter = FILTER_LINEAR; ///< Filter used when interpolating between mip levels.

	/**
	 * pack – Encode this sampler description into a compact 16-bit value.
	 *
	 * Used as a stable, hashable key for matching against the set of samplers
	 * enabled via gpuSetEnabledSamplersEXT.
	 */
	constexpr uint16_t pack() const noexcept {
		return (static_cast<uint16_t>(address_mode_u)) 
			| (static_cast<uint16_t>(address_mode_v) << 2) 
			| (static_cast<uint16_t>(address_mode_w) << 4) 
			| (static_cast<uint16_t>(mag_filter) << 6) 
			| (static_cast<uint16_t>(min_filter) << 7) 
			| (static_cast<uint16_t>(mip_filter) << 8);
	}

	/**
	 * max_packed – Largest value pack() can produce, useful for sizing lookup tables
	 * indexed by it (which need max_packed() + 1 entries).
	 *
	 * Derived from pack()'s bit layout — three 2 bit address modes followed by three 1 bit
	 * filters — rather than from the largest enumerators that happen to exist today, so
	 * that adding an addressing or filtering mode cannot silently undersize a table.
	 */
	constexpr static uint16_t max_packed() {
		return 0b1'1'1'11'11'11;
	}
};

/**
 * std::hash<GpuSamplerDesc> – Hashes a GpuSamplerDesc via its packed
 * representation, allowing it to be used as a key in unordered associative
 * containers.
 */
template<>
struct std::hash<GpuSamplerDesc> {
	/**
	 * operator() – Computes the hash of \p desc from its packed representation.
	 *
	 * @param desc Sampler description to hash.
	 */
	size_t operator()(const GpuSamplerDesc& desc) const noexcept {
		return std::hash<uint16_t>{}(desc.pack());
	}
};

/**
 * operator== – Compares two GpuSamplerDesc values by their packed representation.
 *
 * @param a First sampler description to compare.
 * @param b Second sampler description to compare.
 */
constexpr bool operator==(const GpuSamplerDesc& a, const GpuSamplerDesc& b) noexcept {
	return a.pack() == b.pack();
}

// The bit widths pack() (and thus max_packed()) assumes. A new mode past these has to widen
// the packing before it can be stored.
static_assert(ADDRESS_MODE_REPEAT <= 0b11, "pack() gives each address mode two bits");
static_assert(FILTER_LINEAR <= 0b1, "pack() gives each filter one bit");

/**
 * GpuSamplerDescListHash – Hashes the list of samplers enabled by gpuSetEnabledSamplersEXT so
 * that it can be used as a cache key.
 *
 * Order matters: a description's position in the list is the sampler slot it lands in, so two
 * orderings of the same descriptions are different sets. Lives here, named, rather than as a
 * std::hash specialization in each backend's header — those would be two different definitions
 * of one symbol, which is an ODR violation the moment both backends end up in the same binary.
 */
struct GpuSamplerDescListHash {
	size_t operator()(const std::vector<GpuSamplerDesc>& descs) const noexcept {
		size_t out = descs.size();
		for(auto& desc: descs)
			out = out * 31 + desc.pack();
		return out;
	}
};

/**
 * gpuSetEnabledSamplersEXT – Declare the fixed set of samplers available to
 * shaders for subsequent draw/dispatch commands in this command buffer.
 *
 * Shaders select a sampler by its packed GpuSamplerDesc value rather than
 * binding a separate sampler object, avoiding per-material sampler management.
 *
 * @param cmd Command buffer to record into.
 * @param enabled_samplers List of sampler descriptions to enable.
 */
void gpuSetEnabledSamplersEXT(GpuCommandBuffer* cmd, std::span<GpuSamplerDesc> enabled_samplers);
