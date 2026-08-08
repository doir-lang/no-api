#pragma once

// ---------------------------------------------------------------------------
// Allocator Extension
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdlib>

/**
 * default_ – Namespace holding default implementations for optional callback
 * hooks used by the API.
 */
namespace default_ {
	/**
	 * cpu_allocator – Default CPU-side host allocator hook, implementing
	 * realloc/free-style semantics:
	 *
	 * - If \p p is NULL and \p size > 0: allocates new memory.
	 * - If \p p is not NULL and \p size > 0: reallocates and copies data.
	 * - If \p size is 0: frees memory and returns NULL.
	 * - If \p size equals the current allocation size: may return \p p unchanged.
	 *
	 * @param p Existing allocation to resize or free, or NULL to allocate fresh memory.
	 * @param size Requested size in bytes, or 0 to free \p p.
	 */
	inline static void* cpu_allocator(void* p, size_t size) noexcept {
		if(size == 0) {
			if(p) free(p);
			return NULL;
		}

		return realloc(p, size);
	}
}

/**
 * CpuAllocatorFunc – Function pointer type for a custom CPU-side allocator hook,
 * matching the realloc/free semantics documented on default_::cpu_allocator.
 *
 * @param p Existing allocation to resize or free, or NULL to allocate fresh memory.
 * @param size Requested size in bytes, or 0 to free \p p.
 */
typedef void *(*CpuAllocatorFunc)(void *p, size_t size);
