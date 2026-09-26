#pragma once

// ---------------------------------------------------------------------------
// Allocator Extension
// ---------------------------------------------------------------------------

#include "compat.h"

#ifdef __cplusplus
	#include <cstdlib>
#else
	#include <stdlib.h>
#endif

NOAPI_EXTERN_C_BEGIN

/**
 * CpuAllocatorFunc – Function pointer type for a custom CPU-side allocator hook,
 * matching the realloc/free semantics documented on gpuDefaultCpuAllocator.
 *
 * @param p Existing allocation to resize or free, or NULL to allocate fresh memory.
 * @param size Requested size in bytes, or 0 to free \p p.
 */
typedef void *(*CpuAllocatorFunc)(void *p, size_t size);

/**
 * gpuDefaultCpuAllocator – Default CPU-side host allocator hook, implementing
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
NOAPI_INLINE void* gpuDefaultCpuAllocator(void* p, size_t size) NOAPI_NOEXCEPT {
	if(size == 0) {
		if(p) free(p);
		return NULL;
	}

	return realloc(p, size);
}

NOAPI_EXTERN_C_END

#ifdef __cplusplus
/**
 * default_ – Namespace holding default implementations for optional callback
 * hooks used by the API. Kept as the C++ spelling of the gpuDefault* hooks, which are
 * what a C caller (and the default arguments in these headers) names them by.
 */
namespace default_ {
	inline constexpr CpuAllocatorFunc cpu_allocator = gpuDefaultCpuAllocator;
}
#endif
