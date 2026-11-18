#pragma once

// ---------------------------------------------------------------------------
// Ablations Extension
// ---------------------------------------------------------------------------

#include "compute.h"

// Only in a build configured with NOAPI_ENABLE_ABLATIONS, which defines NOAPI_ABLATIONS; without it the backends compile no ablation code at all.
#ifdef NOAPI_ABLATIONS

NOAPI_EXTERN_C_BEGIN

/**
 * ABLATION_FLAGS_EXT – Design decisions gpuSetAblationsEXT can swap back for the conventional approach each one replaced, so a benchmark can measure what the decision is worth.
 *
 * Every ablation renders the same results as the design it replaces, only slower; none is meant for shipping code. A flag a backend has nothing to swap for does nothing there.
 */
typedef enum ABLATION_FLAGS_EXT {
	ABLATION_NONE = 0x0, ///< Every design decision as shipped.
	/**
	 * ABLATION_ADDRESS_ADD – Shaders add every byte offset to the full 64 bit pointer, carry included, instead of keeping the offset apart from the base it was added to (see GpuAddress in the `noapi` module). Both backends.
	 */
	ABLATION_ADDRESS_ADD = 0x1,
	/**
	 * ABLATION_ROOT_DATA_BUFFERS – Each draw and dispatch creates a uniform buffer and a group 0 bind group of its own for its root data, instead of writing a slot of a shared chunk it binds at a dynamic offset. WebGPU only.
	 */
	ABLATION_ROOT_DATA_BUFFERS = 0x2,
	/**
	 * ABLATION_SAMPLER_LOOKUP – Shaders find a sampler's slot by reading a lookup buffer bound in group 0, instead of comparing against descriptions the pipeline bakes in as override constants. WebGPU only.
	 */
	ABLATION_SAMPLER_LOOKUP = 0x4,
} ABLATION_FLAGS_EXT;

/**
 * gpuSetAblationsEXT – Choose which design decisions queues created afterwards swap for the conventional approach (see ABLATION_FLAGS_EXT).
 *
 * The flags are process wide and read once, by gpuCreateQueue: a queue keeps the ablations it was created with, so changing them never affects a queue that already exists.
 *
 * @code
 * gpuSetAblationsEXT((ABLATION_FLAGS_EXT)(ABLATION_ADDRESS_ADD | ABLATION_SAMPLER_LOOKUP));
 * assert(gpuGetAblationsEXT() == (ABLATION_FLAGS_EXT)(ABLATION_ADDRESS_ADD | ABLATION_SAMPLER_LOOKUP));
 * gpuSetAblationsEXT(ABLATION_NONE);
 * @endcode
 *
 * @param ablations The ablations to apply, ABLATION_NONE for none.
 */
void gpuSetAblationsEXT(ABLATION_FLAGS_EXT ablations);

/**
 * gpuGetAblationsEXT – The ablations gpuSetAblationsEXT last chose, which the next queue created gets.
 *
 * @code
 * assert(gpuGetAblationsEXT() == ABLATION_NONE);
 * @endcode
 */
ABLATION_FLAGS_EXT gpuGetAblationsEXT(void);

/**
 * gpuGetQueueAblationsEXT – The ablations `queue` was created with.
 *
 * @code
 * assert(gpuGetQueueAblationsEXT(queue) == ABLATION_NONE);
 * @endcode
 *
 * @param queue Queue to ask about.
 */
ABLATION_FLAGS_EXT gpuGetQueueAblationsEXT(const GpuQueue* queue);

NOAPI_EXTERN_C_END

#endif // NOAPI_ABLATIONS
