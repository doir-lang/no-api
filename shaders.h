#pragma once

// ---------------------------------------------------------------------------
// Shader Compilation
// ---------------------------------------------------------------------------

/**
 * Both backends take the same shader source: a string of Slang
 * (https://shader-slang.org) handed to the pipeline creation functions as the IR blob.
 * Each backend compiles it internally — to SPIR-V on Vulkan, to WGSL on WebGPU — so one
 * shader runs on both.
 *
 * What makes that possible is that neither backend's shader ABI is written in its own
 * shading language any more. Both are Slang modules named `noapi_backend`, and both are
 * hidden behind one portable module, `noapi`, whose source is byte for byte identical on
 * the two backends. A shader says `import noapi;` and gets the same surface either way:
 *
 * - GpuPtr<T>, a pointer into gpu memory, typed by what it points at. It indexes, steps and
 * compares the way a C pointer does — `p[i]`, `p[i] = v`, `*p`, `p + 1`, `++p`, `q - p`,
 * `p == q` — but nothing is ever dereferenced underneath. On Vulkan the address it holds is a
 * real 64 bit device address; on WebGPU there are no pointers at all, so it is the tagged
 * {offset, monobuffer} pair gpuEncodeWebGPUAddressEXT produces and a load is a switch over the
 * monobuffers. The shader sees neither of those.
 *
 * A backend supplies exactly two memory functions: a load and a store of one 32 bit word at an
 * address. Everything else — indexing, arithmetic, vectors, matrices, pointers to pointers —
 * is generic code in the portable module on top of those two. A shader's own struct joins in
 * by conforming to IGpuLoadable (how wide one is, and how to read one; GpuReader walks the
 * fields so no offset is written down twice) and, if anything writes one, IGpuStorable.
 *
 * - The root data the dispatch or draw was given: gpuComputeData<T>() in a compute shader,
 * gpuVertexData<T>() / gpuFragmentData<T>() / gpuIndexData<T>() in a graphics one. Each hands
 * back a GpuPtr<T>, since what is at the other end is the shader's own business.
 *
 * - gpuGetSamplerIndex, which turns a GpuSamplerDesc into the sampler slot holding it,
 * and gpuSample, which samples the texture at a heap index through that slot.
 *
 * The backend module is not fixed on WebGPU: the monobuffer switch, the per texture
 * sample wrappers and the sampler switch all depend on how many of each the queue
 * currently has, so it is regenerated (still as Slang) and the shader recompiled whenever
 * that changes. This is why the pipeline creation functions keep hold of the source they
 * were given rather than a compiled blob.
 *
 * Entry points are found by stage, not by name: the source handed to a pipeline must
 * contain exactly one entry point of the stage being compiled (marked the Slang way, with
 * [shader("compute")], [shader("vertex")] or [shader("fragment")]). Passing the same
 * source as both the vertex and the fragment IR is therefore fine, and is how a single
 * module supplies both halves of a graphics pipeline.
 *
 * The functions below let a program extend what a shader may `import`. They configure the
 * compiler globally rather than per queue, they may be called before any queue exists, and
 * a change to any of them invalidates whatever the compiler had cached, so pipelines
 * created afterwards see it.
 */

#include "compat.h"

NOAPI_EXTERN_C_BEGIN

/**
 * GpuShaderDiagnosticCallbackEXT – Hook the compiler reports warnings and errors through.
 *
 * @param message The compiler's diagnostic text, not null terminated.
 * @param userdata The pointer handed to gpuSetShaderDiagnosticCallbackEXT alongside it.
 */
typedef void (*GpuShaderDiagnosticCallbackEXT)(GpuStringView message, void* userdata);

/**
 * gpuSetShaderDiagnosticCallbackEXT – Route compiler diagnostics somewhere.
 *
 * When none is set, diagnostics are printed to stderr. A failed compile always reports
 * before the pipeline creation function returns null.
 *
 * @param callback Where to report, or NULL to go back to printing.
 * @param userdata Passed back to \p callback untouched.
 */
void gpuSetShaderDiagnosticCallbackEXT(GpuShaderDiagnosticCallbackEXT callback, void* userdata);

/**
 * gpuAddShaderSearchPathEXT – Add a directory the compiler looks in to resolve an
 * `import`ed module it does not already know.
 *
 * Paths are searched in the order they were added, after the modules registered by
 * gpuAddShaderModuleEXT have been consulted. There is no filesystem under Emscripten, so
 * the web build should register sources directly instead.
 *
 * @param path Directory to search. Copied, so it need not outlive the call.
 */
void gpuAddShaderSearchPathEXT(const char* path);

/**
 * gpuAddShaderModuleEXT – Register a module by name, from source held in memory, so that
 * shaders can `import` it without it existing as a file.
 *
 * Registering a name that is already registered replaces it. The names `noapi` and
 * `noapi_backend` belong to the API and cannot be replaced.
 *
 * @param name Module name, as an importing shader spells it.
 * @param source The module's Slang source. Copied, so it need not outlive the call.
 */
void gpuAddShaderModuleEXT(const char* name, GpuStringView source);

/**
 * gpuRemoveShaderModuleEXT – Forget a module registered by gpuAddShaderModuleEXT.
 *
 * @param name Module name to remove.
 * @return True if a module by that name was registered.
 */
bool gpuRemoveShaderModuleEXT(const char* name);

/**
 * gpuAddShaderPreprocessorDefineEXT – Predefine a preprocessor macro for every shader
 * compiled afterwards, including the modules registered above.
 *
 * @param name Macro name.
 * @param value Macro replacement text, or NULL for an empty one.
 */
void gpuAddShaderPreprocessorDefineEXT(const char* name, const char* value);

/**
 * gpuResetShaderConfigurationEXT – Drop every registered module, search path and macro,
 * putting the compiler back to how it started.
 */
void gpuResetShaderConfigurationEXT(void);

NOAPI_EXTERN_C_END
