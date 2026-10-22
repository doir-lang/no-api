# Agent guidelines for noapi

## Where documentation lives

- Documentation belongs in the headers: the public headers at the root (`compute.h`, `graphics.h`, `surface.h`, ...) and each backend's `noapi.h` / `noapi.hpp`.
  A header is the one place a reader goes to learn what a definition does and how to call it.
- Implementation files (`vulkan/*.cpp`, `webgpu/*.cpp`) carry no documentation comments.
  Do not repeat a function's header documentation above its definition, and do not add `/** */` blocks to internal helpers.
  The only comments in a `.cpp` file are free-standing ones explaining code that is difficult to follow on its own (see Comments).

## Comments

- Keep comments minimal.
  Write one only where the code does something difficult to follow on its own: a non-obvious invariant, a subtle memory or lifetime rule, a backend or driver quirk being worked around, an API mapping that is not one-to-one (such as Vulkan's reversed packed formats), or why an apparently simpler approach would be wrong.
  Do not restate what the code already says.
- Cap every free-standing comment (a `//` or `/* */` block that is not a documentation comment) at 25 words.
  If it needs more, the code or the header's documentation should carry the explanation instead.
- Editors here soft-wrap, so write each comment, and each paragraph of a documentation comment, on a single line rather than hard-wrapping it at a column.
- Do not leave commented-out code, change logs, or notes about what was changed in a comment; those belong in commit messages.

## Documentation

- Put a Doxygen comment (`/** */`, or `///<` for a trailing enumerator or member) directly above every public definition in a header: each function, struct, enum, typedef, macro and template that a user of the API can name.
  Definitions internal to a backend (inside `GPU::detail` or similar) do not need one.
- Follow the existing form: open with `name – summary`, then any detail, then an example in a `@code` / `@endcode` block, then `@tparam` / `@param` for each parameter and `@note` for caveats such as "C++ only" or a backend difference.
  Every documentation comment on a function includes an example.

  ```c
  /**
   * gpuMalloc – Allocate GPU memory and return a CPU-accessible pointer.
   * For MEMORY_DEFAULT, the returned pointer is persistently CPU-mapped, so the CPU can write to it directly like ordinary heap memory.
   *
   * @code
   * float* values = (float*)gpuMalloc(queue, 4 * sizeof(float), 16, MEMORY_DEFAULT);
   * assert(values);
   * values[0] = 1.0f;
   * gpuFree(queue, values);
   * @endcode
   *
   * @param queue The GPU queue (device) on which the memory will be used.
   * @param bytes Size of the allocation in bytes.
   * @param align Required alignment in bytes (must be a power of two). Default is 16.
   * @param memory Memory heap type (default is MEMORY_DEFAULT).
   */
  void* gpuMalloc(GpuQueue* queue, size_t bytes, size_t align NOAPI_DEFAULT(16), MEMORY memory NOAPI_DEFAULT(MEMORY_DEFAULT));
  ```

- An example is the body of a function that is handed a headless `GpuQueue* queue`, and the backend objects it was created from as `backend` (a `GpuVulkanDefault` or `GpuWebGPUDefault`); it frees whatever it creates, and checks results with `assert`.
  There is no surface, so an example that draws renders into a texture it creates.
- A plain `@code` block must compile as both C and C++; mark one that uses C++ only API (templates, overloads, `string_to_bytes`, `auto`) as `@code{.cpp}`.
  An example in a backend's `noapi.h` / `noapi.hpp` is only compiled against that backend.
- Keep examples short, and make them correct against the current API: they must compile and pass on both backends.
  Check them with `python3 tools/doctest.py build build-vk`, which builds and runs every example in each build directory and reports public definitions missing documentation or an example (`--check` reports without building).
- When the two backends behave differently (for example `gpuSyncMemoryEXT` being a no-op except on WebGPU), say so in the documentation.
- A documentation comment is not a free-standing comment, so the 25-word cap does not apply to it, but keep it as brief as the definition allows.

## C compatibility

- The public headers must parse as both C11 and C++26; follow the rules laid out at the top of `compat.h`.
- Declare C-callable functions inside `NOAPI_EXTERN_C_BEGIN` / `NOAPI_EXTERN_C_END`, give each overload set one canonical C entry point, and put the other overloads, templates and anything else with no C spelling inside `#ifdef __cplusplus`.
- Spell default arguments and default member initializers `NOAPI_DEFAULT(value)`, and give descriptor structs a `GPU_*_DEFAULT` initializer macro so C callers can reach the documented defaults.
- Use the `Gpu*Span`, `GpuOptional*` and `GpuStringView` stand-ins rather than `std::span`, `std::optional` or `std::string_view` in C-visible signatures, and `NOAPI_CONST_REF(T)` for const reference parameters.
- `examples/c_compat.c` and the C half of every doc example check that the headers stay valid C; extend `c_compat.c` when adding C-visible API that no example exercises.

## Style

- The API is plain-data structs and opaque handles operated on by free functions (`gpuMalloc(queue, ...)`, `gpuFree(queue, ptr)`), not methods.
  Use member functions only for operator overloads and what a language protocol requires (such as `begin()` / `end()` or conversion operators on the compat types).
- Name public functions `gpuCamelCase`, with an `EXT` suffix for anything beyond the original No Graphics API design (surfaces, samplers, diagnostics, capability queries).
  Name public types `GpuPascalCase` and enums and their values `UPPER_SNAKE_CASE`.
  Name backend internals, locals and struct members `snake_case`, adding a trailing underscore only when the plain name is a keyword (`default_`).
- Prefer qualified names to `using namespace` or aliasing a name into another scope: write `GPU::detail::format2vulkan(...)`.
- Indent with tabs.
- Keep the two backends in step: a change to the public API is implemented in both `vulkan/` and `webgpu/`, and in the Slang side of `shaders.h` when it affects shaders.

## Failure and invariants

- Every function that can fail says in its documentation how it reports failure (a null pointer, `false`, an error string or `std::expected`, `errno`) and what state it leaves its arguments in; prefer leaving them as they were.
- Never drop the result of a call that can fail, whether a noapi, Vulkan, WebGPU or Slang call: check it (`VK_CHECK` in the Vulkan backend), or discard it explicitly with `(void)` when failure is impossible or harmless there.
  The tests assert on it.
- When a limit on one value follows from another (such as the number of bits in a tagged GPU address), name the constant once, derive the others from it, and `assert` or `static_assert` the limit where the value is accepted.
- Keep example- and test-only code, such as helpers and loaders, in `examples/` or `doctests/` rather than in the library.

## Organization

- Where possible, group related definitions into sections (for example: types, enumerations, queue creation, memory management, textures, pipelines, barriers), under the existing `// ----` banner comments, and order them so a reader meets a type before the functions built on it.
- Whenever possible, define a function before anything that uses it, so a file reads bottom-up from its building blocks.
  When functions use each other in a cycle, define first the one that more of the file uses.
- Keep each backend's `noapi.cpp` in the same order as the declarations in the headers it implements, so the two backends can be read side by side.

## Tests

- The header examples are the unit tests.
  A change to a function's behaviour comes with an example, or a change to its existing one, that shows it.
- `doctests/` holds the harness that runs them: a headless device, and an `assert` that fails one example without stopping the rest.
  Do not edit the generated sources in the build directory; edit the header.
- `python3 tools/coverage.py [-v] [vulkan] [webgpu]` builds an instrumented copy of each backend, runs the examples, and reports per-file line coverage (`-v` lists the uncovered lines).
  Use it to find backend code no example reaches.
- The programs in `examples/` are standalone samples that open a window; they assert on results and should run cleanly on both backends.
  Prefer extending an existing one over adding a new program; add one only when no existing program fits, and list it in the README's Examples table.
- Before finishing, build the WebGPU and Vulkan configurations (`build/` and `build-vk/`, see the README), run `tools/doctest.py build build-vk`, and run the example programs your change touches.

## Bugs

- If you find a bug while working, fix it, with a test that would have caught it, rather than only reporting it.
  Only when the fix would likely take a long time, report it instead, saying what you found and where.
