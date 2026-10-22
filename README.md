# noapi

A small, modern GPU API built on the design Sebastian Aaltonen describes in
["No Graphics API"](https://www.sebastianaaltonen.com/blog/no-graphics-api). It has two backends,
**Vulkan** and **WebGPU**, and the WebGPU one also runs in the browser through Emscripten.

The idea is to drop the decades of state tracking that today's GPUs no longer need:

- **Pointers instead of bindings.** Each draw or dispatch receives one 64-bit GPU pointer to its
  root data, much like a thread function receives a single `void*`. What that pointer points at is
  up to the shader.
- **CUDA-style memory.** `gpuMalloc` returns CPU-mapped GPU memory (UMA or PCIe ReBAR), and
  `gpuHostToDevicePointer` turns a host pointer into an address the GPU can use. You don't create
  buffer objects or bind memory to them.
- **A bindless texture heap.** Textures live in one flat descriptor heap, indexed by 32-bit
  integers. There are no descriptor sets.
- **Stage-only barriers.** A barrier names a producer stage, a consumer stage and a few hazard
  flags. Nothing tracks the state of individual resources.
- **One shader language.** Shaders are written once in [Slang](https://shader-slang.org) and
  compiled at runtime: to SPIR-V on Vulkan, to WGSL on WebGPU.

The public headers parse as both **C11 and C++26**, so one library can be called from either
language.

## A taste

```cpp
// Root data is plain memory: allocate it, fill it in, hand the GPU its address
struct ShaderData { gpu* input; gpu* output; };

auto input  = gpuMalloc<float>(queue, 5, MEMORY_DEFAULT);
auto output = gpuMalloc<float>(queue, 5, MEMORY_READBACK);
auto data   = gpuMalloc<ShaderData>(queue);
data->input  = gpuHostToDevicePointer(queue, input);
data->output = gpuHostToDevicePointer(queue, output);
auto data_gpu = gpuHostToDevicePointer(queue, data);
gpuSyncMemoryEXT(queue, data_gpu);

auto pipeline = gpuCreateComputePipeline(queue, string_to_bytes(slang_source));

auto cmd = gpuStartCommandRecording(queue);
gpuSyncMemoryEXT(cmd, data->input);  // A no-op everywhere except WebGPU
gpuSetPipeline(cmd, pipeline);
gpuDispatch(cmd, data_gpu, {1, 1, 1});
gpuSyncMemoryEXT(cmd, data->output);
auto submission = gpuSubmit(queue, {&cmd, 1});
gpuWaitSemaphore(queue, gpuGetSubmissionSemaphoreEXT(queue), submission);
```

```slang
import noapi;

struct ShaderData : IGpuLoadable {
	GpuPtr<float> input;
	GpuPtr<float> output;

	static const uint gpuStride = 16;
	static ShaderData gpuLoad(GpuAddress at) {
		var from = GpuReader(at);
		ShaderData out;
		out.input = from.next<GpuPtr<float> >();
		out.output = from.next<GpuPtr<float> >();
		return out;
	}
}

[shader("compute")]
[numthreads(16, 1, 1)]
void computeMain(uint3 thread : SV_DispatchThreadID) {
	let data = *gpuComputeData<ShaderData>();
	data.output[thread.x] = data.input[thread.x] * 2.0;
}
```

`GpuPtr<T>` behaves like a C pointer (`p[i]`, `*p`, `p + 1`, `q - p`, ...) on both backends. On
Vulkan it holds a real device address. WebGPU has no pointers, so there it holds a tagged
offset into one of a handful of "monobuffers". The shader looks the same either way. See
[`shaders.h`](shaders.h) for the full shader-side interface.

## Layout

| Path | Contents |
| --- | --- |
| [`compute.h`](compute.h) | Core API shared by compute and graphics: queues, memory, textures, pipelines, barriers, semaphores |
| [`graphics.h`](graphics.h) | Rasterization: render passes, raster/blend/depth-stencil state, indexed, indirect and meshlet draws |
| [`surface.h`](surface.h) | Swapchain extension, a single reconfigurable object in the style of WebGPU |
| [`samplers.h`](samplers.h) | Sampler extension: a small fixed set of samplers, picked in shaders by a packed 16-bit description |
| [`shaders.h`](shaders.h) | Shader compilation: Slang modules, search paths, preprocessor defines, diagnostics |
| [`sync.h`](sync.h) | Synchronization helpers: submission semaphores, wait-idle, memory sync |
| [`allocator.h`](allocator.h) | Hook for a custom CPU allocator |
| [`compat.h`](compat.h) | The macros and span/optional stand-ins that keep the headers valid C |
| [`vulkan/`](vulkan) | Vulkan backend (`noapi.h` C interface, `noapi.hpp` C++ conveniences) |
| [`webgpu/`](webgpu) | WebGPU backend (Dawn natively, the browser under Emscripten) |
| [`examples/`](examples) | Standalone sample programs |
| [`doctests/`](doctests) | Harness that runs the examples in the headers' documentation |
| [`tools/`](tools) | `doctest.py` (check and run the header examples) and `coverage.py` (backend line coverage) |

Functions ending in `EXT` are extensions to the original design, such as surfaces, samplers,
diagnostics and capability queries.

## Requirements

- CMake 3.21+ and a compiler with C++26 support (`<print>`, `<expected>`, `#elifdef`)
- **Vulkan backend:** the Vulkan SDK and a driver that supports `VK_EXT_descriptor_heap`,
  `VK_KHR_shader_untyped_pointers` and `VK_EXT_extended_dynamic_state3`.
  `VK_EXT_mesh_shader` (for meshlet pipelines) and `VK_KHR_swapchain_maintenance1` are used if
  the driver has them.
- **WebGPU backend:** nothing extra. Dawn is fetched and built automatically through
  [WebGPU-distribution](https://github.com/eliemichel/WebGPU-distribution).
- **Web:** [Emscripten](https://emscripten.org). The build targets wasm64 (`-sMEMORY64=1`) because
  GPU pointers are 64-bit tagged addresses.

These are fetched with `FetchContent` when `find_package` can't find them: glslang,
vk-bootstrap, VulkanMemoryAllocator, volk and GLFW (GLFW is only needed by the examples), and doctest for the header examples.

## Building

```sh
git clone --recursive <repo-url> noapi
cd noapi
# If you already cloned without --recursive:
git submodule update --init --recursive
```

**Native, WebGPU (the default):**

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/examples/0_basic
```

**Native, Vulkan:**

```sh
cmake -S . -B build-vk -G Ninja -DCMAKE_BUILD_TYPE=Release \
	-DNOAPI_DEFAULT_BACKEND=vulkan -DNOAPI_ENABLE_WEBGPU=OFF
cmake --build build-vk
```

**Web:**

```sh
emcmake cmake -S . -B build-em -G Ninja -DCMAKE_BUILD_TYPE=Release \
	-DNOAPI_SLANG_ROOT=/path/to/slang-wasm-release
cmake --build build-em
cmake --build build-em --target serve   # serves the examples at http://localhost:8080
```

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `NOAPI_ENABLE_VULKAN` | `ON` (`OFF` under Emscripten) | Build the Vulkan backend |
| `NOAPI_ENABLE_WEBGPU` | `ON` | Build the WebGPU backend |
| `NOAPI_DEFAULT_BACKEND` | `webgpu` if enabled, else `vulkan` | Backend behind the `noapi::noapi` alias |
| `NOAPI_SLANG_ROOT` | *(empty)* | Use a prebuilt Slang release instead of building the submodule |
| `NOAPI_ENABLE_LTO` | `ON` when top level | Link-time optimization for the optimized configurations |
| `NOAPI_BUILD_DOCTESTS` | `ON` when top level (`OFF` under Emscripten) | Build `noapi-doctests`, the header examples |
| `NOAPI_ENABLE_COVERAGE` | `OFF` | Instrument the backends for gcov (used by `tools/coverage.py`) |

Slang is a large dependency. If it isn't installed and `NOAPI_SLANG_ROOT` isn't set, the
`external/slang` submodule is built with everything this project doesn't use turned off. A
[prebuilt release](https://github.com/shader-slang/slang/releases) is faster to build with, and
it also gets `spirv-opt` optimization of the generated SPIR-V, which a submodule build skips.

## Using it in your project

```cmake
add_subdirectory(noapi)
target_link_libraries(my_app PRIVATE noapi::noapi)   # or noapi-vulkan / noapi-webgpu directly
```

The backend you link defines `NOAPI_BACKEND_VULKAN` or `NOAPI_BACKEND_WEBGPU`. Include that
backend's header (`<vulkan/noapi.hpp>` or `<webgpu/noapi.hpp>`, or the `.h` version from C), then
create a queue. `gpuSetupDefaultVulkanEXT` or `gpuSetupDefaultWebGPUEXT` creates the instance,
device and surface for you. If you already have your own, pass them to `gpuCreateQueue`.
[`examples/0_basic.cpp`](examples/0_basic.cpp) is a complete example from start to finish.

## Tests

The examples in the headers' documentation are the unit tests. `tools/doctest.py` extracts every `@code` block, compiles it as C and as C++, and runs it against a headless device, then reports public declarations that are missing documentation or an example:

```sh
python3 tools/doctest.py build build-vk   # build and run the examples in each build directory
python3 tools/doctest.py --check          # only check the documentation
python3 tools/coverage.py -v vulkan       # line coverage of a backend from those examples and the example programs
```

`coverage.py` also runs each program in `examples/` for a fixed number of frames (it sets `NOAPI_EXAMPLE_FRAMES`), which needs a display; pass `--doctests-only` to measure the header examples alone.

## Examples

| Example | Shows |
| --- | --- |
| `0_basic` | Compute round trip, split barriers, surface setup, a triangle; also links a C translation unit to check the headers stay valid C |
| `1_texture` | Loading an image into a texture and sampling it |
| `2_compute_texture` | Generating a texture in a compute shader, then displaying it |
| `3_meshlet` | Mesh shader pipelines with direct and indirect meshlet draws |
| `4_pbr` | Image-based PBR lighting of the Utah teapot, using an HDR environment map |
| `5_msaa_stencil_outline` | 4x MSAA resolved into the surface, a stencil outline around the teapot, and a culled triangle strip floor |
| `6_raytrace` | A compute shader ray tracer |

Asset credits are listed in [`examples/credits.txt`](examples/credits.txt).

## License

[MIT](LICENSE) © Joshua Dahl
