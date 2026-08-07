option(WEBGPU_BUILD_FROM_SOURCE "Tell webgpu distribution that we want to build dawn ourselves" ON)
if(NOT EMSCRIPTEN)
	set(WEBGPU_BACKEND "DAWN" CACHE STRING "Set default")
endif()
set(DAWN_VERSION "7991" CACHE STRING "\
	Version of the Dawn release to use. Must correspond to the number after 'chromium/' \
	in the tag name of an existing release on DAWN_MIRROR. \
	Warning: The webgpu.hpp file provided in include/ may not be compatible with other \
	versions than the default.")

include(FetchContent)
FetchContent_Declare(webgpu-distribution
	GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
	GIT_TAG v0.3.0-gamma
)
FetchContent_MakeAvailable(webgpu-distribution)

add_library(noapi-webgpu noapi.cpp)
target_link_libraries(noapi-webgpu PUBLIC webgpu)