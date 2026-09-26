cmake_minimum_required(VERSION 3.21)
project(NoAPI CXX C)

# A gpu* is a device address carrying a 3 bit monobuffer tag above a 61 bit offset (see
# gpuEncodeWebGPUAddressEXT), and the generated WGSL reads one as a vec2<u32>, so the host pointer it
# is stored in has to be 64 bits wide. wasm32 would silently truncate the tag away and leave every
# root data struct half the size its shader expects, so the web build targets wasm64. Everything
# linked in has to agree, hence the directory wide flags rather than per target ones.
if(EMSCRIPTEN)
	add_compile_options(-sMEMORY64=1)
	add_link_options(-sMEMORY64=1)
endif()

# A generator expression is no use here: option() wants a plain constant
if(EMSCRIPTEN)
	option(NOAPI_ENABLE_VULKAN "Enable the Vulkan backend" OFF)
else()
	option(NOAPI_ENABLE_VULKAN "Enable the Vulkan backend" ON)
endif()
option(NOAPI_ENABLE_WEBGPU "Enable the WebGPU backend" ON)

# Whichever backend is available unless asked otherwise, so that disabling one is enough
if(NOAPI_ENABLE_WEBGPU)
	set(NOAPI_DEFAULT_BACKEND_INIT "webgpu")
else()
	set(NOAPI_DEFAULT_BACKEND_INIT "vulkan")
endif()
set(NOAPI_DEFAULT_BACKEND "${NOAPI_DEFAULT_BACKEND_INIT}" CACHE STRING "Backend exposed as the noapi::noapi alias")
set_property(CACHE NOAPI_DEFAULT_BACKEND PROPERTY STRINGS "vulkan" "webgpu")

if(NOT NOAPI_ENABLE_VULKAN AND NOT NOAPI_ENABLE_WEBGPU)
	message(FATAL_ERROR "At least one of NOAPI_ENABLE_VULKAN or NOAPI_ENABLE_WEBGPU must be enabled")
endif()

if(NOAPI_ENABLE_VULKAN)
	add_subdirectory(vulkan)
endif()

if(NOAPI_ENABLE_WEBGPU)
	add_subdirectory(webgpu)
endif()

# Alias the selected default backend as noapi::noapi
if(NOAPI_DEFAULT_BACKEND STREQUAL "vulkan")
	if(NOT NOAPI_ENABLE_VULKAN)
		message(FATAL_ERROR "NOAPI_DEFAULT_BACKEND is set to vulkan, but NOAPI_ENABLE_VULKAN is OFF")
	endif()
	add_library(noapi::noapi ALIAS noapi-vulkan)
elseif(NOAPI_DEFAULT_BACKEND STREQUAL "webgpu")
	if(NOT NOAPI_ENABLE_WEBGPU)
		message(FATAL_ERROR "NOAPI_DEFAULT_BACKEND is set to webgpu, but NOAPI_ENABLE_WEBGPU is OFF")
	endif()
	add_library(noapi::noapi ALIAS noapi-webgpu)
else()
	message(FATAL_ERROR "NOAPI_DEFAULT_BACKEND must be one of: vulkan, webgpu (got \"${NOAPI_DEFAULT_BACKEND}\")")
endif()

if(PROJECT_IS_TOP_LEVEL)
	add_subdirectory(test)
endif()