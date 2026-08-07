cmake_minimum_required(VERSION 3.21)
project(NoAPI CXX C)

set(CMAKE_CXX_STANDARD 26)

option(NOAPI_ENABLE_VULKAN "Enable the Vulkan backend" ON)
option(NOAPI_ENABLE_WEBGPU "Enable the WebGPU backend" ON)

set(NOAPI_DEFAULT_BACKEND "WebGPU" CACHE STRING "Backend exposed as the noapi::noapi alias")
set_property(CACHE NOAPI_DEFAULT_BACKEND PROPERTY STRINGS "Vulkan" "WebGPU")

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
if(NOAPI_DEFAULT_BACKEND STREQUAL "Vulkan")
	if(NOT NOAPI_ENABLE_VULKAN)
		message(FATAL_ERROR "NOAPI_DEFAULT_BACKEND is set to Vulkan, but NOAPI_ENABLE_VULKAN is OFF")
	endif()
	add_library(noapi::noapi ALIAS noapi-vulkan)
elseif(NOAPI_DEFAULT_BACKEND STREQUAL "WebGPU")
	if(NOT NOAPI_ENABLE_WEBGPU)
		message(FATAL_ERROR "NOAPI_DEFAULT_BACKEND is set to WebGPU, but NOAPI_ENABLE_WEBGPU is OFF")
	endif()
	add_library(noapi::noapi ALIAS noapi-webgpu)
else()
	message(FATAL_ERROR "NOAPI_DEFAULT_BACKEND must be one of: Vulkan, WebGPU (got \"${NOAPI_DEFAULT_BACKEND}\")")
endif()

if(PROJECT_IS_TOP_LEVEL)
	add_subdirectory(test)
endif()