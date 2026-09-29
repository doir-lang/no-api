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
	# Slang is a whole compiler front end, and the embedded core module it deserializes on the
	# first gpuCreateShaderModule dwarfs the 16MB fixed heap emscripten defaults to, so the heap
	# has to be allowed to grow or the first shader compile aborts with OOM. The high initial size
	# is only there to keep that compile from walking up to it one grow at a time.
	add_link_options(-sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=512MB -sMAXIMUM_MEMORY=4GB)
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

# ---------------------------------------------------------------------------
# Link time optimization
# ---------------------------------------------------------------------------

# Optimized configurations only: an LTO link is slow enough to be felt on every rebuild, and it
# inlines away the frames a debugger would otherwise have to step through.
#
# Turned on per target rather than through a directory wide CMAKE_INTERPROCEDURAL_OPTIMIZATION,
# because everything vendored in here -- Slang, Dawn, glslang, vk-bootstrap -- arrives through
# add_subdirectory and would inherit the setting. They are an order of magnitude more code than
# this project, so they would dominate the build time, and Slang in particular is not built with
# it upstream.
option(NOAPI_ENABLE_LTO "Build the optimized configurations with link time optimization" ${PROJECT_IS_TOP_LEVEL})

if(NOAPI_ENABLE_LTO)
	include(CheckIPOSupported)
	check_ipo_supported(RESULT NOAPI_LTO_SUPPORTED OUTPUT NOAPI_LTO_UNSUPPORTED_REASON)
	if(NOT NOAPI_LTO_SUPPORTED)
		message(STATUS "LTO was asked for but this toolchain does not support it: ${NOAPI_LTO_UNSUPPORTED_REASON}")
	endif()
endif()

# noapi_enable_lto(<target>...)
#
# Marks each target for LTO in the optimized configurations, if it was both asked for and
# supported. The per configuration properties are what makes this work under a multi config
# generator, where there is no one build type to test against at configure time.
function(noapi_enable_lto)
	if(NOT NOAPI_LTO_SUPPORTED)
		return()
	endif()
	foreach(target ${ARGN})
		set_target_properties(${target} PROPERTIES
			INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
			INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON
			INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON
		)
	endforeach()
endfunction()

# ---------------------------------------------------------------------------
# Slang
# ---------------------------------------------------------------------------

# Shaders are handed to both backends as Slang source and compiled at runtime (see shaders.h),
# so both link the compiler rather than a shading language front end each. The target is
# declared here because the backends link it, but what it resolves to is decided further down:
# Slang has to be configured *after* them.
add_library(noapi-slang INTERFACE)

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

# ---------------------------------------------------------------------------
# Slang
# ---------------------------------------------------------------------------

# Configured last, on purpose. Slang vendors copies of dependencies that Dawn and the Vulkan
# backend also bring -- Vulkan-Headers and SPIRV-Headers -- and CMake has only one global name
# for each of them, so whoever is configured first wins for everybody. Slang guards its copies
# behind `if(NOT TARGET ...)` and so adapts to what is already there, while Dawn and vk-bootstrap
# both need their own: Dawn's vulkan-utility-libraries wants newer Vulkan headers than the system
# has, and vk-bootstrap wants newer ones than Slang ships. Configuring Slang after them is what
# lets all three get the headers they were built against.
#
# An already built Slang is used when there is one, either installed where find_package can
# see it or unpacked from a release and pointed at with NOAPI_SLANG_ROOT. That is also how
# the Emscripten build is meant to be fed, since building the compiler for wasm from here is
# not something this project drives -- point NOAPI_SLANG_ROOT at a slang-wasm release.
# Failing both, the submodule is built, with everything this project has no use for (the
# command line tools, the tests and examples, the LLVM based host compiler) turned off.
set(NOAPI_SLANG_ROOT "" CACHE PATH "Root of a prebuilt Slang release to use instead of building the submodule")

if(NOAPI_SLANG_ROOT)
	find_library(NOAPI_SLANG_LIBRARY NAMES slang REQUIRED
		HINTS "${NOAPI_SLANG_ROOT}/lib" "${NOAPI_SLANG_ROOT}/bin" NO_DEFAULT_PATH)
	target_include_directories(noapi-slang INTERFACE "${NOAPI_SLANG_ROOT}/include")
	target_link_libraries(noapi-slang INTERFACE "${NOAPI_SLANG_LIBRARY}")
	message(STATUS "Using the prebuilt Slang at ${NOAPI_SLANG_ROOT}")
else()
	find_package(slang QUIET)
	if(NOT slang_FOUND AND NOT TARGET slang)
		if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/slang/CMakeLists.txt")
			message(FATAL_ERROR
				"The Slang submodule is empty and no prebuilt Slang was given. Either run\n"
				"  git submodule update --init --recursive\n"
				"or point NOAPI_SLANG_ROOT at an unpacked Slang release.")
		endif()

		message(STATUS "Building Slang from external/slang")
		set(SLANG_ENABLE_GFX OFF CACHE BOOL "" FORCE)
		set(SLANG_ENABLE_SLANGD OFF CACHE BOOL "" FORCE)
		set(SLANG_ENABLE_SLANGC OFF CACHE BOOL "" FORCE)
		set(SLANG_ENABLE_SLANGI OFF CACHE BOOL "" FORCE)
		set(SLANG_ENABLE_SLANGRT OFF CACHE BOOL "" FORCE)
		set(SLANG_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
		set(SLANG_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
		set(SLANG_ENABLE_REPLAYER OFF CACHE BOOL "" FORCE)
		set(SLANG_ENABLE_SLANG_RHI OFF CACHE BOOL "" FORCE)
		set(SLANG_ENABLE_DXIL OFF CACHE BOOL "" FORCE)
		# Slang strips debug info out of its binaries into a sidecar .dwarf with a POST_BUILD
		# ${CMAKE_OBJCOPY} / ${CMAKE_STRIP}. Those are the *host* tools, which a cross build has no
		# business running on its outputs: under Emscripten they are pointed at wasm archives (where
		# the host llvm-objcopy aborts outright) and at slang.js (which is not an object file at all).
		# Nothing here consumes the sidecars anyway, so they are not worth having on any platform.
		set(SLANG_ENABLE_SPLIT_DEBUG_INFO OFF CACHE BOOL "" FORCE)
		# Only needed for Slang's C++ / CPU targets, which this project never asks for
		set(SLANG_SLANG_LLVM_FLAVOR DISABLE CACHE STRING "" FORCE)
		# Slang's slang-glslang wrapper has to be off, and it costs something.
		#
		# It drags in two vendored dependencies that cannot coexist with what is already here.
		# Its glslang defines the same glslang::* targets the Vulkan backend imports from the
		# system for gpuBlitTextureEXT, and its spirv-tools defines the same custom targets
		# (spirv-tools-tables and friends) as the copy Dawn builds. Both are hard configure
		# errors, and ordering does not help the spirv-tools one because neither copy guards its
		# target names.
		#
		# Pointing Slang at the system glslang instead (SLANG_USE_SYSTEM_GLSLANG) does not work
		# either: the wrapper includes "SPIRV/GlslangToSpv.h", which only resolves against the
		# vendored layout, so it fails to compile.
		#
		# What this costs: Slang hosts spirv-opt inside that wrapper, so a submodule build emits
		# SPIR-V straight from the Slang IR optimizer with no spirv-opt pass after it, and logs a
		# recoverable "failed to load downstream compiler 'spirv-opt'" the first time it looks.
		# The output is valid and runs; it is just unoptimized. A prebuilt release ships
		# libslang-glslang.so ready built, so NOAPI_SLANG_ROOT does not have this problem -- which
		# is the reason to prefer it for anything but a source build.
		set(SLANG_ENABLE_SLANG_GLSLANG OFF CACHE BOOL "" FORCE)
		add_subdirectory(external/slang EXCLUDE_FROM_ALL)
	endif()

	target_link_libraries(noapi-slang INTERFACE slang)
endif()
