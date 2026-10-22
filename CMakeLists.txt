cmake_minimum_required(VERSION 3.21)
project(NoAPI CXX C)

# gpu* pointers carry a 3 bit tag above a 61 bit offset that wasm32 truncates, so everything linked in must be wasm64.
if(EMSCRIPTEN)
	add_compile_options(-m64)
	add_link_options(-m64)
	# Slang's embedded core module outgrows emscripten's 16MB default heap, and its recursion overflows the 64KB default stack ("table index is out of bounds").
	add_link_options(-sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=512MB -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=16MB)
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

# Per target, not CMAKE_INTERPROCEDURAL_OPTIMIZATION, which vendored Slang, Dawn and vk-bootstrap would inherit; optimized configurations only, as LTO links are slow.
option(NOAPI_ENABLE_LTO "Build the optimized configurations with link time optimization" ${PROJECT_IS_TOP_LEVEL})

if(NOAPI_ENABLE_LTO)
	include(CheckIPOSupported)
	check_ipo_supported(RESULT NOAPI_LTO_SUPPORTED OUTPUT NOAPI_LTO_UNSUPPORTED_REASON)
	if(NOT NOAPI_LTO_SUPPORTED)
		message(STATUS "LTO was asked for but this toolchain does not support it: ${NOAPI_LTO_UNSUPPORTED_REASON}")
	endif()
endif()

# noapi_enable_lto(<target>...)
# Marks each target for LTO in the optimized configurations, when LTO was asked for and is supported.
function(noapi_enable_lto)
	if(NOT NOAPI_LTO_SUPPORTED)
		return()
	endif()
	foreach(target ${ARGN})
		# Per configuration properties, since a multi config generator has no single build type at configure time
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

# Both backends compile Slang source at runtime (see shaders.h). Declared here so they can link it; Slang itself is configured last, below.
add_library(noapi-slang INTERFACE)

if(NOAPI_ENABLE_VULKAN)
	add_subdirectory(vulkan)
endif()

if(NOAPI_ENABLE_WEBGPU)
	add_subdirectory(webgpu)
endif()

# Instruments the backends for tools/coverage.py. Unoptimized so that line counts match the source; everything else keeps its build type.
option(NOAPI_ENABLE_COVERAGE "Instrument the backends for gcov line coverage" OFF)
if(NOAPI_ENABLE_COVERAGE)
	foreach(backend noapi-vulkan noapi-webgpu)
		if(TARGET ${backend})
			# Nothing in the backends throws, and without unwinding paths gcov stops reporting cleanup-only blocks as unexecuted lines.
			target_compile_options(${backend} PRIVATE --coverage -O0 -fno-exceptions)
			target_link_options(${backend} PUBLIC --coverage)
		endif()
	endforeach()
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
	add_subdirectory(examples)
endif()

# The examples in the headers' documentation, run headlessly by tools/doctest.py. Emscripten has no headless device to run them on.
if(EMSCRIPTEN)
	option(NOAPI_BUILD_DOCTESTS "Build the header documentation examples" OFF)
else()
	option(NOAPI_BUILD_DOCTESTS "Build the header documentation examples" ${PROJECT_IS_TOP_LEVEL})
endif()
if(NOAPI_BUILD_DOCTESTS)
	add_subdirectory(doctests)
endif()

# ---------------------------------------------------------------------------
# Slang
# ---------------------------------------------------------------------------

# Configured last: Slang reuses existing Vulkan-Headers and SPIRV-Headers targets, while Dawn and vk-bootstrap each need their own newer copies.

# Emscripten builds are meant to point NOAPI_SLANG_ROOT at a slang-wasm release rather than build the submodule for wasm.
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
		# Slang's POST_BUILD debug info split runs the host objcopy and strip, which fail on Emscripten's wasm outputs; nothing here uses the .dwarf sidecars.
		set(SLANG_ENABLE_SPLIT_DEBUG_INFO OFF CACHE BOOL "" FORCE)
		# Only needed for Slang's C++ / CPU targets, which this project never asks for
		set(SLANG_SLANG_LLVM_FLAVOR DISABLE CACHE STRING "" FORCE)
		# slang-glslang hosts spirv-opt, loaded at runtime. Emscripten leaves it off: a runtime loaded module is a side module the page must fetch.
		if(EMSCRIPTEN)
			set(SLANG_ENABLE_SLANG_GLSLANG OFF CACHE BOOL "" FORCE)
			# Emscripten makes a shared library a side module the page fetches at runtime, which nothing deploys beside it
			set(SLANG_LIB_TYPE STATIC CACHE STRING "" FORCE)
		else()
			set(SLANG_ENABLE_SLANG_GLSLANG ON CACHE BOOL "" FORCE)
		endif()
		# Dawn already added SPIRV-Tools and a second copy fails to configure, so Slang links Dawn's as a "system" package with an empty config file
		if(SLANG_ENABLE_SLANG_GLSLANG AND TARGET SPIRV-Tools-opt)
			set(NOAPI_SPIRV_TOOLS_PACKAGE "${CMAKE_CURRENT_BINARY_DIR}/noapi-spirv-tools")
			file(WRITE "${NOAPI_SPIRV_TOOLS_PACKAGE}/SPIRV-ToolsConfig.cmake" "# SPIRV-Tools targets come from Dawn's copy, already added\n")
			set(SPIRV-Tools_DIR "${NOAPI_SPIRV_TOOLS_PACKAGE}" CACHE PATH "" FORCE)
			set(SLANG_USE_SYSTEM_SPIRV_TOOLS ON CACHE BOOL "" FORCE)
		else()
			set(SLANG_USE_SYSTEM_SPIRV_TOOLS OFF CACHE BOOL "" FORCE)
		endif()
		add_subdirectory(external/slang EXCLUDE_FROM_ALL)
		# Nothing links the wrapper, as Slang loads it at runtime, so without this it is never built
		if(TARGET slang-glslang)
			add_dependencies(noapi-slang slang-glslang)
		endif()
	endif()

	target_link_libraries(noapi-slang INTERFACE slang)
endif()
