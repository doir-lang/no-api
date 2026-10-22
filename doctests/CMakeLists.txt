# Runs the @code examples in the headers as doctest test cases; see tools/doctest.py.
find_package(Python3 REQUIRED COMPONENTS Interpreter)

find_package(doctest QUIET)
if(NOT doctest_FOUND AND NOT TARGET doctest::doctest)
	message(STATUS "doctest not found via find_package; fetching from GitHub")
	include(FetchContent)
	FetchContent_Declare(doctest
		GIT_REPOSITORY https://github.com/doctest/doctest
		GIT_TAG v2.4.12
	)
	set(DOCTEST_NO_INSTALL ON CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(doctest)
endif()

if(NOAPI_DEFAULT_BACKEND STREQUAL "vulkan")
	set(NOAPI_DOCTEST_BACKEND vulkan)
else()
	set(NOAPI_DOCTEST_BACKEND webgpu)
endif()

set(NOAPI_DOCTEST_HEADERS
	allocator.h compat.h compute.h graphics.h samplers.h shaders.h surface.h sync.h
	vulkan/noapi.h vulkan/noapi.hpp webgpu/noapi.h webgpu/noapi.hpp
)
list(TRANSFORM NOAPI_DOCTEST_HEADERS PREPEND "${PROJECT_SOURCE_DIR}/")

set(NOAPI_DOCTEST_GENERATED
	${CMAKE_CURRENT_BINARY_DIR}/generated/examples_${NOAPI_DOCTEST_BACKEND}.cpp
	${CMAKE_CURRENT_BINARY_DIR}/generated/examples_${NOAPI_DOCTEST_BACKEND}.c
)
add_custom_command(
	OUTPUT ${NOAPI_DOCTEST_GENERATED}
	COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/tools/doctest.py --generate ${CMAKE_CURRENT_BINARY_DIR}/generated
	DEPENDS ${PROJECT_SOURCE_DIR}/tools/doctest.py ${NOAPI_DOCTEST_HEADERS}
	COMMENT "Extracting the header examples"
)

add_executable(noapi-doctests harness.cpp tests.cpp ${NOAPI_DOCTEST_GENERATED})
target_include_directories(noapi-doctests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(noapi-doctests PRIVATE noapi::noapi doctest::doctest)
# Fault injection: on Linux the linker redirects the backends' calls into Slang and Dawn through wrappers in tests.cpp, which can make them fail on demand
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	set(NOAPI_DOCTEST_WRAPPED slang_createGlobalSession2)
	if(NOAPI_DOCTEST_BACKEND STREQUAL "webgpu")
		list(APPEND NOAPI_DOCTEST_WRAPPED wgpuInstanceRequestAdapter wgpuAdapterRequestDevice wgpuInstanceProcessEvents
			wgpuSurfaceGetCapabilities wgpuSurfaceGetCurrentTexture wgpuSurfacePresent)
	endif()
	list(TRANSFORM NOAPI_DOCTEST_WRAPPED PREPEND "LINKER:--wrap=")
	target_link_options(noapi-doctests PRIVATE ${NOAPI_DOCTEST_WRAPPED})
	target_compile_definitions(noapi-doctests PRIVATE NOAPI_DOCTEST_FAULTS)
endif()

# WebGPU has no headless surface, so where the examples brought GLFW along the harness presents to a hidden window instead (see harness.cpp)
if(TARGET glfw3webgpu)
	target_link_libraries(noapi-doctests PRIVATE glfw3webgpu)
	target_include_directories(noapi-doctests PRIVATE ${PROJECT_SOURCE_DIR}/examples)
	target_compile_definitions(noapi-doctests PRIVATE NOAPI_DOCTEST_WINDOW)
endif()
set_target_properties(noapi-doctests PROPERTIES CXX_STANDARD 26 C_STANDARD 11 C_STANDARD_REQUIRED ON)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
	set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/generated/examples_${NOAPI_DOCTEST_BACKEND}.c PROPERTIES COMPILE_OPTIONS -fexceptions)
endif()
