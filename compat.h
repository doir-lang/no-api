#pragma once

// ---------------------------------------------------------------------------
// C Compatibility Layer
// ---------------------------------------------------------------------------

/**
 * Every public header in this API is written to parse as both C++ and C99, so that the
 * same (C++ built) library can be called from either language. This header holds the
 * pieces that make that possible: the macros that spell a C++ only piece of syntax in a
 * way C can swallow, and the small vocabulary types that stand in for the standard
 * library ones the API would otherwise take by value.
 *
 * The rules the rest of the headers follow:
 *
 * - Functions meant to be called from C are declared inside a
 * NOAPI_EXTERN_C_BEGIN/END pair, so they get C language linkage and can be linked
 * against from a C translation unit. Only one function of a given name may have C
 * linkage, so overload sets are given one canonical C entry point each; the remaining
 * overloads are C++ only conveniences that forward to it.
 *
 * - Templates, and anything else that has no C spelling at all (std::hash
 * specializations, the std::expected returning setup helpers, the backends' internal
 * object definitions), live inside #ifdef __cplusplus.
 *
 * - Default arguments and default member initializers are written NOAPI_DEFAULT(value),
 * which disappears in C. A C caller therefore has to pass every argument; the
 * descriptor structs come with GPU_*_DEFAULT initializer macros so that their
 * documented defaults are still reachable.
 *
 * - std::span, std::optional and std::string_view parameters are replaced by the
 * GpuXSpan / GpuOptionalX / GpuStringView structs below. In C they are plain structs;
 * in C++ they convert implicitly to and from the standard type they stand for and carry
 * enough of its interface (size(), data(), begin()/end(), operator*, ...) to be used
 * like it.
 *
 * - A parameter that is a const reference in C++ is written NOAPI_CONST_REF(T), which is
 * a const T* in C. This is the one place the two faces of a declaration are not the
 * same type: it relies on a reference being passed as a pointer, which both the Itanium
 * C++ ABI (every Unix-like target) and MSVC guarantee.
 */

#ifdef __cplusplus
	#include <cstddef>
	#include <cstdint>
	#include <iterator>
	#include <optional>
	#include <span>
	#include <string_view>
	#include <type_traits>

	#define NOAPI_EXTERN_C_BEGIN extern "C" {
	#define NOAPI_EXTERN_C_END }

	/**
	 * NOAPI_DEFAULT – A default argument or default member initializer. Present in C++,
	 * gone in C (where the value is documented by the matching GPU_*_DEFAULT macro instead).
	 */
	#define NOAPI_DEFAULT(...) = __VA_ARGS__

	/**
	 * NOAPI_CONST_REF – A read only by-reference parameter: const T& in C++, const T* in C.
	 */
	#define NOAPI_CONST_REF(TYPE) const TYPE&

	#define NOAPI_INLINE inline
	#define NOAPI_CONSTEXPR constexpr
	#define NOAPI_NOEXCEPT noexcept
#else
	#include <stdbool.h>
	#include <stddef.h>
	#include <stdint.h>

	#define NOAPI_EXTERN_C_BEGIN
	#define NOAPI_EXTERN_C_END

	#define NOAPI_DEFAULT(...)
	#define NOAPI_CONST_REF(TYPE) const TYPE*

	#define NOAPI_INLINE static inline
	#define NOAPI_CONSTEXPR static inline
	#define NOAPI_NOEXCEPT
#endif

// ---------------------------------------------------------------------------
// Spans
// ---------------------------------------------------------------------------

/**
 * NOAPI_SPAN_TYPE – Declares a {pointer, count} view over a contiguous, read only run of
 * ELEMENT (which is always const qualified, and for which `ELEMENT*` is the pointer the
 * struct stores).
 *
 * The two members are named ptr and count rather than data and size so that the C++ face
 * can also offer the data()/size() accessors std::span is used through; C code reads the
 * members directly.
 *
 * In C++ the type converts implicitly from std::span, from any contiguous range (a
 * std::vector or std::array of the element type), and from a {pointer, count} pair, and
 * back to std::span, so it can be passed and consumed exactly like the std::span it
 * replaced.
 */
#ifdef __cplusplus
	#define NOAPI_SPAN_TYPE(NAME, ELEMENT) \
		struct NAME { \
			ELEMENT* ptr; \
			size_t count; \
			\
			NAME() noexcept : ptr(nullptr), count(0) {} \
			NAME(ELEMENT* ptr, size_t count) noexcept : ptr(ptr), count(count) {} \
			NAME(std::span<ELEMENT> span) noexcept : ptr(span.data()), count(span.size()) {} \
			NAME(std::span<std::remove_const_t<ELEMENT>> span) noexcept : ptr(span.data()), count(span.size()) {} \
			/* Any contiguous range of the element type: a std::vector, a std::array, a C array */ \
			template<typename Range> \
				requires requires(Range&& range) { \
					static_cast<ELEMENT*>(std::data(range)); \
					static_cast<size_t>(std::size(range)); \
				} \
			NAME(Range&& range) noexcept : ptr(std::data(range)), count(std::size(range)) {} \
			\
			operator std::span<ELEMENT>() const noexcept { return {ptr, count}; } \
			\
			ELEMENT* data() const noexcept { return ptr; } \
			size_t size() const noexcept { return count; } \
			size_t size_bytes() const noexcept { return count * sizeof(ELEMENT); } \
			bool empty() const noexcept { return count == 0; } \
			ELEMENT* begin() const noexcept { return ptr; } \
			ELEMENT* end() const noexcept { return ptr + count; } \
			ELEMENT& operator[](size_t index) const noexcept { return ptr[index]; } \
		}
#else
	#define NOAPI_SPAN_TYPE(NAME, ELEMENT) \
		typedef struct NAME { \
			ELEMENT* ptr; \
			size_t count; \
		} NAME
#endif

/**
 * GpuByteSpan – A read only view of raw bytes, used for the shader IR blobs handed to the
 * pipeline creation functions. Stands in for std::span<const std::byte>.
 */
#ifdef __cplusplus
	struct GpuByteSpan {
		const void* ptr;
		size_t count; ///< Size of the run in bytes.

		GpuByteSpan() noexcept : ptr(nullptr), count(0) {}
		GpuByteSpan(const void* ptr, size_t count) noexcept : ptr(ptr), count(count) {}
		GpuByteSpan(std::span<const std::byte> span) noexcept : ptr(span.data()), count(span.size()) {}
		GpuByteSpan(std::span<std::byte> span) noexcept : ptr(span.data()), count(span.size()) {}

		operator std::span<const std::byte>() const noexcept { return {data(), count}; }

		const std::byte* data() const noexcept { return static_cast<const std::byte*>(ptr); }
		size_t size() const noexcept { return count; }
		size_t size_bytes() const noexcept { return count; }
		bool empty() const noexcept { return count == 0; }
		const std::byte* begin() const noexcept { return data(); }
		const std::byte* end() const noexcept { return data() + count; }
	};
#else
	typedef struct GpuByteSpan {
		const void* ptr;
		size_t count;
	} GpuByteSpan;
#endif

/**
 * GpuStringView – A borrowed, not necessarily null terminated run of characters, used for
 * the messages handed to an error callback. Stands in for std::string_view.
 */
#ifdef __cplusplus
	struct GpuStringView {
		const char* ptr;
		size_t count; ///< Length of the text in characters, excluding any null terminator.

		GpuStringView() noexcept : ptr(nullptr), count(0) {}
		GpuStringView(const char* ptr, size_t count) noexcept : ptr(ptr), count(count) {}
		GpuStringView(std::string_view view) noexcept : ptr(view.data()), count(view.size()) {}

		operator std::string_view() const noexcept { return {ptr, count}; }

		const char* data() const noexcept { return ptr; }
		size_t size() const noexcept { return count; }
		bool empty() const noexcept { return count == 0; }
	};
#else
	typedef struct GpuStringView {
		const char* ptr;
		size_t count;
	} GpuStringView;
#endif

/**
 * GpuCStringSpan – A read only view of an array of null terminated strings, as the
 * platform layers want their extension and layer lists.
 */
NOAPI_SPAN_TYPE(GpuCStringSpan, const char* const);

// ---------------------------------------------------------------------------
// Optionals
// ---------------------------------------------------------------------------

/**
 * NOAPI_OPTIONAL_TYPE – Declares a {flag, value} struct standing in for
 * std::optional<TYPE>.
 *
 * The flag is a plain member named has_value (so C code can read and write it), which is
 * why the C++ face spells the test as `if(optional)` rather than `optional.has_value()`.
 * Everything else std::optional is used through — operator*, operator->, value_or, and
 * implicit conversion to and from std::optional itself — is provided, along with an
 * implicit conversion from a bare value and from std::nullopt.
 */
#ifdef __cplusplus
	#define NOAPI_OPTIONAL_TYPE(NAME, TYPE) \
		struct NAME { \
			bool has_value; \
			TYPE value; \
			\
			NAME() noexcept : has_value(false), value() {} \
			NAME(std::nullopt_t) noexcept : has_value(false), value() {} \
			NAME(const TYPE& value) noexcept : has_value(true), value(value) {} \
			/* The {flag, value} shape the GPU_*_DEFAULT initializer macros are written in */ \
			NAME(bool has_value, const TYPE& value) noexcept : has_value(has_value), value(value) {} \
			NAME(const std::optional<TYPE>& optional) noexcept : has_value(optional.has_value()), value(optional.value_or(TYPE{})) {} \
			\
			operator std::optional<TYPE>() const noexcept { \
				if(has_value) return value; \
				return std::nullopt; \
			} \
			\
			explicit operator bool() const noexcept { return has_value; } \
			const TYPE& operator*() const noexcept { return value; } \
			TYPE& operator*() noexcept { return value; } \
			const TYPE* operator->() const noexcept { return &value; } \
			TYPE* operator->() noexcept { return &value; } \
			TYPE value_or(const TYPE& fallback) const noexcept { return has_value ? value : fallback; } \
		}
#else
	#define NOAPI_OPTIONAL_TYPE(NAME, TYPE) \
		typedef struct NAME { \
			bool has_value; \
			TYPE value; \
		} NAME
#endif
