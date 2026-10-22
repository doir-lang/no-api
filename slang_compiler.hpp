#pragma once

// ---------------------------------------------------------------------------
// Shader Compilation – shared implementation
// ---------------------------------------------------------------------------

// The backend independent half of shaders.h. A header because each backend library is linked alone and needs its own copy; backends supply `noapi_backend`.

#ifdef __cplusplus

#include "shaders.h"

#include <slang.h>
#include <slang-com-ptr.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace GPU::shaders {

// Which entry point a compile looks for; compute.h's STAGE names pipeline stages for barriers instead.
enum class SHADER_STAGE { COMPUTE, VERTEX, FRAGMENT, MESH };

// A mesh shader counts as graphics, so it gets the draw's root pointers.
constexpr inline bool is_graphics(SHADER_STAGE stage) { return stage != SHADER_STAGE::COMPUTE; }

// ---------------------------------------------------------------------------
// The portable module
// ---------------------------------------------------------------------------

// Shaders' `noapi` module, identical on both backends: everything is built on noapi_backend's one word load and store. GPU_STORES and GPU_TEXTURE_STORES gate writes.
constexpr static const char* PORTABLE_MODULE = R"slang(
module noapi;
import noapi_backend;

// ---------------------------------------------------------------------------
// Addresses
// ---------------------------------------------------------------------------

/**
 * GpuAddress – Somewhere in gpu memory: the 64 bits memory stores a pointer as, low word
 * first.
 *
 * What those bits mean is the backend's business — a device address on Vulkan, a tagged
 * {offset, monobuffer} pair on WebGPU — and this module only ever does two things with
 * them: hand them to the backend's load and store, and add a byte count, which every
 * encoding has to leave as ordinary 64 bit arithmetic.
 *
 * A shader works in GpuPtr<T> instead and only meets this when it describes a layout of
 * its own, which is what IGpuLoadable's two methods are handed.
 */
public struct GpuAddress {
	/// The address's raw 64 bits, low word first.
	public uint2 bits;

	public __init(uint2 raw) { bits = raw; }

	/// Steps forward by `bytes`, or back for a negative count.
	public GpuAddress offset(int bytes) {
		let low = bits.x + uint(bytes);
		// The high word takes the sign extension of `bytes` plus the carry out of the low word,
		// which is a full 64 bit add either way. Vulkan needs that: a device address really is
		// a 64 bit number. On WebGPU the low word is an offset inside one monobuffer, which
		// cannot reach 4GiB, so nothing ever carries into the tag the high word holds.
		return GpuAddress(uint2(low, bits.y + (bytes < 0 ? 0xFFFFFFFFu : 0u) + (low < bits.x ? 1u : 0u)));
	}

	/// The one load every backend provides: the 32 bit word at this address.
	public uint loadU32() { return gpuBackendLoadU32(bits); }

#if GPU_STORES
	/// The one store every backend provides.
	public void storeU32(uint value) { gpuBackendStoreU32(bits, value); }
#endif
}

// ---------------------------------------------------------------------------
// Pointers
// ---------------------------------------------------------------------------

/**
 * GpuPtr – A pointer into gpu memory, typed by what it points at: GpuPtr<float>,
 * GpuPtr<Vertex>, GpuPtr<GpuPtr<Vertex> >. This is what gpuMalloc handed back and what a
 * dispatch or draw passes as its root data.
 *
 * It behaves like a C pointer. `p[i]` reads the ith element and `p[i] = v` writes it, `*p`
 * reads the one it points at, arithmetic counts elements rather than bytes — `p + 1`,
 * `p += 2`, `++p`, `q - p` — and `==`, `!=` and the orderings compare addresses. `p.cast<U>()`
 * is the cast between pointer types, and `p.byteOffset(n)` the step the element type cannot
 * describe.
 *
 * Underneath there is no dereferencing at all: every access is one 4 byte aligned 32 bit
 * word through the backend's load or store, because that is all WebGPU's emulation can
 * address, and T's IGpuLoadable conformance says how many of those an element is made of. So
 * pointing at a wider or richer type costs a backend nothing.
 *
 * The one place the C likeness stops is writing through `*p`: an operator can only hand back
 * a value, never a place to assign to, so `*p = v` does not compile and the write is spelled
 * `p[0] = v`.
 */
public struct GpuPtr<T : IGpuLoadable> {
	/// Where it points. A GpuPtr is this and the layout T, and nothing else.
	public GpuAddress address;

	public __init(GpuAddress at) { address = at; }

	/// The address of the `index`th element.
	public GpuAddress addressOf(int index) { return address.offset(index * int(T.gpuStride)); }

	/// Reinterprets as a pointer to `U`, the way a cast between pointer types does in C.
	public GpuPtr<U> cast<U : IGpuLoadable>() { return GpuPtr<U>(address); }

	/// Steps by raw bytes rather than by elements, for a layout the element type does not
	/// describe -- a header in front of an array, say.
	public GpuPtr<T> byteOffset(int bytes) { return GpuPtr<T>(address.offset(bytes)); }

	/// Whether this is the null pointer, which is what gpuIndexData gives a non indexed draw.
	public bool isNull() { return all(address.bits == uint2(0u, 0u)); }
}

// Indexing is two extensions rather than one member, because what `p[i]` can do depends on T:
// everything can be read, and a T that is storable -- and a stage that has stores at all -- can
// be assigned to as well. Slang picks whichever of the two has the accessors the use needs.

public extension<T : IGpuLoadable> GpuPtr<T> {
	/// The element at `index`, read through T's layout.
	public __subscript(int index) -> T {
		get { return T.gpuLoad(addressOf(index)); }
	}
}

#if GPU_STORES
public extension<T : IGpuStorable> GpuPtr<T> {
	/// The element at `index`, read or written through T's layout.
	public __subscript(int index) -> T {
		get { return T.gpuLoad(addressOf(index)); }
		[nonmutating] set { newValue.gpuStore(addressOf(index)); }
	}
}
#endif

// Slang resolves an operator by looking its name up in the enclosing scope rather than on the
// operand's type, so an operator cannot be a member (error E30073) and these are free
// functions. `operator<` and its friends are written with a space before the generic
// parameters, or the lexer takes `<<` for a shift.

/// Reads what `p` points at. Assignment has to go through `p[0]` instead.
public __prefix T operator*<T : IGpuLoadable>(GpuPtr<T> p) { return p[0]; }

public GpuPtr<T> operator+<T : IGpuLoadable>(GpuPtr<T> p, int elements) { return GpuPtr<T>(p.addressOf(elements)); }
public GpuPtr<T> operator+<T : IGpuLoadable>(int elements, GpuPtr<T> p) { return GpuPtr<T>(p.addressOf(elements)); }
public GpuPtr<T> operator-<T : IGpuLoadable>(GpuPtr<T> p, int elements) { return GpuPtr<T>(p.addressOf(-elements)); }

/// How many elements apart two pointers into the same allocation are.
public int operator-<T : IGpuLoadable>(GpuPtr<T> a, GpuPtr<T> b) {
	return int(a.address.bits.x - b.address.bits.x) / int(T.gpuStride);
}

public void operator+=<T : IGpuLoadable>(inout GpuPtr<T> p, int elements) { p = p + elements; }
public void operator-=<T : IGpuLoadable>(inout GpuPtr<T> p, int elements) { p = p - elements; }

public __prefix GpuPtr<T> operator++<T : IGpuLoadable>(inout GpuPtr<T> p) { p = p + 1; return p; }
public __postfix GpuPtr<T> operator++<T : IGpuLoadable>(inout GpuPtr<T> p) { let was = p; p = p + 1; return was; }
public __prefix GpuPtr<T> operator--<T : IGpuLoadable>(inout GpuPtr<T> p) { p = p - 1; return p; }
public __postfix GpuPtr<T> operator--<T : IGpuLoadable>(inout GpuPtr<T> p) { let was = p; p = p - 1; return was; }

public bool operator==<T : IGpuLoadable>(GpuPtr<T> a, GpuPtr<T> b) { return all(a.address.bits == b.address.bits); }
public bool operator!=<T : IGpuLoadable>(GpuPtr<T> a, GpuPtr<T> b) { return any(a.address.bits != b.address.bits); }

public bool operator< <T : IGpuLoadable>(GpuPtr<T> a, GpuPtr<T> b) {
	return a.address.bits.y == b.address.bits.y
		? a.address.bits.x < b.address.bits.x
		: a.address.bits.y < b.address.bits.y;
}
public bool operator> <T : IGpuLoadable>(GpuPtr<T> a, GpuPtr<T> b) { return b < a; }
public bool operator<= <T : IGpuLoadable>(GpuPtr<T> a, GpuPtr<T> b) { return !(b < a); }
public bool operator>= <T : IGpuLoadable>(GpuPtr<T> a, GpuPtr<T> b) { return !(a < b); }

// ---------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------

/**
 * IGpuLoadable – What a type has to say about itself before a GpuPtr can point at it: how wide
 * one is in gpu memory, and how to read one through an address. IGpuStorable adds writing it.
 *
 * Everything the API offers conforms — uint, int and float, any vector or matrix of them, and
 * GpuPtr itself, so a pointer to a pointer works — and a shader conforms its own structs the
 * same way, after which they index like anything else. A conformance is written against
 * GpuAddress and the other conformances, never against a backend primitive, which is what keeps
 * it portable; GpuReader and GpuWriter spare it the field offsets:
 *
 *     struct Vertex : IGpuLoadable, IGpuStorable {
 *         float2 position;
 *         float3 color;
 *
 *         static const uint gpuStride = 20;
 *
 *         static Vertex gpuLoad(GpuAddress at) {
 *             var from = GpuReader(at);
 *             Vertex out;
 *             out.position = from.next<float2>();
 *             out.color = from.next<float3>();
 *             return out;
 *         }
 *
 *         void gpuStore(GpuAddress at) {
 *             var to = GpuWriter(at);
 *             to.next(position);
 *             to.next(color);
 *         }
 *     }
 *
 * A type only has to be storable if something writes one: leaving IGpuStorable off is what a
 * read only layout looks like, and it is what a layout a WebGPU graphics stage sees has to look
 * like, since there are no stores to build on there (GPU_STORES is 0 and IGpuStorable does not
 * exist). A struct written for both a compute and a graphics stage therefore guards its store
 * half with `#if GPU_STORES`.
 */
public interface IGpuLoadable {
	/// How many bytes one occupies, which is also how far apart they sit in an array. Nothing
	/// here inserts padding, so a conformance has to account for whatever padding the host
	/// side's layout has.
	static const uint gpuStride;

	/// Reads one out of the memory at `at`.
	static This gpuLoad(GpuAddress at);
}

#if GPU_STORES
/// IGpuStorable – A layout that can also be written back, which is what makes `p[i] = v` legal.
public interface IGpuStorable : IGpuLoadable {
	/// Writes this one into the memory at `at`.
	void gpuStore(GpuAddress at);
}
#endif

// The three scalars everything else is made of. Each is one word, reinterpreted rather than
// converted, so the bits arrive exactly as the host wrote them.

public extension uint : IGpuLoadable {
	public static const uint gpuStride = 4;
	public static uint gpuLoad(GpuAddress at) { return at.loadU32(); }
}

public extension int : IGpuLoadable {
	public static const uint gpuStride = 4;
	public static int gpuLoad(GpuAddress at) { return asint(at.loadU32()); }
}

public extension float : IGpuLoadable {
	public static const uint gpuStride = 4;
	public static float gpuLoad(GpuAddress at) { return asfloat(at.loadU32()); }
}

// A pointer is its raw 64 bits, low word first -- the same eight bytes a gpu* occupies in host
// memory, which is what makes a root data struct full of pointers load field by field.
public extension<T : IGpuLoadable> GpuPtr<T> : IGpuLoadable {
	public static const uint gpuStride = 8;
	public static GpuPtr<T> gpuLoad(GpuAddress at) {
		return GpuPtr<T>(GpuAddress(uint2(at.loadU32(), at.offset(4).loadU32())));
	}
}

// A vector is its elements back to back, and a matrix is its rows back to back: a float4x4 is
// the sixteen floats at the address in row major order, so a column major matrix on the host
// side arrives transposed.

public extension<T : IGpuLoadable, let N : int> vector<T, N> : IGpuLoadable {
	public static const uint gpuStride = T.gpuStride * uint(N);

	public static vector<T, N> gpuLoad(GpuAddress at) {
		vector<T, N> out;
		[ForceUnroll] for(int i = 0; i < N; ++i) out[i] = T.gpuLoad(at.offset(i * int(T.gpuStride)));
		return out;
	}
}

public extension<T : IGpuLoadable, let R : int, let C : int> matrix<T, R, C> : IGpuLoadable {
	public static const uint gpuStride = vector<T, C>.gpuStride * uint(R);

	public static matrix<T, R, C> gpuLoad(GpuAddress at) {
		matrix<T, R, C> out;
		[ForceUnroll] for(int r = 0; r < R; ++r)
			out[r] = vector<T, C>.gpuLoad(at.offset(r * int(vector<T, C>.gpuStride)));
		return out;
	}
}

#if GPU_STORES
// The write half of each of those, which only exists where the stage has stores at all

public extension uint : IGpuStorable {
	public void gpuStore(GpuAddress at) { at.storeU32(this); }
}

public extension int : IGpuStorable {
	public void gpuStore(GpuAddress at) { at.storeU32(asuint(this)); }
}

public extension float : IGpuStorable {
	public void gpuStore(GpuAddress at) { at.storeU32(asuint(this)); }
}

public extension<T : IGpuLoadable> GpuPtr<T> : IGpuStorable {
	public void gpuStore(GpuAddress at) {
		at.storeU32(address.bits.x);
		at.offset(4).storeU32(address.bits.y);
	}
}

public extension<T : IGpuStorable, let N : int> vector<T, N> : IGpuStorable {
	public void gpuStore(GpuAddress at) {
		[ForceUnroll] for(int i = 0; i < N; ++i) this[i].gpuStore(at.offset(i * int(T.gpuStride)));
	}
}

public extension<T : IGpuStorable, let R : int, let C : int> matrix<T, R, C> : IGpuStorable {
	public void gpuStore(GpuAddress at) {
		[ForceUnroll] for(int r = 0; r < R; ++r)
			this[r].gpuStore(at.offset(r * int(vector<T, C>.gpuStride)));
	}
}
#endif

/// GpuReader – Walks a run of values in order, so that a layout does not have to write down
/// each field's offset. See IGpuLoadable for what that looks like.
public struct GpuReader {
	public GpuAddress at;

	public __init(GpuAddress from) { at = from; }

	/// Reads the next `T` and steps past it.
	[mutating] public T next<T : IGpuLoadable>() {
		let value = T.gpuLoad(at);
		at = at.offset(int(T.gpuStride));
		return value;
	}

	/// Steps past `bytes` without reading them, for padding the host side's layout has.
	[mutating] public void skip(int bytes) { at = at.offset(bytes); }
}

#if GPU_STORES
/// GpuWriter – GpuReader's other half: writes a run of values in order.
public struct GpuWriter {
	public GpuAddress at;

	public __init(GpuAddress to) { at = to; }

	/// Writes `value` and steps past it.
	[mutating] public void next<T : IGpuStorable>(T value) {
		value.gpuStore(at);
		at = at.offset(int(T.gpuStride));
	}

	/// Steps past `bytes` without writing them, leaving whatever was there.
	[mutating] public void skip(int bytes) { at = at.offset(bytes); }
}
#endif

// ---------------------------------------------------------------------------
// Root data
// ---------------------------------------------------------------------------

// What the root data points at is the shader's business, so each of these is asked for the
// type: gpuComputeData<Uniforms>(), or gpuVertexData<GpuPtr<Vertex> >() for root data that is
// itself just a pointer.

#if GPU_COMPUTE
/// The root data pointer gpuDispatch was given.
public GpuPtr<T> gpuComputeData<T : IGpuLoadable>() { return GpuPtr<T>(GpuAddress(gpuBackendRootCompute())); }
#else
/// The root data pointer the draw gave the vertex stage.
public GpuPtr<T> gpuVertexData<T : IGpuLoadable>() { return GpuPtr<T>(GpuAddress(gpuBackendRootVertex())); }
/**
 * The root data pointer gpuDrawMeshlets gave the mesh stage.
 *
 * The same slot as gpuVertexData -- a mesh shader replaces the vertex stage rather than
 * joining it, so there is only ever one of the two in a pipeline and they share the pointer.
 * This name exists so a mesh shader does not have to read as though it had a vertex stage.
 */
public GpuPtr<T> gpuMeshletData<T : IGpuLoadable>() { return GpuPtr<T>(GpuAddress(gpuBackendRootVertex())); }
/// The root data pointer the draw gave the pixel stage.
public GpuPtr<T> gpuFragmentData<T : IGpuLoadable>() { return GpuPtr<T>(GpuAddress(gpuBackendRootFragment())); }
/// The index buffer the draw was given, or a null pointer for a non indexed draw.
public GpuPtr<T> gpuIndexData<T : IGpuLoadable>() { return GpuPtr<T>(GpuAddress(gpuBackendRootIndex())); }
#endif

// ---------------------------------------------------------------------------
// Samplers
// ---------------------------------------------------------------------------

public static const uint ADDRESS_MODE_CLAMP = 0;
public static const uint ADDRESS_MODE_MIRROR_REPEAT = 1;
public static const uint ADDRESS_MODE_REPEAT = 2;

public static const uint FILTER_NEAREST = 0;
public static const uint FILTER_LINEAR = 1;

/**
 * GpuSamplerDesc – The shader side of the GpuSamplerDesc in samplers.h. Packs the same
 * way, so the value gpuPackSamplerDesc produces is the one gpuSamplerDescPackEXT produces
 * for the same description on the host.
 */
public struct GpuSamplerDesc {
	public uint address_mode_u;
	public uint address_mode_v;
	public uint address_mode_w;
	public uint mag_filter;
	public uint min_filter;
	public uint mip_filter;
}

public uint gpuPackSamplerDesc(GpuSamplerDesc d) {
	return (d.address_mode_u)
		| (d.address_mode_v << 2)
		| (d.address_mode_w << 4)
		| (d.mag_filter << 6)
		| (d.min_filter << 7)
		| (d.mip_filter << 8);
}

public GpuSamplerDesc gpuDefaultSampler() {
	return GpuSamplerDesc(ADDRESS_MODE_REPEAT, ADDRESS_MODE_REPEAT, ADDRESS_MODE_REPEAT,
		FILTER_LINEAR, FILTER_LINEAR, FILTER_LINEAR);
}

/**
 * gpuGetSamplerIndex – The slot holding `desc`, or slot 0 (the default sampler) when that
 * description was never handed to gpuSetEnabledSamplersEXT.
 */
public uint gpuGetSamplerIndex(GpuSamplerDesc desc) { return gpuBackendSamplerSlot(gpuPackSamplerDesc(desc)); }

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

/**
 * gpuSample – Sample the texture at `heap_index` in the active texture heap (see
 * gpuSetActiveTextureHeapPtr) through the sampler in `slot` (see gpuGetSamplerIndex).
 *
 * `uv` runs 0..1 over the texture itself, and `layer` and `mip` are relative to the view
 * the heap entry describes. On WebGPU several textures share one atlas, so the
 * coordinates are scaled onto this texture's share of it; the shader never sees that.
 */
public float4 gpuSample(uint heap_index, uint slot, float3 uv, uint layer, float mip) {
	return gpuBackendSample(heap_index, slot, uv, layer, mip);
}

/// gpuSample for the common case: a 2D texture, mip 0, layer 0.
public float4 gpuSample2D(uint heap_index, uint slot, float2 uv) {
	return gpuBackendSample(heap_index, slot, float3(uv, 0.0), 0u, 0.0);
}

/**
 * gpuStoreTexture – Write one texel of the texture at `heap_index` in the active texture
 * heap, which has to be a read/write entry (gpuRWTextureViewDescriptor of a texture created
 * with USAGE_STORAGE) rather than the sampled kind gpuSample takes.
 *
 * `texel` is a texel coordinate in the mip the entry describes, not a normalized one, and
 * `layer` is relative to that entry the way gpuSample's is. A coordinate outside the texture
 * is dropped, not wrapped, whatever addressing mode anything else is sampling it with: a
 * sampler is not involved in a store.
 *
 * Write only, with no matching load, because the two are not equally portable: writing an
 * image whose format the shader never named costs a Vulkan feature every device offers
 * (shaderStorageImageWriteWithoutFormat), and reading one costs a second feature that real
 * devices decline. A shader that needs its own output back reads it through a sampled entry
 * after a barrier.
 */
#if GPU_TEXTURE_STORES
public void gpuStoreTexture(uint heap_index, uint3 texel, uint layer, float4 value) {
	gpuBackendStoreTexture(heap_index, texel, layer, value);
}

/// gpuStoreTexture for the common case: a 2D texture, layer 0.
public void gpuStoreTexture2D(uint heap_index, uint2 texel, float4 value) {
	gpuBackendStoreTexture(heap_index, uint3(texel, 0u), 0u, value);
}
#endif
)slang";

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

// `generation` counts changes, so a session built against an older one is discarded and registrations reach later pipelines.
struct Registry {
	std::mutex mutex;
	std::vector<std::string> search_paths;
	std::vector<std::pair<std::string, std::string>> modules; // Ordered: registration order is load order
	std::vector<std::pair<std::string, std::string>> defines;
	GpuShaderDiagnosticCallbackEXT diagnostic_callback = nullptr;
	void* diagnostic_userdata = nullptr;
	uint64_t generation = 1;
};

inline Registry& registry() {
	static Registry instance;
	return instance;
}

// The two module names the API owns, which a user registration may not shadow
inline bool is_reserved_module(std::string_view name) {
	return name == "noapi" || name == "noapi_backend";
}

inline void report_diagnostic(std::string_view message) {
	if(message.empty()) return;

	GpuShaderDiagnosticCallbackEXT callback;
	void* userdata;
	{
		auto& reg = registry();
		std::lock_guard lock(reg.mutex);
		callback = reg.diagnostic_callback;
		userdata = reg.diagnostic_userdata;
	}

	if(callback) callback(GpuStringView{message.data(), message.size()}, userdata);
	else fprintf(stderr, "[noapi shader] %.*s\n", (int)message.size(), message.data());
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

// Holds the parsed core module, the expensive part of starting Slang. Not thread safe, so every use is serialized on compile_mutex.
inline slang::IGlobalSession* global_session() {
	static Slang::ComPtr<slang::IGlobalSession> instance = [] {
		Slang::ComPtr<slang::IGlobalSession> session;
		if(SLANG_FAILED(slang::createGlobalSession(session.writeRef())))
			report_diagnostic("failed to create the Slang global session");
		return session;
	}();
	return instance.get();
}

inline std::mutex& compile_mutex() {
	static std::mutex instance;
	return instance;
}

// A session keyed by everything it bakes in, so WebGPU's recompiles on layout changes don't reload every module.
struct CachedSession {
	Slang::ComPtr<slang::ISession> session;
	SlangCompileTarget target = SLANG_TARGET_UNKNOWN;
	std::string backend_module;
	std::string macro_key;
	uint64_t generation = 0;
	uint64_t next_user_module = 0;

	// Without this, every rebuild (on WebGPU possibly each frame) would rerun the compiler and leave another module in the session forever.
	std::unordered_map<std::string, std::string> code;
};

// Several sessions, since compute and graphics compiles differ in their defines and a single one would reparse the API's modules on every switch.
constexpr size_t SESSION_CACHE_CAPACITY = 8;

// Least recently used first
inline std::vector<CachedSession>& session_cache() {
	static std::vector<CachedSession> instance;
	return instance;
}

// Drops whatever the compiler cached. Called by every gpu*SlangEXT mutation.
inline void invalidate_sessions() {
	std::lock_guard lock(compile_mutex());
	session_cache().clear();
}

// Callers hold compile_mutex(). The result stays valid until the next call.
inline CachedSession* get_session(SlangCompileTarget target, std::string_view backend_module,
	const std::vector<std::pair<std::string, std::string>>& extra_defines, std::string& out_error
) {
	auto& reg = registry();

	// Everything that has to match for the cached session to still be the right one
	std::string macro_key;
	for(auto& [name, value]: extra_defines)
		macro_key += name + '=' + value + ';';

	uint64_t generation;
	{
		std::lock_guard lock(reg.mutex);
		generation = reg.generation;
	}

	auto& cache = session_cache();
	// A registry change clears the cache, but one can land between taking the generation above and here.
	std::erase_if(cache, [&](const CachedSession& cached) { return cached.generation != generation; });
	auto found = std::find_if(cache.begin(), cache.end(), [&](const CachedSession& cached) {
		return cached.target == target && cached.backend_module == backend_module && cached.macro_key == macro_key;
	});
	if(found != cache.end()) {
		std::rotate(found, found + 1, cache.end());
		return &cache.back();
	}

	// Either way the compiler can't run: Slang never started (reported once, by global_session), or won't open a session
	auto unavailable = [&out_error]() -> CachedSession* {
		out_error = "the Slang compiler is unavailable";
		return nullptr;
	};
	auto* global = global_session();
	if(!global) return unavailable();

	// Snapshot the registry so the session can be built without holding its lock
	std::vector<std::string> search_paths;
	std::vector<std::pair<std::string, std::string>> modules;
	std::vector<std::pair<std::string, std::string>> defines;
	{
		std::lock_guard lock(reg.mutex);
		search_paths = reg.search_paths;
		modules = reg.modules;
		defines = reg.defines;
	}
	defines.insert(defines.end(), extra_defines.begin(), extra_defines.end());

	std::vector<const char*> search_path_pointers;
	search_path_pointers.reserve(search_paths.size());
	for(auto& path: search_paths)
		search_path_pointers.push_back(path.c_str());

	std::vector<slang::PreprocessorMacroDesc> macros;
	macros.reserve(defines.size());
	for(auto& [name, value]: defines)
		macros.push_back({name.c_str(), value.c_str()});

	// Vulkan's clip space Y points the other way, so SPIR-V inverts Y. spvDescriptorHeapEXT emits heap access as SPV_EXT_descriptor_heap, needing no binding map.
	auto descriptor_heap = global->findCapability("spvDescriptorHeapEXT");
	// Declared so a mesh shader on an incapable target is a compile error; shaders without one emit none of it.
	auto mesh_shading = global->findCapability("spvMeshShadingEXT");

	slang::CompilerOptionEntry spirv_options[] = {
		{slang::CompilerOptionName::VulkanInvertY, {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}},
		{slang::CompilerOptionName::Capability, {slang::CompilerOptionValueKind::Int, descriptor_heap, 0, nullptr, nullptr}},
		{slang::CompilerOptionName::Capability, {slang::CompilerOptionValueKind::Int, mesh_shading, 0, nullptr, nullptr}},
	};

	slang::TargetDesc target_desc {
		.format = target,
		// An unset profile makes some paths reject compute shaders; this one covers every stage the API compiles.
		.profile = global->findProfile(target == SLANG_SPIRV ? "spirv_1_5" : "glsl_450"),
		.compilerOptionEntries = target == SLANG_SPIRV ? spirv_options : nullptr,
		.compilerOptionEntryCount = target == SLANG_SPIRV ? (uint32_t)std::size(spirv_options) : 0,
	};

	slang::SessionDesc session_desc {
		.targets = &target_desc,
		.targetCount = 1,
		.searchPaths = search_path_pointers.empty() ? nullptr : search_path_pointers.data(),
		.searchPathCount = (SlangInt)search_path_pointers.size(),
		.preprocessorMacros = macros.empty() ? nullptr : macros.data(),
		.preprocessorMacroCount = (SlangInt)macros.size(),
	};

	Slang::ComPtr<slang::ISession> session;
	if(SLANG_FAILED(global->createSession(session_desc, session.writeRef())) || !session) return unavailable();

	// The API's modules load first: a module compiles as it loads, so a registered one importing noapi needs it already present.
	auto load = [&](const char* name, std::string_view source) -> bool {
		Slang::ComPtr<slang::IBlob> diagnostics;
		auto path = std::string(name) + ".slang";
		auto* module = session->loadModuleFromSourceString(name, path.c_str(),
			std::string(source).c_str(), diagnostics.writeRef());
		if(diagnostics && diagnostics->getBufferSize())
			report_diagnostic({(const char*)diagnostics->getBufferPointer(), diagnostics->getBufferSize()});
		if(!module) {
			out_error = std::string("failed to compile the '") + name + "' module";
			return false;
		}
		return true;
	};

	if(!load("noapi_backend", backend_module)) return nullptr;
	if(!load("noapi", PORTABLE_MODULE)) return nullptr;
	for(auto& [name, source]: modules)
		if(!load(name.c_str(), source)) return nullptr;

	if(cache.size() >= SESSION_CACHE_CAPACITY) cache.erase(cache.begin());
	cache.push_back({
		.session = session,
		.target = target,
		.backend_module = std::string(backend_module),
		.macro_key = std::move(macro_key),
		.generation = generation,
	});
	return &cache.back();
}

// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------

// Indexed by SHADER_STAGE, in its order
inline SlangStage to_slang_stage(SHADER_STAGE stage) {
	constexpr static SlangStage stages[] = {SLANG_STAGE_COMPUTE, SLANG_STAGE_VERTEX, SLANG_STAGE_FRAGMENT, SLANG_STAGE_MESH};
	static_assert(std::size(stages) == (int)SHADER_STAGE::MESH + 1, "One entry per SHADER_STAGE");
	return stages[(int)stage];
}

inline const char* stage_name(SHADER_STAGE stage) {
	constexpr static const char* names[] = {"compute", "vertex", "fragment", "mesh"};
	static_assert(std::size(names) == (int)SHADER_STAGE::MESH + 1, "One entry per SHADER_STAGE");
	return names[(int)stage];
}

inline std::optional<std::string> compile(SlangCompileTarget target, std::string_view backend_module,
	std::string_view source, SHADER_STAGE stage,
	const std::vector<std::pair<std::string, std::string>>& extra_defines, std::string& out_error
) {
	std::lock_guard lock(compile_mutex());

	auto* cached = get_session(target, backend_module, extra_defines, out_error);
	if(!cached) return std::nullopt;
	auto& cache = *cached;
	auto* session = cache.session.get();

	// Keyed by stage and source only; everything else is what the session itself was keyed by.
	auto cache_key = std::to_string((int)stage) + '\n' + std::string(source);
	if(auto found = cache.code.find(cache_key); found != cache.code.end())
		return found->second;

	// A fresh name each time, so reusing the session can never hand back a stale module
	auto module_name = "noapi_shader_" + std::to_string(cache.next_user_module++);

	Slang::ComPtr<slang::IBlob> diagnostics;
	auto* module = session->loadModuleFromSourceString(module_name.c_str(),
		(module_name + ".slang").c_str(), std::string(source).c_str(), diagnostics.writeRef());
	if(diagnostics && diagnostics->getBufferSize())
		report_diagnostic({(const char*)diagnostics->getBufferPointer(), diagnostics->getBufferSize()});
	if(!module) {
		out_error = "the shader failed to compile";
		return std::nullopt;
	}

	// Entry points are found by stage, so one module can supply both halves of a graphics pipeline; two of one stage is an error.
	auto wanted = to_slang_stage(stage);
	Slang::ComPtr<slang::IComponentType> composed; // The module with the one entry point of the wanted stage
	for(SlangInt32 i = 0; i < module->getDefinedEntryPointCount(); ++i) {
		// The stage is only reachable through a composed layout. Neither call fails for an index in range; a null result just skips it.
		Slang::ComPtr<slang::IEntryPoint> candidate;
		Slang::ComPtr<slang::IComponentType> candidate_composed;
		(void)module->getDefinedEntryPoint(i, candidate.writeRef());
		slang::IComponentType* parts[] = {module, candidate.get()};
		if(candidate) (void)session->createCompositeComponentType(parts, 2, candidate_composed.writeRef());
		auto* layout = candidate_composed ? candidate_composed->getLayout(0, diagnostics.writeRef()) : nullptr;
		if(!layout || layout->getEntryPointCount() == 0 || layout->getEntryPointByIndex(0)->getStage() != wanted) continue;

		if(composed) {
			out_error = std::string("the shader defines more than one ") + stage_name(stage) + " entry point";
			return std::nullopt;
		}
		composed = candidate_composed;
	}

	if(!composed) {
		out_error = std::string("the shader defines no ") + stage_name(stage)
			+ " entry point (mark one with [shader(\"" + stage_name(stage) + "\")])";
		return std::nullopt;
	}

	// A caller can't tell a failed link from failed code generation, so they share a path; warnings and errors from both are reported first
	Slang::ComPtr<slang::IComponentType> linked;
	Slang::ComPtr<slang::IBlob> code;
	std::array<Slang::ComPtr<slang::IBlob>, 2> reports;
	auto result = composed->link(linked.writeRef(), reports[0].writeRef());
	if(SLANG_SUCCEEDED(result)) result = linked->getEntryPointCode(0, 0, code.writeRef(), reports[1].writeRef());
	for(auto& report: reports)
		if(report && report->getBufferSize())
			report_diagnostic({(const char*)report->getBufferPointer(), report->getBufferSize()});
	if(SLANG_FAILED(result) || !code) {
		out_error = "failed to link or generate code for the shader";
		return std::nullopt;
	}

	return cache.code.emplace(std::move(cache_key),
		std::string((const char*)code->getBufferPointer(), code->getBufferSize())).first->second;
}

} // namespace GPU::shaders

// ---------------------------------------------------------------------------
// The public EXT surface
// ---------------------------------------------------------------------------

// Not inline: only one translation unit per backend library includes this, so this is the one definition.
extern "C" {

void gpuSetShaderDiagnosticCallbackEXT(GpuShaderDiagnosticCallbackEXT callback, void* userdata) {
	auto& reg = GPU::shaders::registry();
	std::lock_guard lock(reg.mutex);
	reg.diagnostic_callback = callback;
	reg.diagnostic_userdata = userdata;
}

void gpuAddShaderSearchPathEXT(const char* path) {
	if(!path) return;
	{
		auto& reg = GPU::shaders::registry();
		std::lock_guard lock(reg.mutex);
		reg.search_paths.emplace_back(path);
		++reg.generation;
	}
	GPU::shaders::invalidate_sessions();
}

void gpuAddShaderModuleEXT(const char* name, GpuStringView source) {
	if(!name) return;
	if(GPU::shaders::is_reserved_module(name)) {
		GPU::shaders::report_diagnostic(std::string("'") + name + "' is the API's own module and cannot be replaced");
		return;
	}
	{
		auto& reg = GPU::shaders::registry();
		std::lock_guard lock(reg.mutex);
		std::string text(source.ptr ? source.ptr : "", source.ptr ? source.count : 0);

		auto existing = std::find_if(reg.modules.begin(), reg.modules.end(),
			[&](const auto& entry) { return entry.first == name; });
		if(existing != reg.modules.end()) existing->second = std::move(text);
		else reg.modules.emplace_back(name, std::move(text));

		++reg.generation;
	}
	GPU::shaders::invalidate_sessions();
}

bool gpuRemoveShaderModuleEXT(const char* name) {
	if(!name) return false;
	bool removed = false;
	{
		auto& reg = GPU::shaders::registry();
		std::lock_guard lock(reg.mutex);
		auto existing = std::find_if(reg.modules.begin(), reg.modules.end(),
			[&](const auto& entry) { return entry.first == name; });
		if(existing != reg.modules.end()) {
			reg.modules.erase(existing);
			++reg.generation;
			removed = true;
		}
	}
	if(removed) GPU::shaders::invalidate_sessions();
	return removed;
}

void gpuAddShaderPreprocessorDefineEXT(const char* name, const char* value) {
	if(!name) return;
	{
		auto& reg = GPU::shaders::registry();
		std::lock_guard lock(reg.mutex);
		reg.defines.emplace_back(name, value ? value : "");
		++reg.generation;
	}
	GPU::shaders::invalidate_sessions();
}

void gpuResetShaderConfigurationEXT(void) {
	{
		auto& reg = GPU::shaders::registry();
		std::lock_guard lock(reg.mutex);
		reg.search_paths.clear();
		reg.modules.clear();
		reg.defines.clear();
		++reg.generation;
	}
	GPU::shaders::invalidate_sessions();
}

} // extern "C"

#endif // __cplusplus
