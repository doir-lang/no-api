#pragma once

// ---------------------------------------------------------------------------
// Shader Compilation – shared implementation
// ---------------------------------------------------------------------------

/**
 * The half of shaders.h that is the same whichever backend is built: the global Slang
 * session, the registry behind the gpu*SlangEXT functions, and the portable `noapi`
 * module every shader imports.
 *
 * This is a header rather than a translation unit of its own because the two backends are
 * separate static libraries that are never linked into the same binary (see the note on
 * GpuSamplerDescListHash in samplers.h). Each one therefore gets its own copy, which is
 * also what keeps the gpu*SlangEXT definitions from being two definitions of one symbol.
 *
 * A backend supplies the other half: the source of the `noapi_backend` module that the
 * portable module is written against. On Vulkan that source is fixed (NOAPI_BACKEND_MODULE
 * in vulkan/noapi.cpp); on WebGPU it is regenerated from the queue's monobuffer and
 * monotexture set every time that set changes (generate_backend_module in
 * webgpu/noapi.cpp).
 */

#ifdef __cplusplus

#include "shaders.h"

#include <slang.h>
#include <slang-com-ptr.h>

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace GPU::shaders {

/**
 * STAGE – Which entry point a compile is looking for. Deliberately not the STAGE enum in
 * compute.h, which names pipeline stages for barriers rather than shader entry points.
 */
enum class SHADER_STAGE { COMPUTE, VERTEX, FRAGMENT, MESH };

/**
 * is_graphics – Whether a stage belongs to a graphics pipeline, which is every stage but
 * compute. What this selects is which set of root pointers the backend module exposes
 * (GPU_COMPUTE below), and a mesh shader is handed the graphics set.
 */
constexpr inline bool is_graphics(SHADER_STAGE stage) { return stage != SHADER_STAGE::COMPUTE; }

// ---------------------------------------------------------------------------
// The portable module
// ---------------------------------------------------------------------------

/**
 * PORTABLE_MODULE – The `noapi` module, as every shader sees it. Identical on both
 * backends; everything that differs between them is reached through `noapi_backend`.
 *
 * What a backend has to supply is deliberately tiny. Memory is two functions —
 * gpuBackendLoadU32 and gpuBackendStoreU32, one 32 bit word at a raw address — and
 * everything a shader actually writes (a typed pointer, an array index, a matrix, a struct
 * of its own) is built out of those two here, generically, by GpuPtr<T> and the IGpuLoadable
 * conformances. The rest of the backend module is only what cannot be said in terms of a
 * word: which root pointers the dispatch or draw was given, and the sampler and texture
 * entry points.
 *
 * GPU_COMPUTE / GPU_GRAPHICS select which root data accessors exist, and GPU_STORES says
 * whether writes are reachable at all — they are not from a WebGPU graphics stage, where
 * the monobuffers are bound read only, so a pointer there has no setter and `p[i] = v`
 * fails to compile rather than silently going nowhere.
 */
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
)slang";

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/**
 * Registry – What the gpu*SlangEXT functions configure: the extra modules and search
 * paths a shader may import, and the macros every compile starts with.
 *
 * `generation` counts changes to any of it. A cached session was built against one value
 * of it and is thrown away once it no longer matches, which is what makes a registration
 * reach pipelines created afterwards.
 */
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

/// The two module names the API owns, which a user registration may not shadow
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

/**
 * global_session – The one IGlobalSession, created on first use. It holds the parsed core
 * module, which is the expensive part of starting the compiler, so every session is built
 * from this rather than from scratch.
 *
 * A global session is explicitly not thread safe, so every use of it (and of anything
 * created from it) is serialized on compile_mutex below.
 */
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

/**
 * SessionCache – The last session built, kept so that a backend recompiling the same
 * shaders against the same bindings (which WebGPU does on every layout change) does not
 * reload `noapi` and the registered modules each time.
 *
 * Keyed by everything a session bakes in: the target, the backend module's source, the
 * macros, and the registry generation. User shaders are loaded into it under a unique
 * name per compile, so nothing accumulated in it is ever stale.
 */
struct SessionCache {
	Slang::ComPtr<slang::ISession> session;
	SlangCompileTarget target = SLANG_TARGET_UNKNOWN;
	std::string backend_module;
	std::string macro_key;
	uint64_t generation = 0;
	uint64_t next_user_module = 0;

	/**
	 * code – What compile() already produced for this session, keyed by stage and source.
	 *
	 * Not just an optimization. A shader is recompiled whenever the bindings move underneath
	 * it, and on WebGPU a render pipeline is rebuilt whenever the depth, blend or index state
	 * it was baked against changes, which can be as often as every frame. Without this each of
	 * those would run the whole compiler again *and* leave another module behind in the
	 * session, which never forgets one. Everything in here was built against this session, so
	 * it is dropped along with it.
	 */
	std::unordered_map<std::string, std::string> code;
};

inline SessionCache& session_cache() {
	static SessionCache instance;
	return instance;
}

/// Drops whatever the compiler cached. Called by every gpu*SlangEXT mutation.
inline void invalidate_sessions() {
	std::lock_guard lock(compile_mutex());
	auto& cache = session_cache();
	cache.session = nullptr;
	cache.target = SLANG_TARGET_UNKNOWN;
	cache.backend_module.clear();
	cache.macro_key.clear();
	cache.generation = 0;
	cache.code.clear();
}

/**
 * get_session – The session for this target and backend module, reusing the cached one
 * when nothing it was built against has changed.
 *
 * Callers hold compile_mutex().
 */
inline slang::ISession* get_session(SlangCompileTarget target, std::string_view backend_module,
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
	if(cache.session && cache.target == target && cache.generation == generation
		&& cache.backend_module == backend_module && cache.macro_key == macro_key)
		return cache.session.get();

	auto* global = global_session();
	if(!global) {
		out_error = "the Slang global session is unavailable";
		return nullptr;
	}

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

	// Vulkan's clip space points the opposite way down the Y axis to WebGPU's. Rather than
	// making every shader flip for one backend and not the other (which is what the two hand
	// written shader sets this replaced had to do), the SPIR-V target is told to invert Y on
	// the way out, so the same source lands the same way up on both.
	// Slang lowers ResourceDescriptorHeap/SamplerDescriptorHeap two different ways. Without this
	// capability it emits a traditional set/binding descriptor array, which a pipeline created with
	// VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT then has to be handed an explicit
	// VkShaderDescriptorSetAndBindingMappingInfoEXT for -- against binding numbers Slang picked. With
	// it, the heap access is emitted as SPV_EXT_descriptor_heap and needs no mapping at all, which is
	// what the Vulkan backend's heap is built around.
	auto descriptor_heap = global->findCapability("spvDescriptorHeapEXT");
	// Mesh shading is what a mesh entry point lowers through, and saying so here rather than
	// letting it be inferred is what keeps the diagnostic for a mesh shader on a target that
	// cannot run one a compile error rather than a silently mis-lowered module. A shader with no
	// mesh entry point emits none of it, so declaring it costs a device without
	// VK_EXT_mesh_shader nothing -- what such a device refuses is the pipeline, which
	// gpuGetCapabilitiesEXT().mesh_shaders says up front.
	auto mesh_shading = global->findCapability("spvMeshShadingEXT");

	slang::CompilerOptionEntry spirv_options[] = {
		{slang::CompilerOptionName::VulkanInvertY, {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}},
		{slang::CompilerOptionName::Capability, {slang::CompilerOptionValueKind::Int, descriptor_heap, 0, nullptr, nullptr}},
		{slang::CompilerOptionName::Capability, {slang::CompilerOptionValueKind::Int, mesh_shading, 0, nullptr, nullptr}},
	};

	slang::TargetDesc target_desc {
		.format = target,
		// Leaving the profile unset makes the target reject compute shaders on some paths,
		// and every stage this API compiles is covered by this one
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
	if(SLANG_FAILED(global->createSession(session_desc, session.writeRef())) || !session) {
		out_error = "failed to create a Slang session";
		return nullptr;
	}

	// The modules a shader may import, innermost first: the backend's half of the ABI, the
	// portable module written against it, and then whatever the program registered.
	//
	// The API's own two have to go in before the registered ones, not after: a module is
	// compiled as it is loaded, so anything it imports has to already be in the session. A
	// registered module that says `import noapi` -- which is the whole point of being able to
	// register one that describes a shared data layout -- fails to find it otherwise. Nothing
	// is lost by the order, since neither of the API's modules can import a registered one.
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

	cache.session = session;
	cache.target = target;
	cache.backend_module = backend_module;
	cache.macro_key = std::move(macro_key);
	cache.generation = generation;
	cache.code.clear();
	cache.next_user_module = 0;
	return cache.session.get();
}

// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------

inline SlangStage to_slang_stage(SHADER_STAGE stage) {
	switch(stage) {
	case SHADER_STAGE::COMPUTE: return SLANG_STAGE_COMPUTE;
	case SHADER_STAGE::VERTEX: return SLANG_STAGE_VERTEX;
	case SHADER_STAGE::FRAGMENT: return SLANG_STAGE_FRAGMENT;
	case SHADER_STAGE::MESH: return SLANG_STAGE_MESH;
	}
	return SLANG_STAGE_NONE;
}

inline const char* stage_name(SHADER_STAGE stage) {
	switch(stage) {
	case SHADER_STAGE::COMPUTE: return "compute";
	case SHADER_STAGE::VERTEX: return "vertex";
	case SHADER_STAGE::FRAGMENT: return "fragment";
	case SHADER_STAGE::MESH: return "mesh";
	}
	return "unknown";
}

/**
 * compile – Turn a shader's Slang source into the code the backend wants, by linking it
 * against the portable module and the backend module and emitting the one entry point
 * whose stage was asked for.
 *
 * The result is the blob Slang produced: SPIR-V words for SLANG_SPIRV, WGSL text for
 * SLANG_WGSL. It is returned by value because the caller outlives the session that
 * produced it.
 *
 * @param target What to emit.
 * @param backend_module Source of the `noapi_backend` module to link against.
 * @param source The shader's own Slang source.
 * @param stage Which entry point to emit.
 * @param extra_defines Macros the backend needs on top of the registered ones.
 * @param out_error Set to why the compile failed, when it does.
 */
inline std::optional<std::string> compile(SlangCompileTarget target, std::string_view backend_module,
	std::string_view source, SHADER_STAGE stage,
	const std::vector<std::pair<std::string, std::string>>& extra_defines, std::string& out_error
) {
	std::lock_guard lock(compile_mutex());

	auto* session = get_session(target, backend_module, extra_defines, out_error);
	if(!session) return std::nullopt;

	auto& cache = session_cache();

	// Keyed by stage and source only: everything else a compile depends on is what the session
	// itself was keyed by, and a new session starts with an empty cache
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

	// Entry points are found by stage rather than by name, so that one module can supply
	// both halves of a graphics pipeline and be handed to the pipeline twice. A stage's
	// entry point has to be unambiguous: two of the same stage is a shader level mistake.
	auto wanted = to_slang_stage(stage);
	Slang::ComPtr<slang::IEntryPoint> found;
	for(SlangInt32 i = 0; i < module->getDefinedEntryPointCount(); ++i) {
		Slang::ComPtr<slang::IEntryPoint> candidate;
		if(SLANG_FAILED(module->getDefinedEntryPoint(i, candidate.writeRef())) || !candidate)
			continue;

		// The stage is only reachable through a layout, and a module's own layout never
		// lists its entry points, so each candidate is composed with the module to ask
		slang::IComponentType* parts[] = {module, candidate.get()};
		Slang::ComPtr<slang::IComponentType> composed;
		if(SLANG_FAILED(session->createCompositeComponentType(parts, 2, composed.writeRef())) || !composed)
			continue;

		auto* layout = composed->getLayout(0, diagnostics.writeRef());
		if(!layout || layout->getEntryPointCount() == 0) continue;
		if(layout->getEntryPointByIndex(0)->getStage() != wanted) continue;

		if(found) {
			out_error = std::string("the shader defines more than one ") + stage_name(stage) + " entry point";
			return std::nullopt;
		}
		found = candidate;
	}

	if(!found) {
		out_error = std::string("the shader defines no ") + stage_name(stage)
			+ " entry point (mark one with [shader(\"" + stage_name(stage) + "\")])";
		return std::nullopt;
	}

	slang::IComponentType* parts[] = {module, found.get()};
	Slang::ComPtr<slang::IComponentType> composed;
	if(SLANG_FAILED(session->createCompositeComponentType(parts, 2, composed.writeRef())) || !composed) {
		out_error = "failed to compose the shader with its modules";
		return std::nullopt;
	}

	Slang::ComPtr<slang::IComponentType> linked;
	diagnostics = nullptr;
	if(SLANG_FAILED(composed->link(linked.writeRef(), diagnostics.writeRef())) || !linked) {
		if(diagnostics && diagnostics->getBufferSize())
			report_diagnostic({(const char*)diagnostics->getBufferPointer(), diagnostics->getBufferSize()});
		out_error = "failed to link the shader";
		return std::nullopt;
	}

	Slang::ComPtr<slang::IBlob> code;
	diagnostics = nullptr;
	if(SLANG_FAILED(linked->getEntryPointCode(0, 0, code.writeRef(), diagnostics.writeRef())) || !code) {
		if(diagnostics && diagnostics->getBufferSize())
			report_diagnostic({(const char*)diagnostics->getBufferPointer(), diagnostics->getBufferSize()});
		out_error = "failed to generate code for the shader";
		return std::nullopt;
	}
	if(diagnostics && diagnostics->getBufferSize())
		report_diagnostic({(const char*)diagnostics->getBufferPointer(), diagnostics->getBufferSize()});

	return cache.code.emplace(std::move(cache_key),
		std::string((const char*)code->getBufferPointer(), code->getBufferSize())).first->second;
}

} // namespace GPU::shaders

// ---------------------------------------------------------------------------
// The public EXT surface
// ---------------------------------------------------------------------------

// Not inline: these are the definitions the rest of the program links against, and this header
// is included by exactly one translation unit per backend library (that backend's noapi.cpp),
// which is what makes one definition of each the right number.
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
