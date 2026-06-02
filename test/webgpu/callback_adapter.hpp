#pragma once

#include <tuple>
#include <type_traits>
#include <utility>
#include <functional>

/*
-------------------------------------------------------------------------------
Generic Callable + C Callback Adapter
-------------------------------------------------------------------------------

This utility adapts C++ callables (lambdas, functors, or function pointers)
into C-style callbacks that pass a `void*` user context pointer.

Supported callback shapes:

    R (*)(void*, Args...)      // leading user data
    R (*)(Args..., void*)      // trailing user data

The adapter stores the callable and exposes static wrapper functions that
forward arguments from the C callback into the stored callable.

-------------------------------------------------------------------------------
Example
-------------------------------------------------------------------------------

C API:

    using callback_t = int(*)(int value, void* user);

Usage:

    auto adapter = callback_adapter{
        [](int v) { return v * 2; }
    };

    register_callback(
        &decltype(adapter)::trailing_user_data,
        &adapter
    );

-------------------------------------------------------------------------------
Lifetime
-------------------------------------------------------------------------------

The adapter object must outlive all callback invocations since the C API
stores only a raw pointer to it.

-------------------------------------------------------------------------------
*/

template<typename T>
struct function_traits;

/// function pointer 
template<typename R, typename... Args>
struct function_traits<R(*)(Args...)> {
	using return_type = R;
	using args_tuple = std::tuple<Args...>;
};

/// member function (const)
template<typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...) const> {
	using return_type = R;
	using args_tuple = std::tuple<Args...>;
};

/// member function (mutable lambda / non-const functor)
template<typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...)> {
	using return_type = R;
	using args_tuple = std::tuple<Args...>;
};

/// generic callable
template<typename T>
struct function_traits : function_traits<decltype(&T::operator())> {};


template<typename Tcallable>
struct callback_adapter {
	Tcallable callable;

	using traits = function_traits<std::remove_reference_t<Tcallable>>;
	using return_type = typename traits::return_type;
	using args_tuple = typename traits::args_tuple;

	/// Leading user data e.g. R callback(void* user_data, Args...)
	template<typename... Args>
	static return_type leading_user_data(void* user_data, Args... args)
		requires (std::is_invocable_r_v<return_type, Tcallable&, Args...>)
	{
		auto* self = static_cast<callback_adapter*>(user_data);
		return std::invoke(self->callable, std::forward<Args>(args)...);
	}

	/// Trailing user data e.g R callback(Args..., void* user_data)
	template<typename... Args>
	static return_type trailing_user_data(Args... args, void* user_data)
		requires (std::is_invocable_r_v<return_type, Tcallable&, Args...>)
	{
		auto* self = static_cast<callback_adapter*>(user_data);
		return std::invoke(self->callable, std::forward<Args>(args)...);
	}
};