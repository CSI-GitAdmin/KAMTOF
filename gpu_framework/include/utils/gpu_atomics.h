#ifndef GPU_ATOMICS_H
#define GPU_ATOMICS_H

#include <sycl/atomic_ref.hpp> // For SYCL atomics

// Strict atomics are enabled by defeault in DEBUG and ASAN mode. To enable it in all modes, just comment the #if and #endif
#if ! defined (NDEBUG) || defined (ASAN_ENABLED)
#define STRICT_GPU_ATOMICS
#endif

namespace GDF
{

// NOTE: The most strict memory_order NVIDIA/AMD support is 'acq_rel' which is one lower than the strictest one
// supported by the SYCL standard which is 'seq_cst'
#ifdef STRICT_GPU_ATOMICS // Do atomics with atmost consistency between runs
constexpr sycl::memory_order mem_order = sycl::memory_order::acq_rel;
#else // Do atomics the fastest possible way
constexpr sycl::memory_order mem_order = sycl::memory_order::relaxed;
#endif

constexpr sycl::memory_scope mem_scope = sycl::memory_scope::device;
constexpr sycl::access::address_space address_space = sycl::access::address_space::global_space;

template<class T>
SYCL_EXTERNAL void atomic_add(T& data, const T& value)
{
   sycl::atomic_ref<T, mem_order, mem_scope, address_space> atomic_data = sycl::atomic_ref<T, mem_order, mem_scope, address_space>(data);
   atomic_data.fetch_add(value);
}

template<class T>
SYCL_EXTERNAL void atomic_sub(T& data, const T value)
{
   sycl::atomic_ref<T, mem_order, mem_scope, address_space> atomic_data = sycl::atomic_ref<T, mem_order, mem_scope, address_space>(data);
   atomic_data.fetch_sub(value);
}

template<class T>
SYCL_EXTERNAL void atomic_max(T& data, const T value)
{
   sycl::atomic_ref<T, mem_order, mem_scope, address_space> atomic_data = sycl::atomic_ref<T, mem_order, mem_scope, address_space>(data);
   atomic_data.fetch_max(value);
}

template<class T>
SYCL_EXTERNAL void atomic_min(T& data, const T value)
{
   sycl::atomic_ref<T, mem_order, mem_scope, address_space> atomic_data = sycl::atomic_ref<T, mem_order, mem_scope, address_space>(data);
   atomic_data.fetch_min(value);
}

} // namespace GDF

#endif // GPU_ATOMICS_H