#ifndef GPU_API_FUNCTIONS_HPP
#define GPU_API_FUNCTIONS_HPP

#include "gpu_manager_t.h" // For GPUManager_t functions
#include "gpu_globals.h" // For gpu_manager
#include "gpu_atomics.h" // For GPU atomics
#include "silo.h"
#include "datasetstoragegpu.h"

namespace GDF
{

inline void set_gpu_global_local_range(const uint64_t (&glob_range)[1], const uint64_t (&locl_range)[1])
{
   gpu_manager->set_global_local_range_internal(glob_range, locl_range);
}

inline void set_gpu_global_local_range(const uint64_t (&glob_range)[2], const uint64_t (&locl_range)[2])
{
   gpu_manager->set_global_local_range_internal(glob_range, locl_range);
}

inline void set_gpu_global_local_range(const uint64_t (&glob_range)[3], const uint64_t (&locl_range)[3])
{
   gpu_manager->set_global_local_range_internal(glob_range, locl_range);
}

void transfer_to_gpu(const dataSetBase* const dsb_entry, transfer_mode_t transfer_mode);

template <typename ... Types>
void transfer_to_gpu_move(const Types& ... dss_objs)
{
   gpu_manager->transfer_to_gpu_move_internal(dss_objs...);
}

template <typename ... Types>
void transfer_to_gpu_readonly(const Types& ... dss_objs)
{
   gpu_manager->transfer_to_gpu_readonly_internal(dss_objs...);
}

template <typename ... Types>
void transfer_to_gpu_copy(const Types& ... dss_objs)
{
   gpu_manager->transfer_to_gpu_copy_internal(dss_objs...);
}

template <typename ... Types>
void transfer_to_gpu_noinit(const Types& ... dss_objs)
{
   gpu_manager->transfer_to_gpu_noinit_internal(dss_objs...);
}

template <typename ... Types>
void transfer_to_gpu_syncandmove(const Types& ... dss_objs)
{
   gpu_manager->transfer_to_gpu_syncandmove_internal(dss_objs...);
}

void transfer_to_cpu(const dataSetBase* const dsb_entry, transfer_mode_t transfer_mode);

template <typename ... Types>
void transfer_to_cpu_move(Types& ... dss_objs)
{
   gpu_manager->transfer_to_cpu_move_internal(dss_objs...);
}

template <typename ... Types>
void transfer_to_cpu_readonly(Types& ... dss_objs)
{
   gpu_manager->transfer_to_cpu_readonly_internal(dss_objs...);
}

template <typename ... Types>
void transfer_to_cpu_copy(Types& ... dss_objs)
{
   gpu_manager->transfer_to_cpu_copy_internal(dss_objs...);
}

template <typename ... Types>
void transfer_to_cpu_noinit(Types& ... dss_objs)
{
   gpu_manager->transfer_to_cpu_noinit_internal(dss_objs...);
}

template <typename ... Types>
void transfer_to_cpu_syncandmove(Types& ... dss_objs)
{
   gpu_manager->transfer_to_cpu_syncandmove_internal(dss_objs...);
}

template<typename T, typename... Us>
void submit_to_gpu(Us&&... args)
{
   gpu_manager->submit_to_gpu_internal<T>(args...);
}

template<typename T, typename... Us>
void submit_to_gpu_single_workgroup(Us&&... args)
{
   gpu_manager->submit_to_gpu_single_workgroup_internal<T>(args...);
}

template<typename T, typename... Us>
void single_task_gpu(Us&&... args)
{
   gpu_manager->single_task_gpu_internal<T>(args...);
}

inline sycl::queue& get_gpu_queue()
{
   return gpu_manager->get_queue_internal();
}

// Device wide barrier
inline void gpu_barrier()
{
   get_gpu_queue().wait();
   return;
}

inline void print_gpu_memcpy_counts()
{
   return gpu_manager->print_gpu_memcpy_counts_internal();
}

template <class T>
inline bool is_usm_pointer(const T* const ptr)
{
   sycl::usm::alloc m_type = sycl::get_pointer_type(static_cast<const void*>(ptr), get_gpu_queue().get_context());
   if(m_type == sycl::usm::alloc::unknown)
      return false;
   return true;
}

// Allocates a gpu variable with given number of elements either in shared or device mode
template <class T, bool is_malloc_shared = false>
T* malloc_gpu_var(const uint64_t& num_elements)
{
   assert(num_elements >= 0);
   return gpu_manager->malloc_gpu_var_internal<T, is_malloc_shared>(num_elements);
}

// Copies the data from src to dest
template <class T>
void memcpy_gpu_var(T* const dest, const T * const src, const uint64_t& num_elements)
{
   assert(num_elements >= 0);
   if(is_usm_pointer(src))
   {
      if(is_usm_pointer(dest))
      {
         gpu_manager->DtoD_memcpy_counter++;
      }
      else
      {
         gpu_manager->DtoH_memcpy_counter++;
      }
   }
   else
   {
      if(is_usm_pointer(dest))
      {
         gpu_manager->HtoD_memcpy_counter++;
      }
      else
      {
         log_msg("Error in memcpy_gpu_var: Memcpy between two CPU pointers should use memcpy and not memcpy_gpu_var");
      }
   }
   gpu_manager->memcpy_gpu_var_internal<T>(dest, src, num_elements);
}

// Free the gpu variable passed
template <class T>
void free_gpu_var(T* gpu_var)
{
   gpu_manager->free_gpu_var_internal<T>(gpu_var);
}

// API Function for memeset
// 'val' is cast to unsigned char and asssigned to every byte
template <class T>
void memset_gpu_var(T* gpu_var, const int val, const uint64_t& num_elements)
{
   assert(gpu_var && (num_elements >=0));
   sycl::queue& m_que = get_gpu_queue();

   uint64_t num_bytes;
   if constexpr(std::is_same<T, void>::value)
      num_bytes = num_elements;
   else
      num_bytes = (sizeof(T)*num_elements);

   m_que.submit([&](sycl::handler& cgh)
                {
                   cgh.memset(gpu_var, val, num_bytes);
                });
   m_que.wait();
}

SYCL_EXTERNAL inline uint64_t get_1d_index(const sycl::nd_item<3>& item)
{
   return item.get_global_id(2);
}

SYCL_EXTERNAL inline uint64_t get_1d_stride(const sycl::nd_item<3>& item)
{
   return item.get_global_range(2);
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
void memcpy_gpu_var(dataSetStorage<T, TYPE, DIMS>& dss_dest,const dataSetStorage <T, TYPE, DIMS>& dss_src)
{
   assert(dss_src.exists());
   assert(dss_dest.exists());
   assert((dss_src.byte_size() == dss_dest.byte_size()) && (dss_src.size() == dss_dest.size()));

   // If this object does not have a gpu_instance, create one by copying the older values from CPU
   const GPUInstance_t* const src_gpu_instance = dss_src.get_gpu_instance();
   if(!src_gpu_instance)
   {
      GDF::transfer_to_gpu_move(dss_src);
   }
   else
   {
      // If the GPU object already exist, make sure the data to be copied is not OUT_OF_DATE
      if(src_gpu_instance->get_xpu_data_status(GDF::xpu_t::GPU) != GDF::xpu_data_status_t::UP_TO_DATE_READ &&
          src_gpu_instance->get_xpu_data_status(GDF::xpu_t::GPU) != GDF::xpu_data_status_t::UP_TO_DATE_WRITE)
      {
         GDF::transfer_to_gpu_readonly(dss_src);
      }
   }
   assert(src_gpu_instance);
   const T* const src_data = dss_src.gpu_data();
   assert(src_data);

   // Destination should be UP_TO_DATE_WRITE at the end of this
   GDF::transfer_to_gpu_noinit(dss_dest);
   assert(dss_dest.get_gpu_instance());
   T* const dest_data = dss_dest.gpu_data();
   assert(dest_data);

   GDF::memcpy_gpu_var(dest_data, src_data, dss_src.size());
}

inline sycl_device_vendor_t get_device_vendor_t()
{
   return gpu_manager->get_device_vendor_t_internal();
}

// GPU MATH API FUNCTIONS

class kg_axpby
{
public:
   kg_axpby(const uint64_t num_elements_in,
            const strict_fp_t a_in,
            const strict_fp_t* const x_in,
            const strict_fp_t b_in,
            const strict_fp_t* const y_in,
            strict_fp_t* const result_in):
      num_elements(num_elements_in),
      a(a_in),
      x(x_in),
      b(b_in),
      y(y_in),
      result(result_in)
   {}

   SYCL_EXTERNAL void operator()(sycl::nd_item<3> itm) const;

private:
   const uint64_t num_elements;
   const strict_fp_t a;
   const strict_fp_t* const x;
   const strict_fp_t b;
   const strict_fp_t* const y;
   strict_fp_t* const result;
};

void axpy(const uint64_t num_elements, const strict_fp_t alpha, const strict_fp_t* const x, strict_fp_t* const y, uint8_t impl_type = 0);

void dot_product(const uint64_t num_elements, const strict_fp_t * const vec_a, const strict_fp_t * const vec_b, strict_fp_t *result, uint8_t impl_type = 0);

void linf_norm(const uint64_t num_elements, const strict_fp_t* const vec, strict_fp_t * const result);

void l0_norm(const uint64_t num_elements, const strict_fp_t* const vec, strict_fp_t * const result);

void l1_norm(const uint64_t num_elements, const strict_fp_t* const vec, strict_fp_t* const result, uint8_t impl_type = 0);

void l2_norm(const uint64_t num_elements, const strict_fp_t* const vec, strict_fp_t* const result, uint8_t impl_type = 0);

void csr_matvec(const uint64_t nrow, const uint64_t ncol, const uint64_t nnz, const int* const ia, const int* const ja, const strict_fp_t * const matval, const
                strict_fp_t * const vec, strict_fp_t* const result, const int impl_type = 0);

void gpu_vec_sum(const uint64_t num_elements, const strict_fp_t* const vec, strict_fp_t * const result);

void transfer_all_silo_vars_to_cpu(const GDF::transfer_mode_t transfer_mode);

void copy_all_silo_vars_to_cpu();

void move_all_silo_vars_to_cpu();

template <class  T>
void transfer_vars_to_gpu_internal(T&& m_data)
{
   return;
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
void transfer_vars_to_gpu_internal(dataSetStorageGPU<T, TYPE, DIMS>& dss_obj)
{
   GDF::transfer_to_gpu(dss_obj.cpu_dss_ptr, transfer_mode_t::MOVE);
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
void transfer_vars_to_gpu_internal(const dataSetStorageGPU<T, TYPE, DIMS>& dss_obj)
{
   GDF::transfer_to_gpu(dss_obj.cpu_dss_ptr, transfer_mode_t::READ_ONLY);
}

template<uint8_t N, typename... Us>
void transfer_vars_to_gpu_impl(Us&&... args)
{
   static_assert(N ==  sizeof...(args));
   (transfer_vars_to_gpu_internal(args), ...);
}

} // namespace GDF

#endif // GPU_API_FUNCTIONS_HPP
