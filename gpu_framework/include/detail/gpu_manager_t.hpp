#ifndef GPU_MANAGER_T_HPP
#define GPU_MANAGER_T_HPP

#include "gpu_manager_t.h"
#include "gpu_helpers.h" // For transfer_mode_to_cstr
#include "gpu_globals.h"
#include <type_traits>

namespace GDF
{

// Primary template: Default case, transfer_vars_to_gpu<uint8_t>() does not exist
template <typename, typename = std::void_t<>>
struct has_extractor : std::false_type {};

// Specialization: Detects if transfer_vars_to_gpu<uint8_t>() exists in the class
template <typename T>
struct has_extractor<T, std::void_t<decltype(std::declval<T>().template transfer_vars_to_gpu<1>())>> : std::true_type {};

// Function that calls transfer_vars_to_gpu<uint8_t>() only if it exists
template <typename T, typename... Us>
void GPUManager_t::callExtractorIfExists(Us&&... args)
{
   if constexpr (has_extractor<T>::value) // Compile-time check
   {
      constexpr uint8_t N = sizeof...(args);
      T obj{extract_gpu_data_for_extractor(std::forward<Us>(args))...};
      obj.template transfer_vars_to_gpu<N>();
   }
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
void GPUManager_t::transfer_to_gpu_internal(const dataSetStorage<T, TYPE, DIMS>& dss_obj,
                                            transfer_mode_t transfer_mode /* = transfer_mode_t::NOT_SET */)
{
   GPUInstance_t& cur_gpu_instance= setup_gpu_instance_and_data_ptr(dss_obj);

   this->transfer_to_gpu_internal(cur_gpu_instance.get_cpu_dsb_ptr(), transfer_mode);
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
void GPUManager_t::transfer_to_cpu_internal(const dataSetStorage<T, TYPE, DIMS>& dss_obj,
                                            transfer_mode_t transfer_mode /* = transfer_mode_t::MOVE */)
{
   const GPUInstance_t* const cur_gpu_instance = dss_obj.get_gpu_instance();
   if(cur_gpu_instance)
      this->transfer_to_cpu_internal(cur_gpu_instance->get_cpu_dsb_ptr(), transfer_mode);
}

template <typename ... Types>
void GPUManager_t::transfer_to_gpu_move_internal(const Types& ... dss_objs)
{
   // Fold expressions reduces (folds) a parameter pack over a binary operator (which is ',' here)
   (transfer_to_gpu_internal(dss_objs, transfer_mode_t::MOVE), ...);
}

template <typename ... Types>
void GPUManager_t::transfer_to_gpu_readonly_internal(const Types& ... dss_objs)
{
   (transfer_to_gpu_internal(dss_objs, transfer_mode_t::READ_ONLY), ...);
}

template <typename ... Types>
void GPUManager_t::transfer_to_gpu_copy_internal(const Types& ... dss_objs)
{
   (transfer_to_gpu_internal(dss_objs, transfer_mode_t::COPY), ...);
}

template <typename ... Types>
void GPUManager_t::transfer_to_gpu_noinit_internal(const Types& ... dss_objs)
{
   (transfer_to_gpu_internal(dss_objs, transfer_mode_t::NOT_INITIALIZE), ...);
}

template <typename ... Types>
void GPUManager_t::transfer_to_gpu_syncandmove_internal(const Types& ... dss_objs)
{
   (transfer_to_gpu_internal(dss_objs, transfer_mode_t::SYNC_AND_MOVE), ...);
}

template <typename ... Types>
void GPUManager_t::transfer_to_cpu_move_internal(Types& ... dss_objs)
{
   // Fold expressions reduces (folds) a parameter pack over a binary operator (which is ',' here)
   (transfer_to_cpu_internal(dss_objs, transfer_mode_t::MOVE), ...);
}

template <typename ... Types>
void GPUManager_t::transfer_to_cpu_readonly_internal(Types& ... dss_objs)
{
   (transfer_to_cpu_internal(dss_objs, transfer_mode_t::READ_ONLY), ...);
}

template <typename ... Types>
void GPUManager_t::transfer_to_cpu_copy_internal(Types& ... dss_objs)
{
   (transfer_to_cpu_internal(dss_objs, transfer_mode_t::COPY), ...);
}

template <typename ... Types>
void GPUManager_t::transfer_to_cpu_noinit_internal(Types& ... dss_objs)
{
   (transfer_to_cpu_internal(dss_objs, transfer_mode_t::NOT_INITIALIZE), ...);
}

template <typename ... Types>
void GPUManager_t::transfer_to_cpu_syncandmove_internal(Types& ... dss_objs)
{
   (transfer_to_cpu_internal(dss_objs, transfer_mode_t::SYNC_AND_MOVE), ...);
}

/* In this function we take in and return an universal reference (&&) instead of a normal refernce (&)
 * as we need to handle both named variables (lvalues) and unnamed temproaries (rvalues).
*/
template <class  T>
T&& GPUManager_t::extract_gpu_data_for_kernel(T&& m_data)
{
   // Generic Function for named and unnamed variables of primitive types (e.g 'int', 'strict_fp_t')
   // We need the std::forward here to return an rvalue reference as an rvalue and reference and a lvalue reference as an lvalue reference
   return std::forward<T>(m_data);
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
dataSetStorageGPURead<T, TYPE, DIMS>& GPUManager_t::extract_gpu_data_for_kernel(dataSetStorageRead<T, TYPE, DIMS>& dss_obj)
{
#ifndef NDEBUG
   // If the variable does not exist, make sure the status is not allocated
   if(!dss_obj.exists())
   {
      assert(dss_obj.get_gpu_instance()->get_xpu_data_status(GDF::xpu_t::CPU) == GDF::xpu_data_status_t::NOT_ALLOCATED);
   }
#endif

   // Check if the gpu_data is not OUT_OF_DATE (or) RESIZED_ON_CPU
   const xpu_data_status_t& gpu_data_status = dss_obj.get_xpu_data_status(xpu_t::GPU);
   if(gpu_data_status == xpu_data_status_t::OUT_OF_DATE || gpu_data_status == xpu_data_status_t::RESIZED_ON_CPU)
   {
      std::string error = "Error in extract_gpu_data_for_kernel for SILO variable " + static_cast<std::string>(dss_obj.name()) + " : The gpu_data_status of the SILO variable is " + GDF::xpu_data_status_to_cstr(gpu_data_status) + ". It must either be UP_TO_DATE_READ (or) UP_TO_DATE_WRITE (or) TEMP_WRITE!";
      log_msg<CDF::LogLevel::ERROR>(error);
   }

   // This function specifically takes in a DSS object and returns respective the DSSGPU object
   assert(dss_obj.get_gpu_instance());
   dataSetBaseGPURead* dsb_gpu = dss_obj.get_gpu_instance()->get_gpu_dsb_ptr();
   assert(dsb_gpu);
   assert(dsb_gpu->void_data() || !dss_obj.exists() || (dss_obj.size() == 0)); // The data can be NULL either if the object doesn't exist (or) size is 0

   // We would be returning this by casting it up using the templates of the GPU variable which might not be the same as the original CPU DSS's templates
   return *(static_cast<dataSetStorageGPURead<T, TYPE, DIMS>*>(dsb_gpu));
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
dataSetStorageGPU<T, TYPE, DIMS>& GPUManager_t::extract_gpu_data_for_kernel(dataSetStorage<T, TYPE, DIMS>& dss_obj)
{
#ifndef NDEBUG
   // If the variable does not exist, make sure the status is not allocated
   if(!dss_obj.exists())
   {
      assert(dss_obj.get_gpu_instance()->get_xpu_data_status(GDF::xpu_t::CPU) == GDF::xpu_data_status_t::NOT_ALLOCATED);
   }
#endif

   // Check if the gpu_data is not OUT_OF_DATE (or) RESIZED_ON_CPU
   const xpu_data_status_t& gpu_data_status = dss_obj.get_xpu_data_status(xpu_t::GPU);
   if(gpu_data_status == xpu_data_status_t::OUT_OF_DATE || gpu_data_status == xpu_data_status_t::RESIZED_ON_CPU)
   {
      std::string error = "Error in extract_gpu_data_for_kernel for SILO variable " + static_cast<std::string>(dss_obj.name()) + " : The gpu_data_status of the SILO variable is " + GDF::xpu_data_status_to_cstr(gpu_data_status) + ". It must either be UP_TO_DATE_READ (or) UP_TO_DATE_WRITE (or) TEMP_WRITE!";
      log_msg<CDF::LogLevel::ERROR>(error);
   }

   // This function specifically takes in a DSS object and returns respective the DSSGPU object
   assert(dss_obj.get_gpu_instance());
   dataSetBaseGPU* dsb_gpu = dss_obj.get_gpu_instance()->get_gpu_dsb_ptr();
   assert(dsb_gpu);
   assert(dsb_gpu->void_data() || !dss_obj.exists() || (dss_obj.size() == 0)); // The data can be NULL either if the object doesn't exist (or) size is 0

   // We would be returning this by casting it up using the templates of the GPU variable which might not be the same as the original CPU DSS's templates
   return *(static_cast<dataSetStorageGPU<T, TYPE, DIMS>*>(dsb_gpu));
}

template <class  T>
T&& GPUManager_t::extract_gpu_data_for_extractor(T&& m_data)
{
   // Generic Function for named and unnamed variables of primitive types (e.g 'int', 'strict_fp_t')
   // We need the std::forward here to return an rvalue reference as an rvalue and reference and a lvalue reference as an lvalue reference
   return std::forward<T>(m_data);
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
dataSetStorageGPURead<T, TYPE, DIMS>& GPUManager_t::extract_gpu_data_for_extractor(dataSetStorageRead<T, TYPE, DIMS>& dss_obj)
{
   GPUInstance_t& cur_gpu_instance = setup_gpu_instance_and_data_ptr(dss_obj);
   dataSetBaseGPU* dsb_gpu = cur_gpu_instance.get_gpu_dsb_ptr();

   // We would be returning this by casting it up using the templates of the GPU variable which might not be the same as the original CPU DSS's templates
   return *(static_cast<dataSetStorageGPU<T, TYPE, DIMS>*>(dsb_gpu));
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
dataSetStorageGPU<T, TYPE, DIMS>& GPUManager_t::extract_gpu_data_for_extractor(dataSetStorage<T, TYPE, DIMS>& dss_obj)
{
   GPUInstance_t& cur_gpu_instance = setup_gpu_instance_and_data_ptr(dss_obj);
   dataSetBaseGPU* dsb_gpu = cur_gpu_instance.get_gpu_dsb_ptr();

   // We would be returning this by casting it up using the templates of the GPU variable which might not be the same as the original CPU DSS's templates
   return *(static_cast<dataSetStorageGPU<T, TYPE, DIMS>*>(dsb_gpu));
}

template<typename T, bool async, typename... Us>
void GPUManager_t::submit_to_gpu_internal(Us&&... args)
{
#ifdef GPU_AUTO_TRANSFER
   callExtractorIfExists<T>(args...);
#endif

   // Call the actual kernel
   m_que.parallel_for(sycl::nd_range<3>{global_range,local_range}, T{extract_gpu_data_for_kernel(std::forward<Us>(args))...});

   if constexpr (!async)
      m_que.wait();
}


template<typename T, typename... Us>
void GPUManager_t::submit_to_gpu_single_workgroup_internal(Us&&... args)
{
   // Call the actual kernel
   m_que.parallel_for(sycl::nd_range<3>{{1,1,SINGLE_WG_SIZE},{1,1,SINGLE_WG_SIZE}}, T{extract_gpu_data_for_kernel(std::forward<Us>(args))...});
   m_que.wait();
}


template<typename T, typename... Us>
void GPUManager_t::single_task_gpu_internal(Us&&... args)
{
   // Call the actual kernel
   m_que.single_task(T{extract_gpu_data_for_kernel(std::forward<Us>(args))...});
   m_que.wait();
}

template <class T, bool is_malloc_shared /* = false */>
T* GPUManager_t::malloc_gpu_var_internal(const size_t& num_elements)
{
   static_assert(!(std::is_same<T, void>::value && is_malloc_shared),
                 "Cannot allocate GPU shared memory for void type");

   T* result = nullptr;

   if constexpr (std::is_same<T, void>::value && is_malloc_shared)
      return nullptr;
   else
      result = malloc_gpu_var_impl<T, is_malloc_shared>(num_elements);
#ifdef GPU_MEM_LOG
   strict_fp_t size;
   if constexpr (std::is_same<void, T>::value)
   {
      size = num_elements;
   }
   else
   {
      size = num_elements * sizeof(T);
   }
   gpu_mem_map.insert({static_cast<void*>(result),size});
   tot_gpu_mem_used += size;
   CVG::string ptr_as_string;
   ptr_as_string.resize(100);
   sprintf(ptr_as_string.data(), "%p", result);
   CVG::string msg = "ptr = " + std::string(ptr_as_string.c_str()) + " | Allocated (MB)  = " + std::to_string(size/1e+06) + " | Running total memory usage(MB) = "
                     + std::to_string(tot_gpu_mem_used/1.0e+06);
   gpu_mem_usage_log << (tot_gpu_mem_used/1.0E+06) << "\n";
   log_msg(msg);
#endif
   return result;
}

// For void type, num_elements == num bytes allocated
template <class T, bool is_malloc_shared>
T* GPUManager_t::malloc_gpu_var_impl(const size_t& num_elements)
{
   assert(num_elements >= 0);
   T* result = nullptr;

   if constexpr (is_malloc_shared)
      result = sycl::malloc_shared<T>(num_elements, m_que);
   else if constexpr (std::is_same<T, void>::value)
      result = sycl::malloc_device(num_elements, m_que);
   else
      result = sycl::malloc_device<T>(num_elements, m_que);

   assert(result || (num_elements == 0));
   return result;
}

template <class T>
void GPUManager_t::memcpy_gpu_var_internal(T* dest, const T * const src, const size_t& num_elements)
{
   if(num_elements != 0)
   {
      assert(src);
      assert(dest);
      if constexpr (std::is_same<T, void>::value)
         m_que.memcpy(dest, src, num_elements).wait();
      else
         m_que.memcpy(dest, src, num_elements*sizeof(T)).wait();
   }
   return;
}

template <class T>
void GPUManager_t::free_gpu_var_internal(T* gpu_var)
{
   if(!gpu_var)
      return;
#ifdef GPU_MEM_LOG
   auto it = gpu_mem_map.find(gpu_var);
   if(it == gpu_mem_map.end())
   {
      log_msg<CDF::LogLevel::ERROR>("Error in free_gpu_var: GPU_MEM_MAP Look up failed!");
   }
   tot_gpu_mem_used -= it->second;
   CVG::string ptr_as_string;
   ptr_as_string.resize(100);
   sprintf(ptr_as_string.data(), "%p", gpu_var);
   std::string msg = "ptr = " + std::string(ptr_as_string.c_str()) + " | Released (MB)  = " + std::to_string(it->second/1e+06) +
                     " | Running total memory usage(MB) = " + std::to_string(tot_gpu_mem_used/1.0e+06);
   log_msg(msg);
   gpu_mem_usage_log << (tot_gpu_mem_used/1.0E+06) << "\n";
   gpu_mem_map.erase(static_cast<void*>(gpu_var));
#endif
   assert(gpu_var);
   sycl::free(gpu_var, m_que);
   return;
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /*= ZEROD*/>
void GPUManager_t::allocate_gpu_instance(const dataSetStorage<T, TYPE, DIMS>& dss_obj)
{
   GPUInstance_t* cur_gpu_instance = nullptr;
   if(!dss_obj.get_gpu_instance())
   {
      cur_gpu_instance = new GPUInstance_t(dss_obj.m_dsb());
      dss_obj.set_gpu_instance(cur_gpu_instance);
   }
   cur_gpu_instance = const_cast<GPUInstance_t*>(dss_obj.get_gpu_instance());
   assert(cur_gpu_instance); // Should not be null
   if(!cur_gpu_instance->get_gpu_dsb_ptr()) // If the m_gpu_dss variable is not allocated, allocate that
   {
      /* 
      *  Note that the members of DSBGPU is default intialized to null/0 values at this point and
      *  will only be populated correctly after the calling allocate_gpu_data_ptr()
      */
      dataSetStorageGPU<T, TYPE, DIMS>* dss_gpu_obj = new dataSetStorageGPU<T, TYPE, DIMS>(dss_obj, this);

      // DSSGPU pointer is stored as DSB pointer inside gpu_instance as it is not templated
      // Should cast up to DSSGPU when members/data of DSSGPU are accessed
      cur_gpu_instance->set_gpu_dsb_ptr(static_cast<dataSetBaseGPU*>(dss_gpu_obj));
   }
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /*= ZEROD*/>
void GPUManager_t::deallocate_gpu_instance(const dataSetStorage<T, TYPE, DIMS>& dss_obj)
{
   assert(dss_obj.exists());
   GPUInstance_t* cur_gpu_instance= const_cast<GPUInstance_t*>(dss_obj.get_gpu_instance());
   assert(cur_gpu_instance); // Should not be NULL
   assert(cur_gpu_instance->get_gpu_dsb_ptr()); // Should not be NULL

   delete cur_gpu_instance->get_gpu_dsb_ptr();
   cur_gpu_instance->set_gpu_dsb_ptr(nullptr);

   delete cur_gpu_instance;
   dss_obj.set_gpu_instance(nullptr);
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
GPUInstance_t& GPUManager_t::setup_gpu_instance_and_data_ptr(const dataSetStorage<T, TYPE, DIMS>& dss_obj)
{
   if(!dss_obj.get_gpu_instance())
   {
      allocate_gpu_instance(dss_obj);
   }
   GPUInstance_t& cur_gpu_instance= *(const_cast<GPUInstance_t*>(dss_obj.get_gpu_instance()));

   // If the CPU silo object doesn't exist, don't allocate gpu_data_ptr
   if(dss_obj.exists() && (cur_gpu_instance.get_xpu_data_status(xpu_t::GPU) == xpu_data_status_t::NOT_ALLOCATED))
   {
      allocate_gpu_data_ptr(&cur_gpu_instance, true);
   }
   return cur_gpu_instance;
}

} // namespace CDF

#endif // GPU_MANAGER_T_HPP
