#ifndef GPU_MANAGER_T_H
#define GPU_MANAGER_T_H

#include <sycl/sycl.hpp> // For SYCL members and functions
#include "silo_fwd.h" // For DSB and DSS
#include "datasetstoragegpu.h" // For DSSGPU, SILO error handling, GPUInstance member functions
#include "gpu_silo_fwd.h" // For DSSGPURead
#include "logger.hpp"
#include "cpu_globals.h"

namespace GDF
{

// Custom device selector for sycl::queue
inline sycl::device custom_device_selector(const sycl_device_t& m_device_type, const std::string& m_device_name, const bool allow_hypergpu = false)
{
   // Get the list of platforms and devices with the requested device and name
   std::vector<sycl::platform> platform_list(sycl::platform::get_platforms());
   std::vector<std::vector<sycl::device>> device_list(platform_list.size());
   std::vector<uint8_t> num_gpus(platform_list.size(), 0);
   for (uint8_t pltfrm_idx = 0; pltfrm_idx < platform_list.size(); pltfrm_idx++)
   {
      const sycl::platform& cur_platform = platform_list[pltfrm_idx];
      device_list[pltfrm_idx].reserve(cur_platform.get_devices().size());
      for (const sycl::device& cur_device : cur_platform.get_devices())
      {
         if (m_device_type == sycl_device_t::DEFAULT && m_device_name == "")
         {
            device_list[pltfrm_idx].emplace_back(cur_device);
            if(cur_device.is_gpu())
               num_gpus[pltfrm_idx]++;
         }
         else
         {
            if( (m_device_type == sycl_device_t::DEFAULT)             ||
                (m_device_type == sycl_device_t::CPU && cur_device.is_cpu()) ||
                (m_device_type == sycl_device_t::GPU && cur_device.is_gpu()) ||
                (m_device_type == sycl_device_t::ACCELERATOR && cur_device.is_accelerator()))
            {
               std::string cur_device_name(cur_device.get_info<sycl::info::device::name>());
               std::string m_device(m_device_name);
               std::transform(cur_device_name.begin(), cur_device_name.end(), cur_device_name.begin(), ::toupper);
               std::transform(m_device.begin(), m_device.end(), m_device.begin(), ::toupper);
               if(cur_device_name.find(m_device) != std::string::npos)
               {
                  device_list[pltfrm_idx].emplace_back(cur_device);
               }
            }
         }
      }
   }

   if (m_device_type == sycl_device_t::DEFAULT && m_device_name == "")
   {
      uint8_t platform_to_select = std::distance(num_gpus.begin(), std::max_element(num_gpus.begin(), num_gpus.end()));
      uint8_t num_devices = device_list[platform_to_select].size();
      if(allow_hypergpu)
      {
         return device_list[platform_to_select][local_rank % num_devices];
      }
      else
      {
         if(num_devices >= local_numprocs)
            return device_list[platform_to_select][local_rank];
      }
   }
   else
   {
      // By now we should have the list the devices in every platform which is okay to be selected
      for (uint8_t pltfrm_idx = 0; pltfrm_idx < platform_list.size(); pltfrm_idx++)
      {
         const uint8_t num_devices = device_list[pltfrm_idx].size();
         if(allow_hypergpu)
         {
            if(num_devices == 0)
               continue;
            return device_list[pltfrm_idx][local_rank % num_devices];
         }
         else
         {
            if(num_devices < local_numprocs)
               continue;
            return device_list[pltfrm_idx][local_rank];
         }
      }
   }

   if(allow_hypergpu)
   {
      std::string err_msg = "No device of requested type is available!";
      log_msg<CDF::LogLevel::ERROR>(err_msg);
   }
   else
   {
      std::string err_msg = " Number of device of requested type is lesser than the number of cores launched on this node!";
      log_msg<CDF::LogLevel::ERROR>(err_msg);
   }

   return sycl::device();
}

class GPUManager_t
{
public:
   /* Using "const uint64_t (&glob_range)[3]" to clarify to the compiler that we want a reference to an array,
    * rather than the (invalid) array of references (i.e. uint64_t & glob_range[3])
    *
    * Also, this only accepts an array of 3 integers, rather than an array of arbitrary size (which will happen if we use "uint64_t glob_range[3]")
   */
   GPUManager_t(sycl_device_t deviceType, const uint64_t (&glob_range)[3], const uint64_t (&locl_range)[3], const std::string& device_name = ""):
      m_que (custom_device_selector(deviceType, device_name, true)),
      global_range{glob_range[0], glob_range[1], glob_range[2]}, local_range{locl_range[0], locl_range[1], locl_range[2]},
      HtoD_memcpy_counter(0), DtoH_memcpy_counter(0), DtoD_memcpy_counter(0)
   {
      // We currently only support 1D grids and blocks
      assert(glob_range[0] == 1);
      assert(glob_range[1] == 1);
      assert(locl_range[0] == 1);
      assert(locl_range[1] == 1);
      assert(glob_range[2] % locl_range[2] == 0);

      print_device_name();
      set_device_properties();

      std::string glob_range_msg = "Total number of threads launched for the GPU solver : " + std::to_string(global_range[2]);
      std::string locl_range_msg = "Number of threads launched per workgroup for the GPU solver : " + std::to_string(local_range[2]);
      log_msg(glob_range_msg);
      log_msg(locl_range_msg);
   }

   GPUManager_t(): m_que(sycl::default_selector_v), global_range({0,0,0}), local_range({0,0,0}),
      HtoD_memcpy_counter(0), DtoH_memcpy_counter(0), DtoD_memcpy_counter(0)
   {
      print_device_name();
      set_device_properties();

      std::string glob_range_msg = "Total number of threads launched for the GPU solver : " + std::to_string(global_range[2]);
      std::string locl_range_msg = "Number of threads launched per workgroup for the GPU solver : " + std::to_string(local_range[2]);
      log_msg(glob_range_msg);
      log_msg(locl_range_msg);
   }

private:

   template <typename ... Types>
   void transfer_to_gpu_move_internal(const Types& ... dss_objs);

   template <typename ... Types>
   void transfer_to_gpu_readonly_internal(const Types& ... dss_objs);

   template <typename ... Types>
   void transfer_to_gpu_copy_internal(const Types& ... dss_objs);

   template <typename ... Types>
   void transfer_to_gpu_noinit_internal(const Types& ... dss_objs);

   template <typename ... Types>
   void transfer_to_gpu_syncandmove_internal(const Types& ... dss_objs);

   template <typename ... Types>
   void transfer_to_cpu_move_internal(Types& ... dss_objs);

   template <typename ... Types>
   void transfer_to_cpu_readonly_internal(Types& ... dss_objs);

   template <typename ... Types>
   void transfer_to_cpu_copy_internal(Types& ... dss_objs);

   template <typename ... Types>
   void transfer_to_cpu_noinit_internal(Types& ... dss_objs);

   template <typename ... Types>
   void transfer_to_cpu_syncandmove_internal(Types& ... dss_objs);

   template <class  T>
   T&& extract_gpu_data_for_kernel(T&& m_data);

   template <class  T,  CDF::StorageType  TYPE,  uint8_t DIMS = ZEROD>
   dataSetStorageGPURead<T, TYPE, DIMS>& extract_gpu_data_for_kernel(dataSetStorageRead<T, TYPE, DIMS>& dss_obj);

   template <class  T,  CDF::StorageType  TYPE,  uint8_t DIMS = ZEROD>
   dataSetStorageGPU<T, TYPE, DIMS>& extract_gpu_data_for_kernel(dataSetStorage<T, TYPE, DIMS>& dss_obj);

   template <typename T, typename... Us>
   void callExtractorIfExists(Us&&... args);

   template <class T>
   T&& extract_gpu_data_for_extractor(T&& m_data);

   template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
   dataSetStorageGPU<T, TYPE, DIMS>& extract_gpu_data_for_extractor(dataSetStorage<T, TYPE, DIMS>& dss_obj);

   template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
   dataSetStorageGPURead<T, TYPE, DIMS>& extract_gpu_data_for_extractor(dataSetStorageRead<T, TYPE, DIMS>& dss_obj);

   template<typename T, typename... Us>
   void submit_to_gpu_internal(Us&&... args);

   template<typename T, typename... Us>
   void submit_to_gpu_single_workgroup_internal(Us&&... args);

   template<typename T, typename... Us>
   void single_task_gpu_internal(Us&&... args);

   template <class  T,  CDF::StorageType  TYPE,  uint8_t DIMS = ZEROD>
   bool check_gpu_offsets_internal(const dataSetStorage<T, TYPE, DIMS>& dss_obj);

   inline void print_device_name()
   {
      std::string message = "Device Set || Name: " + m_que.get_device().get_info<sycl::info::device::name>();
      log_msg(message);
   }

   void set_global_local_range_internal(const uint64_t (&glob_range)[1], const uint64_t (&locl_range)[1])
   {
      assert(glob_range[0] % locl_range[0] == 0);
      global_range = {1, 1, glob_range[0]};
      local_range = {1, 1, locl_range[0]};
      std::string message = "Note that you are setting a 1D global range as {1,1," + std::to_string(glob_range[0]) +
                            "} and 1D local range as {1,1," + std::to_string(locl_range[0]) + "}";
      log_msg(message);
   }

   void set_global_local_range_internal(const uint64_t (&glob_range)[2], const uint64_t (&locl_range)[2])
   {
      // We currently only support 1D grids and blocks
      assert(glob_range[0] == 1);
      assert(locl_range[0] == 1);
      assert(glob_range[1] % locl_range[1] == 0);

      global_range = {1, glob_range[0], glob_range[1]};
      local_range = {1, locl_range[0], locl_range[1]};
      std::string message = "Note that you are setting a 2D global range as {1," + std::to_string(glob_range[0]) + "," + std::to_string(glob_range[1]) +
                            "} and 2D local range as {1," + std::to_string(locl_range[0]) + "," + std::to_string(locl_range[1]) + "}";
      log_msg(message);
   }

   void set_global_local_range_internal(const uint64_t (&glob_range)[3], const uint64_t (&locl_range)[3])
   {
      // We currently only support 1D grids and blocks
      assert(glob_range[0] == 1);
      assert(glob_range[1] == 1);
      assert(locl_range[0] == 1);
      assert(locl_range[1] == 1);
      assert(glob_range[2] % locl_range[2] == 0);

      global_range = {glob_range[0], glob_range[1], glob_range[2]};
      local_range = {locl_range[0], locl_range[1], locl_range[2]};
      std::string message = "Note that you are setting a 3D global range as {" + std::to_string(glob_range[0]) + "," + std::to_string(glob_range[1]) + "," + std::to_string(glob_range[2]) +
                            "} and 3D local range as {" + std::to_string(locl_range[0]) + "," + std::to_string(locl_range[1]) + "," + std::to_string(locl_range[2]) + "}";
      log_msg(message);
   }

   // Allocates a gpu variable with given number of elements either in shared or device mode
   template <class T, bool is_malloc_shared = false>
   T* malloc_gpu_var_internal(const uint64_t& num_elements);

   template <class T, bool is_malloc_shared>
   T* malloc_gpu_var_impl (const uint64_t& num_elements);

   // Copies the data from src to dest
   template <class T>
   void memcpy_gpu_var_internal(T* dest, const T * const src, const uint64_t& num_elements);

   // Free the gpu variable passed
   template <class T>
   void free_gpu_var_internal(T* gpu_var);

   template <class  T,  CDF::StorageType  TYPE,  uint8_t DIMS = ZEROD>
   void update_gpu_offsets_internal(const dataSetStorage<T, TYPE, DIMS>& dss_obj);

   template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
   void allocate_gpu_instance(const dataSetStorage<T, TYPE, DIMS>& dss_obj);

   template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
   void deallocate_gpu_instance(const dataSetStorage<T, TYPE, DIMS>& dss_obj);

   template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
   GPUInstance_t& setup_gpu_instance_and_data_ptr(const dataSetStorage<T, TYPE, DIMS>& dss_obj);

   void allocate_gpu_data_ptr(GPUInstance_t* cur_gpu_instance, const bool set_offsets);

   void deallocate_gpu_data_ptr(GPUInstance_t* cur_gpu_instance);

   void resize_gpu_data_ptr(const dataSetBase* const dsb_obj);

   void transfer_to_gpu_internal(const dataSetBase* const dsb_entry, transfer_mode_t transfer_mode = transfer_mode_t::NOT_SET);

   template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
   void transfer_to_gpu_internal(const dataSetStorage<T, TYPE, DIMS>& dss_obj,
                                 transfer_mode_t transfer_mode = transfer_mode_t::NOT_SET);

   template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
   void transfer_to_cpu_internal(const dataSetStorage<T, TYPE, DIMS>& dss_obj,
                                 transfer_mode_t transfer_mode = transfer_mode_t::MOVE);

   void transfer_to_cpu_internal(const dataSetBase* const dsb_entry, transfer_mode_t transfer_mode = transfer_mode_t::MOVE);

   sycl::queue& get_queue_internal() // Needed for use in dataSetBaseGPU.h
   {
      return m_que;
   }

   sycl_device_vendor_t get_device_vendor_t_internal()
   {
      return m_device_vendor;
   }

   void set_device_properties();

   void print_gpu_memcpy_counts_internal();

   sycl::queue m_que;
   sycl::range<3> global_range;
   sycl::range<3> local_range;

   // Device Properties
   std::string m_device_name;
   sycl_device_vendor_t m_device_vendor;
   sycl_device_t m_device_type;

public:
   uint64_t HtoD_memcpy_counter;
   uint64_t DtoH_memcpy_counter;
   uint64_t DtoD_memcpy_counter;

   // The friends functions are the APIs by which the devs interacts with the (private) member functions of gpu_manager_t class

   friend inline void set_gpu_global_local_range(const uint64_t (&glob_range)[1], const uint64_t (&locl_range)[1]);
   friend inline void set_gpu_global_local_range(const uint64_t (&glob_range)[2], const uint64_t (&locl_range)[2]);
   friend inline void set_gpu_global_local_range(const uint64_t (&glob_range)[3], const uint64_t (&locl_range)[3]);

   friend void transfer_to_cpu(const dataSetBase* const dsb_entry, transfer_mode_t transfer_mode);
   template <typename ... Types>
   friend void transfer_to_gpu_move(const Types& ... dss_objs);
   template <typename ... Types>
   friend void transfer_to_gpu_readonly(const Types& ... dss_objs);
   template <typename ... Types>
   friend void transfer_to_gpu_copy(const Types& ... dss_objs);
   template <typename ... Types>
   friend void transfer_to_gpu_noinit(const Types& ... dss_objs);
   template <typename ... Types>
   friend void transfer_to_gpu_syncandmove(const Types& ... dss_objs);

   friend void transfer_to_gpu(const dataSetBase* const dsb_entry, transfer_mode_t transfer_mode);
   template <typename ... Types>
   friend void transfer_to_cpu_move(Types& ... dss_objs);
   template <typename ... Types>
   friend void transfer_to_cpu_readonly(Types& ... dss_objs);
   template <typename ... Types>
   friend void transfer_to_cpu_copy(Types& ... dss_objs);
   template <typename ... Types>
   friend void transfer_to_cpu_noinit(Types& ... dss_objs);
   template <typename ... Types>
   friend void transfer_to_cpu_syncandmove(Types& ... dss_objs);

   template<typename T, typename... Us>
   friend void submit_to_gpu(Us&&... args);

   template<typename T, typename... Us>
   friend void submit_to_gpu_extractor(Us&&... args);

   template<typename T, typename... Us>
   friend void submit_to_gpu_single_workgroup(Us&&... args);

   template<typename T, typename... Us>
   friend void single_task_gpu(Us&&... args);

   friend inline sycl::queue& get_gpu_queue();

   friend inline sycl_device_vendor_t get_device_vendor_t();

   friend inline void print_gpu_memcpy_counts();

   template <class T, bool is_malloc_shared>
   friend T* malloc_gpu_var(const uint64_t& num_elements);
   template <class T>
   friend void memcpy_gpu_var(T* dest, const T * const src, const uint64_t& num_elements);
   template <class T>
   friend void free_gpu_var(T* gpu_var);
};

}  // namespace GDF

#include "gpu_manager_t.hpp"

#endif // GPU_MANAGER_T_H
