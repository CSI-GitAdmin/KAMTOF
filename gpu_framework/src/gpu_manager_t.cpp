#include "gpu_manager_t.h"
#include <cassert>
#include <algorithm>
#include <sys/mman.h>

namespace GDF
{

#ifdef GPU_DEVELOP
void protect_m_data(const dataSetBase* const dsb_entry, GPUInstance_t& cur_gpu_instance, const xpu_data_status_t& cpu_data_status)
{
   /*
      // Validate/Invalidate the CPU/GPU data pointers based on the stauses
      if(cpu_data_status == xpu_data_status_t::OUT_OF_DATE)
         cur_gpu_instance.invalidate_cpu_data_ptr();
      else
         cur_gpu_instance.validate_cpu_data_ptr();

      if((gpu_data_status == xpu_data_status_t::OUT_OF_DATE) || (gpu_data_status == xpu_data_status_t::RESIZED_ON_CPU))
         cur_gpu_instance.invalidate_gpu_data_ptr();
      else
         cur_gpu_instance.validate_gpu_data_ptr();
   */

   if(cpu_data_status == xpu_data_status_t::OUT_OF_DATE)
   {
      // Remove both read and write access to the data pointer
      if(mprotect(cur_gpu_instance.get_valid_cpu_data(), dsb_entry->get_allocation_size(), PROT_NONE) == -1)
      {
         log_msg<CDF::LogLevel::ERROR>(std::string("Failure while protecting data in transfer_to_gpu for variable: ") + dsb_entry->name());
      }
   }
   else if(cpu_data_status == xpu_data_status_t::UP_TO_DATE_READ)
   {
      // Remove write access to the data pointer
      if(mprotect(cur_gpu_instance.get_valid_cpu_data(), dsb_entry->get_allocation_size(), PROT_READ) == -1)
      {
         log_msg<CDF::LogLevel::ERROR>(std::string("Failure while protecting data in transfer_to_gpu for variable: ") + dsb_entry->name());
      }
   }
   else if(cpu_data_status == xpu_data_status_t::UP_TO_DATE_WRITE || cpu_data_status == xpu_data_status_t::TEMP_WRITE)
   {
      // Enable both read and write access to the data pointer
      if(mprotect(cur_gpu_instance.get_valid_cpu_data(), dsb_entry->get_allocation_size(), PROT_READ | PROT_WRITE) == -1)
      {
         log_msg<CDF::LogLevel::ERROR>(std::string("Failure while protecting data in transfer_to_gpu for variable: ") + dsb_entry->name());
      }
   }
}
#endif

void GPUManager_t::set_device_properties()
{
   // Set the device name
   m_device_name = m_que.get_device().get_info<sycl::info::device::name>();

   // Set the device vendor
   std::string device_vendor(m_que.get_device().get_info<sycl::info::device::vendor>());
   std::transform(device_vendor.begin(), device_vendor.end(), device_vendor.begin(), ::toupper);
   if(device_vendor.find("AMD") != std::string::npos)
   {
      m_device_vendor = sycl_device_vendor_t::AMD;
   }
   else if(device_vendor.find("INTEL") != std::string::npos)
   {
      m_device_vendor = sycl_device_vendor_t::INTEL;
   }
   else if(device_vendor.find("NVIDIA") != std::string::npos)
   {
      m_device_vendor = sycl_device_vendor_t::NVIDIA;
   }
   else
   {
      std::string err_msg = "Error while selecting compute device: Compute device vendor " + device_vendor +" is not supported";
      log_msg<CDF::LogLevel::ERROR>(err_msg);
   }

   // Set the sycl device type
   sycl::info::device_type device_type = m_que.get_device().get_info<sycl::info::device::device_type>();
   switch(device_type)
   {
   case sycl::info::device_type::cpu:
   {
      m_device_type = sycl_device_t::CPU;
      break;
   }
   case sycl::info::device_type::gpu:
   {
      m_device_type = sycl_device_t::GPU;
      break;
   }
   case sycl::info::device_type::accelerator:
   {
      m_device_type = sycl_device_t::ACCELERATOR;
      break;
   }
   default:
   {
      std::string err_msg = "Error while selecting compute device: Compute device type is not supported, "
                            "Please select set compute_device_type to CPU (or) GPU in preliminary.in";
      log_msg<CDF::LogLevel::ERROR>(err_msg);
   }
   }
}

void GPUManager_t::transfer_to_gpu_internal(const dataSetBase* const dsb_entry, transfer_mode_t transfer_mode /* = transfer_mode_t::NOT_SET */)
{
   assert(dsb_entry->get_gpu_instance()); // GPUInstance should be setup before coming here

   GPUInstance_t& cur_gpu_instance = *(const_cast<GPUInstance_t*>(dsb_entry->get_gpu_instance()));

   int8_t do_memcpy = 0; // 0 -> no memcpy ; 1 -> memcpy cpu to gpu ; -1 -> memcpy gpu to cpu

   const transfer_mode_t& kernel_transfer_mode = cur_gpu_instance.get_kernel_transfer_mode();
   if(transfer_mode == transfer_mode_t::NOT_SET)
   {
      assert(kernel_transfer_mode != transfer_mode_t::NOT_SET);
      transfer_mode = kernel_transfer_mode;
   }
#ifndef NDEBUG
   else
   {
      // If the kernel_transfer_mode was passed in, the kernel_transfer_mode inside the DSS object should be NONE
      assert(kernel_transfer_mode == transfer_mode_t::NOT_SET);
   }
#endif

   const xpu_data_status_t& cpu_data_status = cur_gpu_instance.get_xpu_data_status(xpu_t::CPU);
   const xpu_data_status_t& gpu_data_status = cur_gpu_instance.get_xpu_data_status(xpu_t::GPU);

   assert(cpu_data_status != xpu_data_status_t::RESIZED_ON_CPU);

   switch(gpu_data_status)
   {
   // The data is already up to date on the GPU and hence does not require a memcpy
   case xpu_data_status_t::UP_TO_DATE_WRITE:
   {
      assert((cpu_data_status == xpu_data_status_t::OUT_OF_DATE) || (cpu_data_status == xpu_data_status_t::TEMP_WRITE));
      if(transfer_mode == transfer_mode_t::SYNC_AND_MOVE)
         do_memcpy = -1;
      break;
   }

      // The data is already up to date on GPU and now only neeeds to change permission if needed and doesn't require a memcpy
   case xpu_data_status_t::UP_TO_DATE_READ:
   {
      assert((cpu_data_status == xpu_data_status_t::OUT_OF_DATE) || (cpu_data_status == xpu_data_status_t::TEMP_WRITE) ||
             (cpu_data_status == xpu_data_status_t::UP_TO_DATE_READ));
      if(transfer_mode == transfer_mode_t::SYNC_AND_MOVE && cpu_data_status != xpu_data_status_t::UP_TO_DATE_READ)
         do_memcpy = -1;
      break;
   }

   case xpu_data_status_t::OUT_OF_DATE:
   {
      assert((cpu_data_status == xpu_data_status_t::UP_TO_DATE_WRITE) || (cpu_data_status == xpu_data_status_t::UP_TO_DATE_READ));
      if(transfer_mode != transfer_mode_t::NOT_INITIALIZE)
         do_memcpy = 1;
      break;
   }

   case xpu_data_status_t::TEMP_WRITE:
   {
      assert(cpu_data_status == xpu_data_status_t::UP_TO_DATE_WRITE || cpu_data_status == xpu_data_status_t::UP_TO_DATE_READ);
      if(transfer_mode != transfer_mode_t::NOT_INITIALIZE)
         do_memcpy = 1;
      break;
   }

   case xpu_data_status_t::RESIZED_ON_CPU:
   {
      assert((cpu_data_status ==  xpu_data_status_t::UP_TO_DATE_WRITE) || (cpu_data_status ==  xpu_data_status_t::UP_TO_DATE_READ));
      resize_gpu_data_ptr(dsb_entry);
      if(transfer_mode != transfer_mode_t::NOT_INITIALIZE)
         do_memcpy = true;
      break;
   }

   case xpu_data_status_t::NOT_ALLOCATED: // Handle GPU SILO variables which do not exist
   {
      assert(cpu_data_status == xpu_data_status_t::NOT_ALLOCATED);
      break;
   }

   default:
   {
      std::string error = "Error in transfer_to_gpu for SILO variable " + static_cast<std::string>(dsb_entry->name()) + " :  xpu_data_status_t is set to unkown value."
                                                                                                                        " Please check if a swicth case is added for the current"
                                                                                                                        " xpu_data_status_t ";
      log_msg<CDF::LogLevel::ERROR>(error);
      break;
   }
   }

   if(gpu_data_status != xpu_data_status_t::NOT_ALLOCATED) // For SILO variables that doesn't exist, skip memcpy and status change
   {
      if(do_memcpy)
      {
         // do_memcpy should be -1 only if trasnfer mode is SYNC_AND_MOVE
         assert((do_memcpy == -1 && transfer_mode == transfer_mode_t::SYNC_AND_MOVE) || do_memcpy == 1);

         // Memcpy can be done even to the temp data
         void* cpu_m_data = cur_gpu_instance.get_valid_cpu_data();
         void* gpu_m_data = cur_gpu_instance.get_valid_gpu_data();

         const void* const src_data_ptr = (do_memcpy == 1) ? cpu_m_data : gpu_m_data;
         void* const dest_data_ptr = (do_memcpy == 1) ? gpu_m_data : cpu_m_data;

#ifndef NDEBUG
         // Sanity check to make sure src != dest and size is the same
         assert(cpu_m_data && gpu_m_data && (cpu_m_data != gpu_m_data));
#endif
         // Do the actual SYCL memcpy
         // GPU_TODO : Add MPI_SAFE_CALL equivalent for SYCL
         m_que.memcpy(dest_data_ptr, src_data_ptr, dsb_entry->byte_size()).wait();
      }

      switch(transfer_mode)
      {
      case transfer_mode_t::MOVE: // Transfer primary ownership from CPU to GPU
      {
         // Set to OUT_OF_DATE is cpu_data_status is UP_TO_DATE_READ (or) UP_TO_DATE_WRITE
         if(cpu_data_status != xpu_data_status_t::TEMP_WRITE)
            cur_gpu_instance.set_xpu_data_status(xpu_t::CPU, xpu_data_status_t::OUT_OF_DATE);
         cur_gpu_instance.set_xpu_data_status(xpu_t::GPU, xpu_data_status_t::UP_TO_DATE_WRITE);
         break;
      }

      case transfer_mode_t::READ_ONLY: // Both devices share the ownership of the data and agree not to write to it
      {
         cur_gpu_instance.set_xpu_data_status(xpu_t::GPU, xpu_data_status_t::UP_TO_DATE_READ);
         if(cpu_data_status == xpu_data_status_t::UP_TO_DATE_WRITE)
            cur_gpu_instance.set_xpu_data_status(xpu_t::CPU, xpu_data_status_t::UP_TO_DATE_READ);
         break;
      }

      case transfer_mode_t::COPY: // Primary ownership retained by CPU but GPU has scratch pad permissions
      {
         assert(cpu_data_status == xpu_data_status_t::UP_TO_DATE_WRITE || cpu_data_status == xpu_data_status_t::UP_TO_DATE_READ);
         cur_gpu_instance.set_xpu_data_status(xpu_t::GPU, xpu_data_status_t::TEMP_WRITE);
         break;
      }

      case transfer_mode_t::NOT_INITIALIZE: // Transfer primary ownership from CPU to GPU without an actual memcpy
      {
         assert(!do_memcpy);
         // Set to OUT_OF_DATE is cpu_data_status is UP_TO_DATE_READ (or) UP_TO_DATE_WRITE
         if(cpu_data_status != xpu_data_status_t::TEMP_WRITE)
            cur_gpu_instance.set_xpu_data_status(xpu_t::CPU, xpu_data_status_t::OUT_OF_DATE);
         cur_gpu_instance.set_xpu_data_status(xpu_t::GPU, xpu_data_status_t::UP_TO_DATE_WRITE);
         break;
      }

      case transfer_mode_t::SYNC_AND_MOVE: // Sync the data between the devices, transfer primary ownership to GPU but CPU retains scratch pad permissions
      {
         cur_gpu_instance.set_xpu_data_status(xpu_t::GPU, xpu_data_status_t::UP_TO_DATE_WRITE);
         cur_gpu_instance.set_xpu_data_status(xpu_t::CPU, xpu_data_status_t::TEMP_WRITE);
         break;
      }

         /*
          * The following two cases can only happen if user-dev explicitly calls transfer_to_gpu with NOT_SET (or)
          * a kernel set_status function missed the a dssgpu variable. Both of these are not intended to happen
         */

      case transfer_mode_t::NOT_SET:
      {
         log_msg<CDF::LogLevel::ERROR>("An internal error occured while moving data from CPU to GPU. The transfer_mode is set to NOT_SET");
      }

      default:
      {
         log_msg<CDF::LogLevel::ERROR>("An internal error occured while moving data from CPU to GPU.");
      }
      }
#ifdef GPU_DEVELOP
      protect_m_data(dsb_entry, cur_gpu_instance, cpu_data_status);
#endif

// Set the read_only flag
#ifndef NDEBUG
      dataSetBaseGPU* gpu_dsb = cur_gpu_instance.get_gpu_dsb_ptr();
      if(gpu_data_status == xpu_data_status_t::UP_TO_DATE_READ)
      {
         gpu_dsb->is_read_only = true;
      }
      else
      {
         gpu_dsb->is_read_only = false;
      }
#endif
   }

   // Reset the kernel_transfer_mode back to NONE
   cur_gpu_instance.set_kernel_transfer_mode(transfer_mode_t::NOT_SET);

   if(do_memcpy == 1)
   {
      HtoD_memcpy_counter++;
#ifndef NDEBUG
      std::string message = static_cast<std::string>(dsb_entry->name()) + " has been transferred to GPU with transfer mode " +
                            GDF::transfer_mode_to_cstr(transfer_mode) + " with memcpy";
      log_msg<CDF::LogLevel::DEBUG>(message);
#endif
   }
   else if(do_memcpy == -1)
   {
      DtoH_memcpy_counter++;
#ifndef NDEBUG
      std::string message = static_cast<std::string>(dsb_entry->name()) + " has been transferred to CPU with transfer mode " +
                            GDF::transfer_mode_to_cstr(transfer_mode) + " with memcpy (transfer_to_gpu)";
      log_msg<CDF::LogLevel::DEBUG>(message);
#endif
   }
   else if(!do_memcpy)
   {
#ifndef NDEBUG
      std::string message = static_cast<std::string>(dsb_entry->name()) + " has been transferred to GPU with transfer mode " +
                            GDF::transfer_mode_to_cstr(transfer_mode) + " without memcpy";
      log_msg<CDF::LogLevel::DEBUG>(message);
#endif
   }
   else
   {
      std::string message = "Error in transfer_to_gpu: do_memcpy must be -1,0 (or) 1";
      log_msg<CDF::LogLevel::DEBUG>(message);
   }
}

void GPUManager_t::transfer_to_cpu_internal(const dataSetBase * const dsb_entry, transfer_mode_t transfer_mode /* = transfer_mode_t::MOVE */)
{
   if(dsb_entry->get_gpu_instance() == nullptr)
   {
      return;
   }

   GPUInstance_t& cur_gpu_instance = *(const_cast<GPUInstance_t*>(dsb_entry->get_gpu_instance()));

   assert(cur_gpu_instance.get_gpu_dsb_ptr());
   assert(cur_gpu_instance.get_valid_cpu_data() || !dsb_entry->exists()); // The data shouldn't be NULL if the dss_object exist
   assert(cur_gpu_instance.get_kernel_transfer_mode() == transfer_mode_t::NOT_SET);

   int8_t do_memcpy = 0; // 0 -> no memcpy ; 1 -> memcpy gpu to cpu ; -1 -> memcpy cpu to gpu

   const xpu_data_status_t& cpu_data_status = cur_gpu_instance.get_xpu_data_status(xpu_t::CPU);
   const xpu_data_status_t& gpu_data_status = cur_gpu_instance.get_xpu_data_status(xpu_t::GPU);

   switch(cpu_data_status)
   {
   case xpu_data_status_t::UP_TO_DATE_WRITE:
   {
      assert((gpu_data_status != xpu_data_status_t::UP_TO_DATE_WRITE) || (gpu_data_status != xpu_data_status_t::UP_TO_DATE_READ));
      if(transfer_mode == transfer_mode_t::SYNC_AND_MOVE)
      {
         do_memcpy = -1;
         if(gpu_data_status == xpu_data_status_t::RESIZED_ON_CPU)
            resize_gpu_data_ptr(dsb_entry);
      }
      break;
   }

   case xpu_data_status_t::UP_TO_DATE_READ:
   {
      assert((gpu_data_status != xpu_data_status_t::UP_TO_DATE_WRITE));
      if(transfer_mode == transfer_mode_t::SYNC_AND_MOVE && gpu_data_status != xpu_data_status_t::UP_TO_DATE_READ)
      {
         do_memcpy = -1;
         if(gpu_data_status == xpu_data_status_t::RESIZED_ON_CPU)
            resize_gpu_data_ptr(dsb_entry);
      }
      break;
   }

   case xpu_data_status_t::OUT_OF_DATE:
   {
      assert((gpu_data_status == xpu_data_status_t::UP_TO_DATE_WRITE) || (gpu_data_status == xpu_data_status_t::UP_TO_DATE_READ));
      if(transfer_mode != transfer_mode_t::NOT_INITIALIZE)
         do_memcpy = 1;
      break;
   }

   case xpu_data_status_t::TEMP_WRITE:
   {
      assert((gpu_data_status == xpu_data_status_t::UP_TO_DATE_WRITE) || (gpu_data_status == xpu_data_status_t::UP_TO_DATE_READ));
      if(transfer_mode != transfer_mode_t::NOT_INITIALIZE)
         do_memcpy = 1;
      break;
   }

   case xpu_data_status_t::RESIZED_ON_CPU:
   {
      std::string error = "Error in transfer_to_cpu for SILO variable " + static_cast<std::string>(dsb_entry->name()) + " : The xpu_data_status_t for CPU is set to RESIZED_ON_CPU"
                                                                                                                        " which cannot happen";
      log_msg<CDF::LogLevel::ERROR>(error);
   }

   case xpu_data_status_t::NOT_ALLOCATED: // Handle GPU SILO variables which do not exist
   {
      assert(gpu_data_status == xpu_data_status_t::NOT_ALLOCATED);
      break;
   }

   default:
   {
      std::string error = "Error in transfer_to_cpu for SILO variable " + static_cast<std::string>(dsb_entry->name()) + " : xpu_data_status_t is set to unkown value. Please check if a swicth case"
                                                                                                                        " is added for the current xpu_data_status_t ";
      log_msg<CDF::LogLevel::ERROR>(error);
   }
   }

   if(cpu_data_status != xpu_data_status_t::NOT_ALLOCATED) // For SILO variables that doesn't exist, skip memcpy and status change
   {
      if(do_memcpy)
      {
         // do_memcpy should be -1 only if trasnfer mode is SYNC_AND_MOVE
         assert((do_memcpy == -1 && transfer_mode == transfer_mode_t::SYNC_AND_MOVE) || do_memcpy == 1);

         // Memcpy can be done even to the temp data
         void* cpu_m_data = cur_gpu_instance.get_valid_cpu_data();
         void* gpu_m_data = cur_gpu_instance.get_valid_gpu_data();

         const void* const src_data_ptr = (do_memcpy == 1) ? gpu_m_data : cpu_m_data;
         void* const dest_data_ptr = (do_memcpy == 1) ? cpu_m_data : gpu_m_data;

#ifndef NDEBUG
         // Sanity check to make sure src != dest and size is the same
         assert(cpu_m_data && gpu_m_data && (cpu_m_data != gpu_m_data));
#endif \
   // Do the actual SYCL memcpy \
   // GPU_TODO : Add MPI_SAFE_CALL equivalent for SYCL
         m_que.memcpy(dest_data_ptr, src_data_ptr, dsb_entry->byte_size()).wait();
      }

      switch(transfer_mode)
      {
      case transfer_mode_t::MOVE:
      {
         if((gpu_data_status == xpu_data_status_t::UP_TO_DATE_READ) || (gpu_data_status == xpu_data_status_t::UP_TO_DATE_WRITE))
            cur_gpu_instance.set_xpu_data_status(xpu_t::GPU, xpu_data_status_t::OUT_OF_DATE);
         cur_gpu_instance.set_xpu_data_status(xpu_t::CPU, xpu_data_status_t::UP_TO_DATE_WRITE);
         break;
      }

      case transfer_mode_t::READ_ONLY:
      {
         cur_gpu_instance.set_xpu_data_status(xpu_t::CPU, xpu_data_status_t::UP_TO_DATE_READ);
         if(gpu_data_status == xpu_data_status_t::UP_TO_DATE_WRITE)
            cur_gpu_instance.set_xpu_data_status(xpu_t::GPU, xpu_data_status_t::UP_TO_DATE_READ);
         break;
      }

      case transfer_mode_t::COPY:
      {
         assert(gpu_data_status == xpu_data_status_t::UP_TO_DATE_WRITE || gpu_data_status == xpu_data_status_t::UP_TO_DATE_READ);
         cur_gpu_instance.set_xpu_data_status(xpu_t::CPU, xpu_data_status_t::TEMP_WRITE);
         break;
      }

      case transfer_mode_t::NOT_INITIALIZE:
      {
         assert(!do_memcpy);
         if((gpu_data_status == xpu_data_status_t::UP_TO_DATE_READ) || (gpu_data_status == xpu_data_status_t::UP_TO_DATE_WRITE))
            cur_gpu_instance.set_xpu_data_status(xpu_t::GPU, xpu_data_status_t::OUT_OF_DATE);
         cur_gpu_instance.set_xpu_data_status(xpu_t::CPU, xpu_data_status_t::UP_TO_DATE_WRITE);
         break;
      }

      case transfer_mode_t::SYNC_AND_MOVE: // Sync the data between the devices, transfer primary ownership to GPU but CPU retains scratch pad permissions
      {
         cur_gpu_instance.set_xpu_data_status(xpu_t::GPU, xpu_data_status_t::UP_TO_DATE_WRITE);
         cur_gpu_instance.set_xpu_data_status(xpu_t::CPU, xpu_data_status_t::TEMP_WRITE);
         break;
      }

         /*
          * The following two cases can only happen if user-dev explicitly calls transfer_to_gpu with NOT_SET (or)
          * a kernel set_status function missed the a dssgpu variable. Both of these are not intended to happen
         */

      case transfer_mode_t::NOT_SET:
      {
         log_msg<CDF::LogLevel::ERROR>("An internal error occured while moving data from GPU to CPU. The transfer_mode is set to NOT_SET");
      }

      default:
      {
         log_msg<CDF::LogLevel::ERROR>("An internal error occured while moving data from GPU to CPU.");
      }
      }

#ifdef GPU_DEVELOP
      protect_m_data(dsb_entry, cur_gpu_instance, cpu_data_status);
#endif

// Set the read_only flag
#ifndef NDEBUG
      dataSetBaseGPU* gpu_dsb = cur_gpu_instance.get_gpu_dsb_ptr();
      if(gpu_data_status == xpu_data_status_t::UP_TO_DATE_READ)
      {
         gpu_dsb->is_read_only = true;
      }
      else
      {
         gpu_dsb->is_read_only = false;
      }
#endif
   }


   if(do_memcpy == 1)
   {
      DtoH_memcpy_counter++;
#ifndef NDEBUG
      std::string message = static_cast<std::string>(dsb_entry->name()) + " has been transferred to CPU with transfer mode " +
                            GDF::transfer_mode_to_cstr(transfer_mode) + " with memcpy";
      log_msg<CDF::LogLevel::DEBUG>(message);
#endif
   }
   else if(do_memcpy == -1)
   {
      HtoD_memcpy_counter++;
#ifndef NDEBUG
      std::string message = static_cast<std::string>(dsb_entry->name()) + " has been transferred to CPU with transfer mode " +
                            GDF::transfer_mode_to_cstr(transfer_mode) + " with memcpy (transfer_to_cpu)";
      log_msg<CDF::LogLevel::DEBUG>(message);
#endif
   }
   else if(!do_memcpy)
   {
#ifndef NDEBUG
      std::string message = static_cast<std::string>(dsb_entry->name()) + " has been transferred to CPU with transfer mode " +
                            GDF::transfer_mode_to_cstr(transfer_mode) + " without memcpy";
      log_msg<CDF::LogLevel::DEBUG>(message);
#endif
   }
   else
   {
      std::string message = "Error in transfer_to_cpu: do_memcpy must be -1,0 (or) 1";
      log_msg<CDF::LogLevel::DEBUG>(message);
   }
}

void GPUManager_t::allocate_gpu_data_ptr(GPUInstance_t* cur_gpu_instance, const bool set_offsets)
{
   assert(cur_gpu_instance); // Should not be null
   dataSetBaseGPU* cur_gpu_dsb_ptr = cur_gpu_instance->get_gpu_dsb_ptr();
   const dataSetBase* const cur_cpu_dsb_ptr = cur_gpu_instance->get_cpu_dsb_ptr();
   assert(cur_gpu_dsb_ptr); // Should not be null

   void* cur_m_gpu_data = cur_gpu_dsb_ptr->void_data(); // Get the m_data from DSB
   assert(!cur_m_gpu_data); // This function should only be called when the actual m_data is null

   // Allocate the actual GPU data using malloc_device
   cur_m_gpu_data = malloc_gpu_var_internal<void>(cur_cpu_dsb_ptr->byte_size());
   cur_gpu_dsb_ptr->set_data(cur_m_gpu_data);
   cur_gpu_dsb_ptr->set_size(cur_cpu_dsb_ptr->size());
   if(set_offsets)
      cur_gpu_dsb_ptr->set_offsets(cur_cpu_dsb_ptr->num_offsets(), cur_cpu_dsb_ptr->offsets());

      cur_gpu_instance->set_xpu_data_status(xpu_t::GPU, xpu_data_status_t::OUT_OF_DATE);
   assert(cur_gpu_instance->get_xpu_data_status(xpu_t::CPU) == xpu_data_status_t::UP_TO_DATE_WRITE);
   assert(cur_gpu_instance->get_kernel_transfer_mode() == transfer_mode_t::NOT_SET);
}

void GPUManager_t::deallocate_gpu_data_ptr(GPUInstance_t* cur_gpu_instance)
{
   // Validate the gpu data ptr so that it can be deleted safely
   cur_gpu_instance->validate_gpu_data_ptr();

   // The following asserts should not be null when this function is called
   assert(cur_gpu_instance);
   assert(cur_gpu_instance->get_gpu_dsb_ptr());
   assert(cur_gpu_instance->get_gpu_dsb_ptr()->void_data());

   free_gpu_var_internal(cur_gpu_instance->get_gpu_dsb_ptr()->void_data());
   cur_gpu_instance->get_gpu_dsb_ptr()->set_data(nullptr);
   // We do not free the offsets here as this function is primarily used for resize where the offsets don't change
   // We free the offsets in storageInfo::deallocate_gpu_data_ptr() when the SILO object is destroyed

   cur_gpu_instance->set_xpu_data_status(xpu_t::GPU, xpu_data_status_t::NOT_ALLOCATED);
   cur_gpu_instance->set_xpu_data_status(xpu_t::CPU, xpu_data_status_t::UP_TO_DATE_WRITE);
}

void GPUManager_t::resize_gpu_data_ptr(const dataSetBase* const dsb_obj)
{
   assert(dsb_obj->exists());
   GPUInstance_t* cur_gpu_instance = const_cast<GPUInstance_t*>(dsb_obj->get_gpu_instance());
   deallocate_gpu_data_ptr(cur_gpu_instance);
   // Allocate takes care of the setting the new size but doesn't modify the offsets
   allocate_gpu_data_ptr(cur_gpu_instance, false);
}

void GPUManager_t::print_gpu_memcpy_counts_internal()
{
   std::string htod_msg = "Total number of HtoD Memcpy = " + std::to_string(HtoD_memcpy_counter);
   std::string dtoh_msg = "Total number of DtoH Memcpy = " + std::to_string(DtoH_memcpy_counter);
   log_msg(htod_msg);
   log_msg(dtoh_msg);
}

} // namespace GDF