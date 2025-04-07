#include "gpu_globals.h" // For GPU Globals
#include "gpu_api_functions.h" // For GPU API functions

GDF::sycl_device_t get_device_type_from_string(const std::string& device_type)
{
   if(device_type == "DEFAULT")
      return GDF::sycl_device_t::DEFAULT;
   else if(device_type == "CPU")
      return GDF::sycl_device_t::CPU;
   else if(device_type == "GPU")
      return GDF::sycl_device_t::GPU;
   else if(device_type == "ACCELERATOR")
   {
      log_msg<CDF::LogLevel::WARNING>("The compute device selected is ACCELERATOR");
      return GDF::sycl_device_t::ACCELERATOR;
   }
   else
   {
      std::string err_msg = "Unknown compute device type : " + device_type;
      log_msg<CDF::LogLevel::ERROR>(err_msg);
      return GDF::sycl_device_t::DEFAULT;
   }
}

// Allocate and initialize GPU variables
void setup_gpu_globals()
{
   #ifdef GPU_MEM_LOG
      tot_gpu_mem_used = 0;
      gpu_mem_usage_log.open("gpu_mem_usage.log", std::ios_base::trunc | std::ios_base::out);
   #endif

   // Setup GPU Manager (Has no variable which needs to be accesed on GPU)
   assert(!gpu_manager);
   gpu_manager = new GDF::GPUManager_t(GDF::sycl_device_t::GPU,
                                       {1, 1, 1024 },
                                       {1, 1,  256 } );

   log_msg("Device type selected : DEFAULT");
}

void finalize_gpu_globals()
{
   assert(gpu_manager);
   GDF::gpu_barrier(); // Wait for all the GPU related processes to end

   // Finalize GPU Manager
   delete gpu_manager;
   gpu_manager = nullptr;

   #ifdef GPU_MEM_LOG
      gpu_mem_usage_log.close();
   #endif
}
