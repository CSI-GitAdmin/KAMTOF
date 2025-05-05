#include "gpu_instance_t.h" // For GPUInstance_t member functions
#include "gpu_api_functions.h" // For GPU API functions
#include "datasetbase.h"
#include "pagefault_handler.h"

void dataSetBase::deallocate_gpu_data_ptr()
{
   // The following asserts should not be null when this function is called
   assert(gpu_instance);
   assert(gpu_instance->get_gpu_dsb_ptr());

   // If the data is not allocated, we don't need to free the data pointer
   if((gpu_instance->get_xpu_data_status(GDF::xpu_t::CPU) == GDF::xpu_data_status_t::NOT_ALLOCATED))
   {
      assert(!gpu_instance->get_gpu_dsb_ptr()->offsets());
      assert(!gpu_instance->get_gpu_dsb_ptr()->void_data());
      return;
   }
   if(m_size == 0)
   {
      assert(gpu_instance->get_gpu_dsb_ptr()->void_data() == nullptr);
      gpu_instance->get_gpu_dsb_ptr()->free_offsets();
   }
   else
   {
      // Validate the GPU data pointer
      gpu_instance->validate_gpu_data_ptr();

      GDF::free_gpu_var(gpu_instance->get_gpu_dsb_ptr()->void_data());
      gpu_instance->get_gpu_dsb_ptr()->set_data(nullptr);
      gpu_instance->get_gpu_dsb_ptr()->free_offsets();
   }

   gpu_instance->set_xpu_data_status(GDF::xpu_t::GPU, GDF::xpu_data_status_t::NOT_ALLOCATED);

   if(gpu_instance->get_xpu_data_status(GDF::xpu_t::CPU) != GDF::xpu_data_status_t::NOT_ALLOCATED)
      gpu_instance->set_xpu_data_status(GDF::xpu_t::CPU, GDF::xpu_data_status_t::UP_TO_DATE_WRITE);
}

void dataSetBase::deallocate_gpu_instance()
{
   assert(gpu_instance); // Should not be NULL
   assert(gpu_instance->get_gpu_dsb_ptr()); // Should not be NULL
   assert(!gpu_instance->get_gpu_dsb_ptr()->void_data()); // Should be NULL

   delete gpu_instance->get_gpu_dsb_ptr();
   gpu_instance->set_gpu_dsb_ptr(nullptr);

   delete gpu_instance;
   gpu_instance = nullptr;
}

void dataSetBase::destruct_gpu_instance()
{
   gpu_instance->validate_cpu_data_ptr(); // CPU data should be validated at this point
   deallocate_gpu_data_ptr(); // Deallocate m_gpu_data
   deallocate_gpu_instance(); // Deallocate m_gpu_dss
}

void dataSetBase::set_gpu_data_status_to_resized()
{
   if(gpu_instance) // Only do this if the GPUInstance_t object already exists
   {
      gpu_instance->set_xpu_data_status(GDF::xpu_t::GPU, GDF::xpu_data_status_t::RESIZED_ON_CPU);
      gpu_instance->set_xpu_data_status(GDF::xpu_t::CPU, GDF::xpu_data_status_t::UP_TO_DATE_WRITE);
   }
}

void* dataSetBase::get_gpu_void_data()
{
   assert(gpu_instance);
   assert(gpu_instance->get_gpu_dsb_ptr());
   return gpu_instance->get_gpu_dsb_ptr()->void_data();
}

const void* dataSetBase::get_gpu_void_data() const
{
   assert(gpu_instance);
   assert(gpu_instance->get_gpu_dsb_ptr());
   return gpu_instance->get_gpu_dsb_ptr()->void_data();
}

const GDF::xpu_data_status_t& dataSetBase::get_xpu_data_status(const GDF::xpu_t& device_type) const
{
   assert(gpu_instance);
   return gpu_instance->get_xpu_data_status(device_type);
}

void dataSetBase::transfer_to_cpu(bool read_only) const
{
   if(read_only)
   {
      GDF::transfer_to_cpu(this, GDF::transfer_mode_t::READ_ONLY);
   }
   else
   {
      GDF::transfer_to_cpu(this, GDF::transfer_mode_t::MOVE);
   }
}

#ifndef NDEBUG
void dataSetBase::assert_cpu_data_writeability()
{
   if(!gpu_instance)
      return;
   GDF::xpu_data_status_t cpu_data_status = gpu_instance->get_xpu_data_status(GDF::xpu_t::CPU);
   assert(cpu_data_status != GDF::xpu_data_status_t::OUT_OF_DATE);
   assert(cpu_data_status != GDF::xpu_data_status_t::RESIZED_ON_CPU);
   assert(cpu_data_status != GDF::xpu_data_status_t::UP_TO_DATE_READ);
}
#endif

// NOTE: The actual allocation size is greater than (or) equal to "byte_size" for ENABLE_GPU=ON builds as it must be a multiple of sytstem page size
void dataSetBase::allocate_page_aligned_memory_internal(const uint64_t byte_size)
{
   std::pair<void*, uint64_t> allocated_data = allocate_page_aligned_memory(byte_size);
   m_data = allocated_data.first;
   allocation_size = allocated_data.second;

#ifndef NDEBUG // Check if the dsb pointer already exist
   for(std::set<std::pair<void*, dataSetBase*>>::iterator it = dsb_addr_set.begin(); it != dsb_addr_set.end(); it++)
   {
      if(it->second == this)
      {
         log_error("Trying to add the same DSB pointer " + m_name +" multiple times in the dsb_addr_set!");
      }
   }
#endif

   if(!dsb_addr_set.insert(std::make_pair(m_data, this)).second) // Add this variable and it's m_data to the dsb_addr_set
   {
      log_msg<CDF::LogLevel::ERROR>(std::string("Duplicate insert in dsb_addr_set for variable ") + m_name);
   }
}

void dataSetBase::deallocate_page_aligned_memory_internal()
{
   assert(m_data);
   deallocate_page_aligned_memory(m_data, allocation_size);
   allocation_size = 0;
   if(dsb_addr_set.erase(std::make_pair(m_data, this)) != 1) // Erase this variable and it's m_data to the dsb_addr_set
   {
      log_msg<CDF::LogLevel::ERROR>(std::string("m_data not found in dsb_addr_set for variable ") + m_name);
   }
}

void dataSetBase::copy_over_and_resize_page_aligned_memory(const uint64_t new_byte_size)
{
   assert(m_size != 0 && m_byte_size != 0);
   // Allocate new data and copy over data
   std::pair<void*, uint64_t> allocated_data = allocate_page_aligned_memory(new_byte_size);
   void* new_data = allocated_data.first;
   uint64_t new_allocation_size = allocated_data.second;

   uint64_t copy_byte_size = (new_byte_size <= m_byte_size) ? new_byte_size : m_byte_size;
   memcpy(new_data, m_data, copy_byte_size);

   // Pointer Swap
   delete_m_data();
   // m_size and m_byte_size will be set outside the scope in which this function is invoked
   m_data = new_data;
   allocation_size = new_allocation_size;
   if(!dsb_addr_set.insert(std::make_pair(m_data, this)).second) // Add this variable and it's m_data to the dsb_addr_set
   {
      log_msg<CDF::LogLevel::ERROR>(std::string("Duplicate insert in dsb_addr_set for variable ") + m_name);
   }
}
