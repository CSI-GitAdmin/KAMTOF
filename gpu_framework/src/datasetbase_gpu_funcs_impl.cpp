#include "gpu_instance_t.h" // For GPUInstance_t member functions
#include "gpu_api_functions.h" // For GPU API functions
#include "datasetbase.h"

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

   // Validate the GPU data pointer
   gpu_instance->validate_gpu_data_ptr();

   assert(gpu_instance->get_gpu_dsb_ptr()->void_data());

   GDF::free_gpu_var(gpu_instance->get_gpu_dsb_ptr()->void_data());
   gpu_instance->get_gpu_dsb_ptr()->set_data(nullptr);
   // gpu_instance->get_gpu_dsb_ptr()->free_offsets();

   gpu_instance->set_xpu_data_status(GDF::xpu_t::GPU, GDF::xpu_data_status_t::NOT_ALLOCATED);

   if(gpu_instance->get_xpu_data_status(GDF::xpu_t::CPU) != GDF::xpu_data_status_t::NOT_ALLOCATED)
      gpu_instance->set_xpu_data_status(GDF::xpu_t::CPU, GDF::xpu_data_status_t::UP_TO_DATE_WRITE);
}

void dataSetBase::deallocate_gpu_instance()
{
   assert(gpu_instance); // Should not be NULL
   assert(gpu_instance->get_gpu_dsb_ptr()); // Should not be NULL
   assert(!gpu_instance->get_gpu_dsb_ptr()->void_data()); // Should be NULL

   GDF::free_gpu_var(gpu_instance->get_gpu_dsb_ptr());
   gpu_instance->set_gpu_dsb_ptr(nullptr);

   delete gpu_instance;
   gpu_instance = nullptr;
}

void dataSetBase::destruct_gpu_instance()
{
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

#ifdef GPU_DEVELOP
void dataSetBase::transfer_to_cpu(bool read_only) const
{
   if(!read_only)
   {
      GDF::transfer_to_cpu(this, GDF::transfer_mode_t::MOVE);
   }
   else
   {
      GDF::transfer_to_cpu(this, GDF::transfer_mode_t::READ_ONLY);
   }
}
#endif

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

void dataSetBase::validate_cpu_data_ptr()
{
   assert(gpu_instance);
   gpu_instance->validate_cpu_data_ptr();
}
