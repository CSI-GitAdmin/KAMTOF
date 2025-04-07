#ifndef GPU_INSTANCE_T_H
#define GPU_INSTANCE_T_H

#include "gpu_enums.h" // For gpu enums xpu_data_status_t, transfer_mode_t and xpu_t
#include "gpu_silo_fwd.h" // For DSBGPU
#include "silo_fwd.h" // For DSB

namespace  GDF
{

class GPUInstance_t
{
public:
   GPUInstance_t(const dataSetBase* cpu_dsb);

   ~GPUInstance_t();

   const dataSetBaseGPU* get_gpu_dsb_ptr() const;

   dataSetBaseGPU* get_gpu_dsb_ptr();

   void set_gpu_dsb_ptr(dataSetBaseGPU* other);

   const xpu_data_status_t&  get_xpu_data_status(const xpu_t& device_type) const;

   void set_xpu_data_status(const xpu_t& device_type, const xpu_data_status_t& xpu_data_status);

   const transfer_mode_t& get_kernel_transfer_mode();

   void set_kernel_transfer_mode(const transfer_mode_t&);

   void invalidate_cpu_data_ptr(); // Hide (if not hidden) the CPU data pointer in m_temp_cpu_data

   void validate_cpu_data_ptr(); // Un-hide (if hidden) the CPU data pointer in m_temp_cpu_data

   void invalidate_gpu_data_ptr(); // Hide (if not hidden) the GPU data pointer in m_temp_cpu_data

   void validate_gpu_data_ptr(); // Un-hide (if hidden) the GPU data pointer in m_temp_cpu_data

   const void* get_valid_cpu_data() const; // Return the non-null (or DSB m_data is both are null) CPU data pointer

   void* get_valid_cpu_data(); // Return the non-null (or DSB m_data is both are null) CPU data pointer

   const void* get_valid_gpu_data() const; // Return the non-null (or m_gpu_data is both are null) GPU data pointer

   void* get_valid_gpu_data(); // Return the non-null (or m_gpu_data is both are null) GPU data pointer

   const dataSetBase* const get_cpu_dsb_ptr() const; // Return the CPU DSB associated with this GPU instance

private:
   // Stores a pointer of type DSSGPU in DSBGPU as the object of this class is stored inside of StorageInfo which cannot be templated
   dataSetBaseGPU* m_gpu_dss;
   transfer_mode_t kernel_transfer_mode = transfer_mode_t::NOT_SET;
   xpu_data_status_t cpu_data_status = xpu_data_status_t::UP_TO_DATE_WRITE;
   xpu_data_status_t gpu_data_status = xpu_data_status_t::NOT_ALLOCATED;
   void* m_temp_cpu_data = nullptr;
   void* m_temp_gpu_data = nullptr;
   dataSetBase* m_cpu_dsb = nullptr;
};

} // namespace GDF

#endif // GPU_INSTANCE_T_H