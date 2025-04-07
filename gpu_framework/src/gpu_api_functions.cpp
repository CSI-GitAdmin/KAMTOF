#include "gpu_api_functions.h"

namespace GDF
{

void transfer_to_gpu(const dataSetBase* const dsb_entry, transfer_mode_t transfer_mode)
{
   gpu_manager->transfer_to_gpu_internal(dsb_entry, transfer_mode);
}

void transfer_to_cpu(const dataSetBase* const dsb_entry, transfer_mode_t transfer_mode)
{
   gpu_manager->transfer_to_cpu_internal(dsb_entry, transfer_mode);
}

void transfer_all_silo_vars_to_cpu(const int stream_index, const GDF::transfer_mode_t transfer_mode)
{
   // Lambda to bring back the SILO var
   // auto m_transfer_to_cpu = [=](dataSetBase *entry) { GDF::transfer_to_cpu(entry, transfer_mode);};

   //    // Loop through all the storage types and bring back the variables
}

void copy_all_silo_vars_to_cpu()
{
   log_msg<CDF::LogLevel::WARNING>("COPIED ALL SILO VARIBALES TO CPU");
   GDF::transfer_all_silo_vars_to_cpu(0, GDF::transfer_mode_t::COPY); // FIXME
}

void move_all_silo_vars_to_cpu()
{
   log_msg<CDF::LogLevel::WARNING>("MOVED ALL SILO VARIBALES TO CPU");
   GDF::transfer_all_silo_vars_to_cpu(0, GDF::transfer_mode_t::MOVE); // FIXME
}

} // namespace GDF