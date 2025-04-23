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

void transfer_all_silo_vars_to_cpu(const GDF::transfer_mode_t transfer_mode)
{
   // Lambda to bring back the SILO var
   auto m_transfer_to_cpu = [=](dataSetBase *entry) { GDF::transfer_to_cpu(entry, transfer_mode);};

   m_silo.for_each<CDF::StorageType::CELL>(m_transfer_to_cpu);
   m_silo.for_each<CDF::StorageType::FACE>(m_transfer_to_cpu);
   m_silo.for_each<CDF::StorageType::BOUNDARY>(m_transfer_to_cpu);
   m_silo.for_each<CDF::StorageType::VECTOR>(m_transfer_to_cpu);
   m_silo.for_each<CDF::StorageType::PARAMETER>(m_transfer_to_cpu);
}

void copy_all_silo_vars_to_cpu()
{
   log_msg<CDF::LogLevel::WARNING>("COPIED ALL SILO VARIBALES TO CPU");
   GDF::transfer_all_silo_vars_to_cpu(GDF::transfer_mode_t::COPY);
}

void move_all_silo_vars_to_cpu()
{
   log_msg<CDF::LogLevel::WARNING>("MOVED ALL SILO VARIBALES TO CPU");
   GDF::transfer_all_silo_vars_to_cpu(GDF::transfer_mode_t::MOVE);
}

} // namespace GDF