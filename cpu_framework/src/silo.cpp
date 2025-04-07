#include "silo.h"
#include <limits>

void silo::clear_entries()
{
   for(uint8_t cur_type = 0; cur_type < static_cast<uint8_t>(CDF::StorageType::NUM_STORAGE_TYPES); cur_type++)
   {
      for(std::unordered_map<std::string, dataSetBase*>::iterator cur_entry = m_data[cur_type].begin(); cur_entry != m_data[cur_type].end(); cur_entry++)
      {
         delete cur_entry->second;
      }
      m_data[cur_type].clear();
   }
   m_data.clear();
}

silo::silo()
{
   m_data.resize(static_cast<uint8_t>(CDF::StorageType::NUM_STORAGE_TYPES));
   m_num_elements.resize(static_cast<uint8_t>(CDF::StorageType::NUM_STORAGE_TYPES), std::numeric_limits<uint64_t>::max());
   m_num_elements[static_cast<uint8_t>(CDF::StorageType::PARAMETER)] = 1;
   m_num_elements[static_cast<uint8_t>(CDF::StorageType::VECTOR)] = 0;
}

silo::~silo()
{
   assert(m_data.empty());
}