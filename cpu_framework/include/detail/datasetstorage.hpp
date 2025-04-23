#ifndef DATASETSTORAGE_HPP
#define DATASETSTORAGE_HPP

#include "datasetstorage.h"
#include "extractor.hpp"
#include "logger.hpp"

template <class T, CDF::StorageType TYPE, uint8_t DIMS>
dataSetStorage<T, TYPE, DIMS>::dataSetStorage(const std::string &name, const uint64_t m_num_entries, const uint8_t* const shape, const bool is_unresolved_entry, const bool allocate_mem):
   dataSet<T, DIMS>(name, m_num_entries, TYPE, shape, is_unresolved_entry, allocate_mem)
{}

// Primarily to support local SILO concept, where the user can allocate and deallocate memory in their function scope
template <class T, CDF::StorageType TYPE, uint8_t DIMS>
void dataSetStorage<T, TYPE, DIMS>::allocate_memory()
{
   if(dataSetBase::is_unresolved)
   {
      log_error("Cannot allocate memory for non registered variables!");
   }
   assert(!dataSetBase::m_data);
   dataSetBase::resize_internal(dataSetBase::m_byte_size);
}

// Primarily to support local SILO concept, where the user can allocate and deallocate memory in their function scope
template <class T, CDF::StorageType TYPE, uint8_t DIMS>
void dataSetStorage<T, TYPE, DIMS>::deallocate_memory()
{
   if(dataSetBase::is_unresolved)
   {
      log_error("Cannot de-allocate memory for non registered variables!");
   }
   assert(dataSetBase::m_data);
   dataSetBase::delete_m_data();
   dataSetBase::m_data = nullptr;
}

#endif // DATASETSTORAGE_HPP