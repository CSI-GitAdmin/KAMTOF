#ifndef SILO_HPP
#define SILO_HPP

#include <cassert>

#include "silo.h"
#include "datasetstorage.h"
#include "extractor.hpp"
#include "silo_utils.h"
#include "logger.hpp"
#include <limits>

template <CDF::StorageType TYPE>
void silo::resize(const uint64_t& new_size)
{
   static_assert(TYPE < CDF::StorageType::NUM_STORAGE_TYPES, "Global silo resize needs a valid StorageType");
   static_assert(TYPE != CDF::StorageType::VECTOR, "Global silo size cannot be set for Vector variables");
   static_assert(TYPE != CDF::StorageType::PARAMETER, "Global silo size cannot be set for Parameter variables");
   assert(m_num_elements.size() ==  static_cast<uint8_t>(CDF::StorageType::NUM_STORAGE_TYPES));
   m_num_elements[static_cast<uint8_t>(TYPE)] = new_size;
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS>
dataSetBase* silo::create_entry(std::string name, const uint8_t* const shape, const bool allocate_mem)
{
   dataSetBase* new_entry = new dataSetStorage<T, TYPE, DIMS>(name, m_num_elements[static_cast<uint8_t>(TYPE)], shape, allocate_mem);

   new_entry->m_pod_type = CDF::extractor<T>::PODType(); // extract the data type of the variable
   new_entry->m_storage_type = TYPE;

   m_data[static_cast<uint8_t>(TYPE)].insert({name, new_entry});
   return new_entry;
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS>
dataSetStorage<T, TYPE, DIMS>& silo::register_entry(std::string name, const bool allocate_mem /* = true */)
{
   static_assert(DIMS == ZEROD, "This register entry call can only be used for ZEROD objects");
   return register_entry<T, TYPE, DIMS>(name, nullptr, allocate_mem);
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS>
dataSetStorage<T, TYPE, DIMS>& silo::register_entry(std::string name, const uint8_t* const shape, const bool allocate_mem /* = true */)
{
   assert(m_num_elements[static_cast<uint8_t>(TYPE)] != std::numeric_limits<uint64_t>::max()); // FIXME: Maybe allow for register without resize ?
   std::unordered_map<std::string, dataSetBase*>::iterator it = m_data[static_cast<uint8_t>(TYPE)].find(name);
   dataSetBase* new_entry = nullptr;
   if(it == m_data[static_cast<uint8_t>(TYPE)].end())
   {
      new_entry = create_entry<T, TYPE, DIMS>(name, shape, allocate_mem);
   }
   else
   {
      dataSetBase* cur_entry = it->second;
      if(cur_entry->m_pod_type != CDF::extractor<T>::PODType())
      {
         std::string msg = "Trying to register a variable "+ name +" with same name but different data type isn't allowed";
         log_msg<CDF::LogLevel::ERROR>(msg);
      }
      if(cur_entry->m_storage_type == TYPE)
      {
         std::string msg = "Trying to register a variable "+ name +" with same name and same storage type isn't allowed";
         log_msg<CDF::LogLevel::ERROR>(msg);
      }
      new_entry = create_entry<T, TYPE, DIMS>(name, shape, allocate_mem);
   }
   return *(static_cast<dataSetStorage<T, TYPE, DIMS>*>(new_entry));
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS>
dataSetStorage<T, TYPE, DIMS>& silo::retrieve_entry(std::string name)
{
   std::unordered_map<std::string, dataSetBase*>::iterator it = m_data[static_cast<uint8_t>(TYPE)].find(name);
   dataSetBase* cur_entry = nullptr;
   if(it == m_data[static_cast<uint8_t>(TYPE)].end())
   {
#ifndef NDEBUG
      for(int ii = 0; ii < static_cast<int>(CDF::StorageType::NUM_STORAGE_TYPES); ii++)
      {
         if(TYPE != static_cast<CDF::StorageType>(ii))
         {
            std::unordered_map<std::string, dataSetBase*>::iterator found = m_data[ii].find(name);
            if(found !=  m_data[ii].end())
            {
               std::string msg = "Variable "+ name +" is only registered with type" + get_storage_type_name(static_cast<CDF::StorageType>(ii)) + "but is requested with type " +
                                 get_storage_type_name(TYPE);
               log_msg<CDF::LogLevel::ERROR>(msg);
            }
         }
      }
#endif
      std::string msg = "The variable "+ name +" requested has not been registered";
      log_msg<CDF::LogLevel::ERROR>(msg);
   }
   else
   {
      cur_entry = it->second;
      if(cur_entry->m_pod_type != CDF::extractor<T>::PODType())
      {
         std::string msg = "Trying to retrieve a variable "+ name +" but different data type isn't allowed";
         log_msg<CDF::LogLevel::ERROR>(msg);
      }
      assert(cur_entry->m_storage_type == TYPE);
   }
   return *(static_cast<dataSetStorage<T, TYPE, DIMS>*>(cur_entry));
}

#endif // SILO_HPP