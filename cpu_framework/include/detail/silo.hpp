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

   // Loop through all the elements and resize them
   for(std::unordered_map<std::string, dataSetBase*>::iterator cur_entry = m_data[static_cast<uint8_t>(TYPE)].begin(); cur_entry != m_data[static_cast<uint8_t>(TYPE)].end(); cur_entry++)
   {
      cur_entry->second->resize_internal(new_size);
   }
   for(std::unordered_map<std::string, dataSetBase*>::iterator cur_entry = m_unresolved_data[static_cast<uint8_t>(TYPE)].begin();
        cur_entry != m_unresolved_data[static_cast<uint8_t>(TYPE)].end(); cur_entry++)
   {
      cur_entry->second->resize_internal(new_size);
   }

   m_num_elements[static_cast<uint8_t>(TYPE)] = new_size;
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS>
dataSetBase* silo::create_entry(std::string name, const uint8_t* const shape, const bool allocate_mem, const bool is_unresolved)
{
   assert(!is_unresolved || (is_unresolved && !allocate_mem)); // mmeory should not be allocated for unresolved entries

   // If the SILO container for this TYPE isn't resized yet, do not allocate memory yet
   uint64_t num_entries = (m_num_elements[static_cast<uint8_t>(TYPE)] == std::numeric_limits<uint64_t>::max()) ? 0 : m_num_elements[static_cast<uint8_t>(TYPE)];
   dataSetBase* new_entry = new dataSetStorage<T, TYPE, DIMS>(name, num_entries, shape, is_unresolved, allocate_mem);

   if(!is_unresolved)
      m_data[static_cast<uint8_t>(TYPE)].insert({name, new_entry});
   else
      m_unresolved_data[static_cast<uint8_t>(TYPE)].insert({name, new_entry});

   return new_entry;
}

template <class T, CDF::StorageType TYPE>
dataSetStorage<T, TYPE, ZEROD>& silo::register_entry(std::string name, const bool allocate_mem /* = true */)
{
   return register_entry<T, TYPE, ZEROD>(name, nullptr, allocate_mem);
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS>
dataSetStorage<T, TYPE, DIMS>& silo::register_entry(std::string name, const uint8_t* const shape, const bool allocate_mem /* = true */)
{
   dataSetBase* new_entry = nullptr;

   std::unordered_map<std::string, dataSetBase*>::iterator unresolved_it = m_unresolved_data[static_cast<uint8_t>(TYPE)].find(name);
   if((unresolved_it != m_unresolved_data[static_cast<uint8_t>(TYPE)].end()) && (unresolved_it->second->StorageType() == TYPE))
   {
      new_entry = unresolved_it->second;
      assert(new_entry->is_unresolved);
      if(new_entry->PODType() != CDF::extractor<T>::PODType())
      {
         std::string msg = "Trying to register a variable "+ name +" with same name but different data type isn't allowed";
         log_msg<CDF::LogLevel::ERROR>(msg);
      }

      new_entry->is_unresolved = false;
      new_entry->set_offsets(DIMS, shape);
      new_entry->resize_internal(new_entry->byte_size());

      m_unresolved_data[static_cast<uint8_t>(TYPE)].erase(unresolved_it);
      m_data[static_cast<uint8_t>(TYPE)].insert({name, new_entry});
   }
   else
   {
      std::unordered_map<std::string, dataSetBase*>::iterator it = m_data[static_cast<uint8_t>(TYPE)].find(name);
      if(it == m_data[static_cast<uint8_t>(TYPE)].end())
      {
         new_entry = create_entry<T, TYPE, DIMS>(name, shape, allocate_mem, false);
      }
      else
      {
         dataSetBase* cur_entry = it->second;
         if(cur_entry->PODType() != CDF::extractor<T>::PODType())
         {
            std::string msg = "Trying to register a variable "+ name +" with same name but different data type isn't allowed";
            log_msg<CDF::LogLevel::ERROR>(msg);
         }
         if(cur_entry->StorageType() == TYPE)
         {
            std::string msg = "Trying to register a variable "+ name +" with same name and same storage type isn't allowed";
            log_msg<CDF::LogLevel::ERROR>(msg);
         }
         new_entry = create_entry<T, TYPE, DIMS>(name, shape, allocate_mem, false);
      }
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
      std::unordered_map<std::string, dataSetBase*>::iterator unresolved_it = m_unresolved_data[static_cast<uint8_t>(TYPE)].find(name);
      if(unresolved_it == m_unresolved_data[static_cast<uint8_t>(TYPE)].end()) // If not found in the resolved list, create an entry in the unresolved map
      {
         cur_entry = create_entry<T, TYPE, DIMS>(name, nullptr, false, true);
      }
      else // If found in the unresolved entry, return that
      {
         cur_entry = unresolved_it->second;
      }
   }
   else
   {
      cur_entry = it->second;
   }

   if(cur_entry->PODType() != CDF::extractor<T>::PODType())
   {
      std::string msg = "Trying to retrieve a variable "+ name +" but different data type isn't allowed";
      log_msg<CDF::LogLevel::ERROR>(msg);
   }

   assert(cur_entry->StorageType() == TYPE);
   return *(static_cast<dataSetStorage<T, TYPE, DIMS>*>(cur_entry));
}

template<CDF::StorageType TYPE, typename CallBack>
void silo::for_each(CallBack&& call_back)
{
   for(std::unordered_map<std::string, dataSetBase*>::iterator cur_entry = m_data[static_cast<uint8_t>(TYPE)].begin(); cur_entry != m_data[static_cast<uint8_t>(TYPE)].end(); cur_entry++)
   {
      call_back(cur_entry->second);
   }
}

#endif // SILO_HPP
