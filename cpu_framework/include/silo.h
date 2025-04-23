#ifndef SILO_H
#define SILO_H

#ifndef CDF_GLOBAL
#define CDF_GLOBAL extern
#endif

#include <unordered_map>
#include <string>
#include <vector>
#include "datasetstorage.h"
#include "cpu_framework_enums.h"

class silo
{
public:
   silo();

   ~silo();

   // Function to register ZEROD entries
   template <class T, CDF::StorageType TYPE>
   dataSetStorage<T, TYPE, ZEROD>& register_entry(std::string name, const bool allocate_mem = true);

   // Function to register nD entries
   template <class T, CDF::StorageType TYPE, uint8_t DIMS>
   dataSetStorage<T, TYPE, DIMS>& register_entry(std::string name, const uint8_t* const shape, const bool allocate_mem = true);

   // Retrieve already registered SILO entries
   template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
   dataSetStorage<T, TYPE, DIMS>& retrieve_entry(std::string name);

   // Resize storage types for which all variables of the same size (CELL, FACE etc). Should not be called on VECTOR/PARAMETER
   template <CDF::StorageType TYPE>
   void resize(const uint64_t& new_size);

   template<CDF::StorageType TYPE>
   uint64_t get_size()
   {
      static_assert(TYPE != CDF::StorageType::VECTOR, "This function cannot be called on a VECTOR type");
      static_assert(TYPE != CDF::StorageType::NUM_STORAGE_TYPES, "This function must be called on a valid type");
      return m_num_elements[static_cast<uint8_t>(TYPE)];
   }

   template<CDF::StorageType TYPE, typename CallBack>
   void for_each(CallBack&& call_back);

   // Remove all SILO entries
   void clear_entries();

private:

   // Function to register nD entries
   template <class T, CDF::StorageType TYPE, uint8_t DIMS>
   dataSetStorage<T, TYPE, DIMS>& register_unresolved_entry(std::string name, const uint8_t* const shape, const bool allocate_mem = true);

   // Add a new DSS of given template type into the map
   template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
   dataSetBase* create_entry(std::string name, const uint8_t* const shape, const bool allocate_mem, const bool is_unresolved);


   std::vector<uint64_t> m_num_elements; // Size of each predefined storage types
   std::vector<std::unordered_map<std::string, dataSetBase*>> m_data; // Container to hold registered variables
   std::vector<std::unordered_map<std::string, dataSetBase*>> m_unresolved_data; // Container to hold unregistered variables

};

CDF_GLOBAL silo m_silo;

#include "silo.hpp"

#endif // SILO_H