#ifndef DATASETSTORAGE_H
#define DATASETSTORAGE_H
#include "dataset.h"
#include <cstring>

template <class T, CDF::StorageType TYPE, uint8_t DIMS>
class dataSetStorage : public dataSet<T, DIMS>
{
public:
   dataSetStorage(const std::string& name, const uint64_t m_num_entries, const uint8_t* const shape = nullptr, const bool allocate_mem = true);

   dataSetStorage(const dataSetStorage& other) = delete;

   template <class R>
   inline void operator= (const R& other)
   {
      dataSet<T, DIMS>::operator=(other);
   }

   inline T& operator[] (uint64_t index)
   {
      static_assert(DIMS == ZEROD, "This operator can only be called on ZEROD variables");
      return dataSet<T, DIMS>::operator[](index);
   }

   inline const T& operator[] (uint64_t index) const
   {
      static_assert(DIMS == ZEROD, "This operator can only be called on ZEROD variables");
      return dataSet<T, DIMS>::operator[](index);
   }

   inline operator T& ()
   {
      static_assert(TYPE == CDF::StorageType::PARAMETER && DIMS == 0, "This functionality is only supported for parameters");
      return static_cast<T*>(dataSetBase::m_data)[0];
   }

   inline operator const T& () const
   {
      static_assert(TYPE == CDF::StorageType::PARAMETER && DIMS == 0, "This functionality is only supported for parameters");
      return static_cast<const T*>(dataSetBase::m_data)[0];
   }

   void resize(const uint64_t new_size)
   {
      static_assert(TYPE == CDF::StorageType::VECTOR, "Resizing induvidual SILO variables is only available for VECTOR StorageType");
      if(new_size == 0)
      {
         // Delete the old memory, set the size and byte_size to 0
         if(dataSetBase::m_data)
         {
            delete[] static_cast<char*>(dataSetBase::m_data);
            dataSetBase::m_data = nullptr;
            dataSetBase::m_size = 0;
            dataSetBase::m_byte_size = 0;
         }
#ifndef NDEBUG
         else
         {
            assert(dataSetBase::m_size == 0);
            assert(dataSetBase::m_byte_size == 0);
         }
#endif
      }
      else
      {
         // m_data exists, we need copy it over and free it
         if(dataSetBase::m_data)
         {
            assert(dataSetBase::m_size != 0 && dataSetBase::m_byte_size != 0);
            // Allocate new data and copy over data
            T* new_data = new T[new_size]();
            uint64_t copy_byte_size = sizeof(T) * ((new_size <= dataSetBase::m_size) ? new_size : dataSetBase::m_size);
            memcpy(new_data, dataSetBase::m_data, copy_byte_size);


            // Pointer Swap
            void* delete_data = dataSetBase::m_data;
            dataSetBase::m_data = static_cast<void*>(new_data);

            // Free the old data
            delete[] static_cast<char*>(delete_data);
         }
         else // m_data is null, so we do not need to do copy and free
         {
            dataSetBase::m_data = new T[new_size]();
         }
      }

      dataSetBase::m_size = new_size;
      dataSetBase::m_byte_size = new_size * CDF::extractor<T>::single_element_byte_size();
   }

   inline T* cpu_data()
   {
      return dataSet<T, DIMS>::cpu_data();
   }

   inline const T* cpu_data() const
   {
      return dataSet<T, DIMS>::cpu_data();
   }

   inline const dataSetBase* m_dsb() const
   {
      return static_cast<const dataSetBase*>(this);
   }

   inline dataSetBase* m_dsb()
   {
      return static_cast<dataSetBase*>(this);
   }

#ifdef ENABLE_GPU
   inline const T* gpu_data() const
   {
      return dataSet<T, DIMS>::gpu_data();
   }

   inline T* gpu_data()
   {
      return dataSet<T, DIMS>::gpu_data();
   }
#endif
};

#include "datasetstorage.hpp"

#endif // DATASETSTORAGE_H
