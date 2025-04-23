#ifndef DATASETSTORAGE_H
#define DATASETSTORAGE_H
#include "dataset.h"
#include <cstring>

template <class T, CDF::StorageType TYPE, uint8_t DIMS>
class dataSetStorage : public dataSet<T, DIMS>
{
public:
   dataSetStorage(const std::string& name, const uint64_t m_num_entries, const uint8_t* const shape, const bool is_unresolved_entry, const bool allocate_mem);

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

   template <class... Indices>
   inline T& operator()(Indices&&... idx)
   {
      return dataSet<T,DIMS>::operator () (static_cast<Indices&&>(idx)...);
   }

   template <class... Indices>
   const inline T& operator()(Indices&&... idx) const
   {
      return dataSet<T,DIMS>::operator () (static_cast<Indices&&>(idx)...);
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
      dataSetBase::resize_internal(new_size);
   }

   void allocate_memory();
   void deallocate_memory();

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
