#ifndef DATASET_H
#define DATASET_H

#include "datasetbase.h"

template <class T, uint8_t DIMS>
class dataSet : public dataSetBase
{
public:
   dataSet(const std::string& name, const uint64_t m_num_entries, const CDF::StorageType storage_type, const uint8_t* const shape = nullptr, const bool allocate_mem = true);

   dataSet(const dataSet& other) = delete;

   inline T& operator[] (uint64_t index)
   {
      static_assert(DIMS == ZEROD, "This operator can only be called on ZEROD variables");
      return dataSetBase::operator[]<T>(index);
   }

   inline const T& operator[] (uint64_t index) const
   {
      static_assert(DIMS == ZEROD, "This operator can only be called on ZEROD variables");
      return dataSetBase::operator[]<T>(index);
   }

   template <class R>
   inline void operator= (R& other)
   {
      static_assert(std::is_fundamental_v<R>, "Can only set from POD types");
      static_assert(DIMS == ZEROD, "This operator can only be called on ZEROD variables");
      T* cdata = static_cast<T*>(dataSetBase::m_data);
      for(uint64_t ee = 0; ee < dataSetBase::m_size; ee++)
      {
         new(cdata) T(other);
         cdata++;
      }
   }

   inline const dataSetBase* m_dsb() const
   {
      return static_cast<const dataSetBase*>(this);
   }

   inline dataSetBase* m_dsb()
   {
      return static_cast<dataSetBase*>(this);
   }

   inline T* cpu_data()
   {
      return static_cast<T*>(dataSetBase::cpu_data());
   }

   inline const T* cpu_data() const
   {
      return static_cast<const T*>(dataSetBase::cpu_data());
   }

#ifdef ENABLE_GPU
   inline const T* gpu_data() const
   {
      return static_cast<const T*>(dataSetBase::gpu_data());
   }

   inline T* gpu_data()
   {
      return static_cast<T*>(dataSetBase::gpu_data());
   }
#endif
};

#include "dataset.hpp"

#endif // DATASET_H
