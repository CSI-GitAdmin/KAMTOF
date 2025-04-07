#ifndef DATASETSTORAGE_GPU_HPP
#define DATASETSTORAGE_GPU_HPP

#include "datasetstoragegpu.h"

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
const uint64_t dataSetStorageGPU<T, TYPE, DIMS>::byte_size() const
{
   return total_num_elements() * sizeof(T);
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
const uint64_t dataSetStorageGPU<T, TYPE, DIMS>::total_num_elements() const
{
   if constexpr (DIMS == ZEROD)
   {
      assert(!m_offsets);
      return this->m_size;
   }
   else
   {
      assert(m_offsets);
      return this->m_size*this->m_offsets[0];
   }
}

#endif // DATASETSTORAGE_GPU_HPP