#ifndef DATASET_GPU_HPP
#define DATASET_GPU_HPP

#include "datasetgpu.h"

template <class T, uint8_t DIMS /* = ZEROD */>
const uint64_t dataSetGPU<T, DIMS>::byte_size() const
{
   return total_num_elements() * sizeof(T);
}

template <class T, uint8_t DIMS /* = ZEROD */>
const uint64_t dataSetGPU<T, DIMS>::total_num_elements() const
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

#endif // DATASET_GPU_HPP