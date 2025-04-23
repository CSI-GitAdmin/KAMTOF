#ifndef DATASET_GPU_H
#define DATASET_GPU_H

#include <cstdint> // For uint32_t
#include <cstddef> // For size_t
#include <cassert> // For assert
#include "datasetbasegpu.h"
#include "indexdataset.h"

template <class T, uint8_t DIMS /* = ZEROD */>
class dataSetGPU : public dataSetBaseGPU
{
public:

   dataSetGPU():dataSetBaseGPU()
   {}

   dataSetGPU(void* data, const uint32_t num_offsets, uint32_t* offsets, const size_t num_entries):
      dataSetBaseGPU(data, num_offsets, offsets, num_entries)
   {}

   const size_t byte_size() const;

   const T* data() const
   {
      return static_cast<T*>(this->m_gpu_data);
   }

   T* data()
   {
      return static_cast<T*>(this->m_gpu_data);
   }

   const uint64_t total_num_elements() const;

   inline const T& operator[](uint64_t idx) const
   {
      static_assert(DIMS == 0, "The [] operator is only for ZEROD in release mode and in debug mode for users to index the data as a 1D array");
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(this->m_gpu_data && "Variable is not transfered to GPU (or) Non existent SILO variable is used on the GPU");
      assert(idx < this->m_num_entries);
#endif
      return static_cast<T*>(this->m_gpu_data)[idx];
   }

   inline T& operator[](uint64_t idx)
   {
      static_assert(DIMS == 0, "The [] operator is only for ZEROD in release mode and in debug mode for users to index the data as a 1D array");
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(this->m_gpu_data && "Variable is not transfered to GPU (or) Non existent SILO variable is used on the GPU");
      assert(!is_read_only && "Trying to get access to a read only data in an editable way");
      assert(idx < this->m_num_entries);
#endif
      return static_cast<T*>(this->m_gpu_data)[idx];
   }

   template <class... Indices>
   inline T& operator()(Indices&&... idx)
   {
      static_assert(DIMS > 0, "This interface is not intended for 0 dimensional data");
      static_assert(DIMS + 1 == sizeof...(Indices), "You have provided the wrong number of arguments");
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(this->m_gpu_data && "Variable is not transfered to GPU (or) Non existent SILO variable is used on the GPU");
      assert(!is_read_only && "Trying to get access to a read only data in an editable way");
#endif
      return indexer<T, DIMS>::access(static_cast<T*>(m_gpu_data), m_offsets, DIMS+1, static_cast<Indices&&>(idx)...);
   }

   template <class... Indices>
   const inline T& operator()(Indices&&... idx) const
   {
      static_assert(DIMS > 0, "This interface is not intended for 0 dimensional data");
      static_assert(DIMS + 1 == sizeof...(Indices), "You have provided the wrong number of arguments");
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      assert(this->m_gpu_data && "Variable is not transfered to GPU (or) Non existent SILO variable is used on the GPU");
#endif
      return indexer<T, DIMS>::access_const(static_cast<T*>(m_gpu_data), m_offsets, DIMS+1, static_cast<Indices&&>(idx)...);
   }

   // Both const and non-const objects can call this function
   inline bool exists() const
   {
      // For now, we are doing the simple check if the GPU data exists
      return this->void_data();
      // GPU_TODO: We probably have to think about checking the GPU statuses too!
   }

   ~dataSetGPU(){}
};

#include "datasetgpu.hpp"

template <class T, uint8_t DIMS>
struct sycl::is_device_copyable<dataSetGPU<T, DIMS>> : std::true_type {};

#endif // DATASET_GPU_H