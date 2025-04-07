#ifndef DATASETBASE_GPU_H
#define DATASETBASE_GPU_H

#include <cstdint> // For uint32_t
#include <cstddef> // For size_t
#include <cassert> // For assert

class dataSetBaseGPU
{
public:

   dataSetBaseGPU():
      m_gpu_data(nullptr),
      m_num_offsets(0),
      m_offsets(nullptr),
      m_size(0)
#ifndef NDEBUG
      ,is_read_only(false)
#endif
   {}

   dataSetBaseGPU(void* data, const uint32_t num_offsets, uint32_t* offsets, const size_t num_entries):
      m_gpu_data(data),
      m_num_offsets(num_offsets),
      m_offsets(offsets),
      m_size(num_entries)
   {}

   ~dataSetBaseGPU(){}

   void* void_data()
   {
      return m_gpu_data;
   }

   const void* void_data() const
   {
      return m_gpu_data;
   }

   void set_data(void* data)
   {
      m_gpu_data = data;
   }

   const uint32_t* offsets() const
   {
      return m_offsets;
   }

   const uint32_t num_offsets() const
   {
      return m_num_offsets;
   }

   void set_size(const size_t size_in)
   {
      m_size = size_in;
   }

   const size_t size() const
   {
      return m_size;
   }

protected:
   void* m_gpu_data = nullptr;
   uint32_t m_num_offsets = 0;
   uint32_t* m_offsets = nullptr;
   size_t m_size = 0;
#ifndef NDEBUG
public:
   bool is_read_only = false;
#endif
};

#endif // DATASETBASE_GPU_H