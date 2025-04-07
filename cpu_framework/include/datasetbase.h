#ifndef DATASETBASE_H
#define DATASETBASE_H

#include <cstdint>
#include <cassert>
#include <string>
#include "extractor.hpp"

namespace CDF
{
enum class StorageType : uint8_t;
enum class PODType: uint8_t;
}

#ifdef ENABLE_GPU
namespace GDF
{
class GPUInstance_t;
enum class xpu_data_status_t : uint8_t;
enum class xpu_t : uint8_t;
} // namespace GDF
#endif

class dataSetBase
{
public:
   dataSetBase(const std::string& name, const uint64_t m_num_entries, const CDF::StorageType storage_type, const CDF::PODType pod_type, const uint8_t dims, const uint8_t* const shape = nullptr, const bool allocate_mem = true);

   dataSetBase(const dataSetBase& other) = delete;

   ~dataSetBase();

   const uint64_t size() const
   {
      return m_size;
   }

   void setData(void* data)
   {
      m_data = data;
   }

   bool exists() const // FIXME
   {
      return true;
   }

   const std::string& name() const
   {
      return m_name;
   }

   const uint64_t byte_size() const
   {
      return m_byte_size;
   }

   inline void* cpu_data()
   {
#ifdef GPU_DEVELOP
      switch(writeable_on_cpu)
      {
         case 0:
            transfer_to_cpu();
         default:
         {
            return m_data;
         }
      }
#else
      return m_data;
#endif
   }
   inline const void* cpu_data() const
   {
#ifdef GPU_DEVELOP
      switch(readable_on_cpu)
      {
         case 0:
            transfer_to_cpu(true);
         default:
         {
            return m_data;
         }
      }
#else
      return m_data;
#endif
   }


   template<class T>
   inline T& operator[](const uint64_t index)
   {
#ifndef NDEBUG
      assert(CDF::extractor<T>::PODType() == m_pod_type); // Make sure the calling type is the same as the register type
      assert(index < m_size && !m_offsets); // Ensure that the index is within bounds
#endif
#ifdef GPU_DEVELOP
      switch(writeable_on_cpu)
      {
         case 0:
            transfer_to_cpu();
         default:
         {
#ifndef NDEBUG
            assert_cpu_data_writeability();
#endif
            return static_cast<T*>(m_data)[index];
         }
      }
#else
#ifndef NDEBUG
#ifdef ENABLE_GPU
      assert_cpu_data_writeability();
#endif
#endif
      return static_cast<T*>(m_data)[index];
#endif
   }

   template<class T>
   const inline T& operator[](const uint64_t index) const
   {
#ifndef NDEBUG
      assert(CDF::extractor<T>::PODType() == m_pod_type); // Make sure the calling type is the same as the register type
      assert(index < m_size && !m_offsets); // Ensure that the index is within bounds
#endif
#ifdef GPU_DEVELOP
      switch(readable_on_cpu)
      {
         case 0:
            transfer_to_cpu(true);
         default:
         {
            return static_cast<T*>(m_data)[index];
         }
      }
#else
      return static_cast<T*>(m_data)[index];
#endif
   }

   void* m_data;

   uint64_t m_size;
   uint64_t m_byte_size;

   uint32_t* m_offsets;
   uint8_t m_num_offsets;
   uint8_t* m_shape;

   CDF::PODType m_pod_type;
   CDF::StorageType m_storage_type;
   std::string m_name;

#ifdef ENABLE_GPU
   // The mutable keyword allows const functions modify it
   mutable GDF::GPUInstance_t* gpu_instance = nullptr;
   mutable int writeable_on_cpu = true;
   mutable int readable_on_cpu = true;

   void* gpu_data(void)
   {
      return get_gpu_void_data();
   }
   const void* gpu_data(void) const
   {
      return get_gpu_void_data();
   }

   GDF::GPUInstance_t* get_gpu_instance()
   {
      return gpu_instance;
   }

   const GDF::GPUInstance_t* get_gpu_instance() const
   {
      return gpu_instance;
   }

   void set_gpu_instance(GDF::GPUInstance_t* other) const
   {
      gpu_instance = other;
   }

   void deallocate_gpu_data_ptr();
   void deallocate_gpu_instance();
   void destruct_gpu_instance();
   void set_gpu_data_status_to_resized();
   void* get_gpu_void_data();
   const void* get_gpu_void_data() const;
   const GDF::xpu_data_status_t& get_xpu_data_status(const GDF::xpu_t &device_type) const;
   void assert_cpu_data_writeability();
   void validate_cpu_data_ptr();
#ifdef GPU_DEVELOP
   void transfer_to_cpu(bool read_only = false) const;
#endif
#endif
};

#endif // DATASETBASE_H
