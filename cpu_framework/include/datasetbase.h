#ifndef DATASETBASE_H
#define DATASETBASE_H

#include <cstdint>
#include <cassert>
#include <string>
#include "extractor.hpp"
#include "cpu_globals.h"

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
   dataSetBase(const std::string& name, const uint64_t m_num_entries, const CDF::StorageType storage_type, const CDF::PODType pod_type, const uint8_t dims, const uint8_t* const shape,
               const bool is_unresolved_entry, const bool allocate_mem);

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

   bool exists() const
   {
      return !is_unresolved;
   }

   const std::string& name() const
   {
      return m_name;
   }

   const uint64_t byte_size() const
   {
      return m_byte_size;
   }

   void* cpu_data()
   {
      return m_data;
   }
   const void* cpu_data() const
   {
      return m_data;
   }

   template<class T>
   inline T& operator[](const uint64_t index)
   {
#ifndef NDEBUG
      assert(CDF::extractor<T>::PODType() == m_pod_type); // Make sure the calling type is the same as the register type
      assert(index < m_size && !m_offsets); // Ensure that the index is within bounds
#endif
#ifdef GPU_DEVELOP
      in_const_operator = false;
#endif
      return static_cast<T*>(m_data)[index];
   }

   template<class T>
   const inline T& operator[](const uint64_t index) const
   {
#ifndef NDEBUG
      assert(CDF::extractor<T>::PODType() == m_pod_type); // Make sure the calling type is the same as the register type
      assert(index < m_size && !m_offsets); // Ensure that the index is within bounds
#endif
#ifdef GPU_DEVELOP
      in_const_operator = true;
#endif
      return static_cast<T*>(m_data)[index];
   }

   CDF::StorageType StorageType() const
   {
      return m_storage_type;
   }

   CDF::PODType PODType() const
   {
      return m_pod_type;
   }

   const uint32_t* const offsets() const
   {
      return m_offsets;
   }

   const uint8_t num_offsets() const
   {
      return m_num_offsets;
   }

protected:

   void set_offsets(const uint8_t dims, const uint8_t* const shape);
   void allocate_m_data(const uint64_t byte_size);
   void delete_m_data();
   void resize_internal(const uint64_t new_byte_size);

   void* m_data;

   uint64_t m_size;
   uint64_t m_byte_size;

   uint32_t* m_offsets;
   uint8_t m_num_offsets;
   uint8_t* m_shape;

   CDF::PODType m_pod_type;
   CDF::StorageType m_storage_type;
   std::string m_name;
   bool is_unresolved;

#ifdef ENABLE_GPU
protected:
   // The mutable keyword allows const functions modify it
   mutable GDF::GPUInstance_t* gpu_instance = nullptr;
   uint64_t allocation_size = 0; // The cloest multiple of system_page_size which is greater than m_byte_size

   void deallocate_gpu_data_ptr();
   void allocate_page_aligned_memory_internal(const uint64_t byte_size);
   void deallocate_page_aligned_memory_internal();
   void copy_over_and_resize_page_aligned_memory(const uint64_t new_byte_size);
   void deallocate_gpu_instance();
   void destruct_gpu_instance();
   void* get_gpu_void_data();
   const void* get_gpu_void_data() const;
#ifndef NDEBUG
   void assert_cpu_data_writeability();
#endif

public:

   void transfer_to_cpu(bool read_only = false) const;

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

   uint64_t get_allocation_size() const
   {
      return allocation_size;
   }

   void set_gpu_data_status_to_resized();
   const GDF::xpu_data_status_t& get_xpu_data_status(const GDF::xpu_t &device_type) const;
#endif

   friend class silo;
};

#endif // DATASETBASE_H
