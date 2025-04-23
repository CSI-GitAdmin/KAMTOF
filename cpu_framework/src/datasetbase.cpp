#include <cassert>
#include <memory.h>
#include <sys/mman.h>

#include "datasetbase.h"
#include "extractor.hpp"
#include "silo_utils.h"
#include "cpu_globals.h"
#include "logger.hpp"


dataSetBase::dataSetBase(const std::string& name, const uint64_t m_num_entries, const CDF::StorageType storage_type, const CDF::PODType pod_type, const uint8_t dims,
                         const uint8_t * const shape, const bool is_unresolved_entry, const bool allocate_mem):
   m_data(nullptr),
   m_size(m_num_entries),
   m_byte_size(0),
   m_offsets(nullptr),
   m_num_offsets((dims) ? (dims+1) : 0), // For ZEROD m_num_offsets is 0 and for nD m_num_offsets is n+1 (Where n > 0)
   m_shape(nullptr),
   m_storage_type(storage_type),
   m_pod_type(pod_type),
   m_name(name),
   is_unresolved(is_unresolved_entry)
{
   if(!is_unresolved_entry)
   {
      set_offsets(dims, shape);
      if(allocate_mem && (m_num_entries > 0))
      {
         allocate_m_data(m_byte_size);
      }
   }
}

void dataSetBase::set_offsets(const uint8_t dims, const uint8_t * const shape)
{
   assert(((dims == 0) && !shape) || ((dims != 0) && shape));
   if(dims > 0)
   {
      m_shape = new uint8_t[dims];
      memcpy(m_shape, shape, dims * sizeof(uint8_t));

      m_offsets = new uint32_t[m_num_offsets];
      m_offsets[0] = 1;
      for(uint8_t cur_dim = 0; cur_dim < dims; cur_dim++)
      {
         m_offsets[0] *= m_shape[cur_dim]; // product of all the elements in the shape array
      }
      for(uint8_t cur_dim = 0; cur_dim < dims; cur_dim++)
      {
         m_offsets[cur_dim+1] = m_offsets[cur_dim] / m_shape[cur_dim];
      }
      assert(m_offsets[dims] == 1);

      m_byte_size = m_offsets[0] * m_size * get_single_element_byte_size(m_pod_type);
   }
   else
   {
      assert(!m_offsets && !m_num_offsets && !shape && !m_shape);
      m_byte_size = m_size * get_single_element_byte_size(m_pod_type);
   }
}

void dataSetBase::resize_internal(const uint64_t new_size)
{
   assert(!m_num_offsets || m_offsets); // Make sure offsets are set (or) the element in ZEROD
   uint64_t new_byte_size = (m_offsets ? m_offsets[0] : get_single_element_byte_size(m_pod_type)) * new_size;
   if(new_size == 0)
   {
      // Delete the old memory, set the size and byte_size to 0
      if(m_data)
      {
         delete_m_data();
         m_data = nullptr;
         m_size = 0;
         m_byte_size = 0;
      }
#ifndef NDEBUG
      else
      {
         assert(m_size == 0 || is_unresolved);
         assert(m_byte_size == 0 || is_unresolved);
      }
#endif
   }
   else if(!is_unresolved)
   {
      // m_data exists, we need copy it over and free it
      if(m_data)
      {
#ifdef GPU_DEVELOP
         copy_over_and_resize(new_byte_size);
#else
         assert(m_size != 0 && m_byte_size != 0);
         // Allocate new data and copy over data
         void* new_data = static_cast<void*>(new char[new_byte_size]());
         uint64_t copy_byte_size = (new_byte_size <= m_byte_size) ? new_byte_size : m_byte_size;
         memcpy(new_data, m_data, copy_byte_size);

         // Pointer Swap
         delete_m_data();
         m_data = new_data;
#endif
      }
      else // m_data is null, so we do not need to do copy and free
      {
         allocate_m_data(new_byte_size);
      }
   }
#ifndef NDEBUG
   else // If unresolved, data should be NULL
   {
      assert(!m_data);
   }
#endif

   m_size = new_size;
   m_byte_size = new_byte_size;

#ifdef ENABLE_GPU
   // Trigger the resized status for this variable
   set_gpu_data_status_to_resized();
#endif
}

dataSetBase::~dataSetBase()
{
#ifdef ENABLE_GPU
   if(gpu_instance)
   {
      destruct_gpu_instance();
   }
#endif

   if(m_data)
   {
      delete_m_data();
      m_data = nullptr;
      m_size = 0;
      m_byte_size = 0;
   }

   if(m_shape)
   {
      delete[] m_shape;
      m_shape = nullptr;
   }

   if(m_offsets)
   {
      delete[] m_offsets;
      m_offsets = nullptr;
      m_num_offsets = 0;
   }
}

void dataSetBase::allocate_m_data(const uint64_t byte_size)
{
   assert(!m_data);
#ifdef GPU_DEVELOP
   allocate_page_aligned_memory_internal(byte_size);
#else
   m_data = static_cast<void*>(new char[byte_size]());
#endif
}

void dataSetBase::delete_m_data()
{
   assert(m_data);
#ifdef GPU_DEVELOP
   deallocate_page_aligned_memory_internal();
#else
   delete[] static_cast<char*>(m_data);
#endif
}
