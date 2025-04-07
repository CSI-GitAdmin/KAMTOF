#include <cassert>
#include <memory.h>

#include "datasetbase.h"
#include "extractor.hpp"
#include "silo_utils.h"


dataSetBase::dataSetBase(const std::string& name, const uint64_t m_num_entries, const CDF::StorageType storage_type, const CDF::PODType pod_type, const uint8_t dims, const uint8_t * const shape /* = nullptr */,
                         const bool allocate_mem /* = true */):
   m_data(nullptr),
   m_size(m_num_entries),
   m_byte_size(0),
   m_offsets(nullptr),
   m_num_offsets(dims),
   m_shape(nullptr),
   m_storage_type(storage_type),
   m_pod_type(pod_type),
   m_name(name)
{
   assert(((dims == 0) && !shape) || ((dims != 0) && shape));
   if(dims > 0)
   {
      m_offsets = new uint32_t[dims];
      m_shape = new uint8_t[dims];
      for(uint8_t off_dim = 0; off_dim < dims; off_dim++)
      {
         uint8_t cur_offset = 1;
         for(uint8_t shp_dim = off_dim; shp_dim >= 0; shp_dim--)
         {
            cur_offset *= shape[shp_dim];
         }
         m_offsets[off_dim] = cur_offset * get_single_element_byte_size(pod_type);
      }
      memcpy(m_shape, shape, dims * sizeof(uint8_t));
      m_byte_size = m_offsets[0] * m_size;
   }
   else
   {
      assert(!m_offsets && !m_num_offsets && !shape && !m_shape);
      m_byte_size = m_num_entries * get_single_element_byte_size(pod_type);
   }
   if(allocate_mem && (m_num_entries > 0))
   {
      m_data = static_cast<void*>(new char[m_byte_size]);
   }
}

dataSetBase::~dataSetBase()
{
#ifdef ENABLE_GPU
   if(gpu_instance)
   {
      validate_cpu_data_ptr();
      destruct_gpu_instance();
   }
#endif

   if(m_data)
   {
      delete[] static_cast<char*>(m_data);
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