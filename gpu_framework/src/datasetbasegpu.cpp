#include "gpu_api_functions.h" // For GPU API functions

void dataSetBaseGPU::set_offsets(const uint8_t num_offsets, const uint32_t* offsets)
{
   assert( /* ZEROD */((num_offsets == 0) && !offsets) || /* non-ZEROD */((num_offsets != 0) && offsets));

   assert(!m_offsets);

   if(num_offsets > 0)
      m_offsets = GDF::malloc_gpu_var<uint32_t, true>(num_offsets);

   m_num_offsets = num_offsets;

   // If the variable is not ZEROD, the offsets will not be null pointer and hence we have to set it
   if(offsets)
   {
      assert(m_offsets);
      GDF::memcpy_gpu_var(m_offsets, offsets, m_num_offsets);
   }
#ifndef NDEBUG
   else
   {
      assert((m_num_offsets == 0) && !m_offsets);
   }
#endif
}

void dataSetBaseGPU::free_offsets()
{
   if(m_offsets)
   {
      GDF::free_gpu_var(m_offsets);
      m_offsets = nullptr;
      m_num_offsets = 0;
   }
#ifndef NDEBUG
   else
   {
      // For ZEROD variables, the m_offsets would be a nullptr and the m_num_offsets will be 0 and hence nothing needs to done
      assert((m_num_offsets == 0) && !m_offsets);
   }
#endif
}