#ifndef DATASETSTORAGE_GPU_H
#define DATASETSTORAGE_GPU_H

#include <sycl/sycl.hpp>
// #include "indexdataset.h" // For indexer used in operator()
#include "silo_fwd.h" // For DSB and DSS
#include "datasetbasegpu.h" // For DSBGPU
#include "gpu_instance_t.h" // For GPUInstance member functions
#include "datasetstorage.h"

namespace GDF
{
class GPUManager_t;
} // namespace GDF

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
class dataSetStorageGPU : public dataSetBaseGPU
{
public:
   dataSetStorageGPU(const dataSetStorage<T, TYPE, DIMS>& dss_obj, const GDF::GPUManager_t* manager):
      dataSetBaseGPU(), // The member are null/0 intialized here and will be set later in allocate_gpu_data_ptr() because we also need to set the statuses
      m_gpu_manager((assert(manager), manager)),
      is_primary(true),
      cpu_dss_ptr(dss_obj.m_dsb()),
      m_name(dss_obj.name().c_str())
   {
#ifndef NDEBUG
      if(dss_obj.exists())
         assert(cpu_dss_ptr->get_gpu_instance()->get_xpu_data_status(GDF::xpu_t::CPU) == GDF::xpu_data_status_t::UP_TO_DATE_WRITE);
      else
         assert(cpu_dss_ptr->get_gpu_instance()->get_xpu_data_status(GDF::xpu_t::CPU) == GDF::xpu_data_status_t::NOT_ALLOCATED);
      assert(cpu_dss_ptr->get_gpu_instance()->get_xpu_data_status(GDF::xpu_t::GPU) == GDF::xpu_data_status_t::NOT_ALLOCATED);
#endif
   }

   dataSetStorageGPU(const dataSetStorageGPU& other):
      dataSetBaseGPU(other.m_gpu_data, other.m_num_offsets, other.m_offsets, other.m_size),
      m_gpu_manager(other.m_gpu_manager),
      is_primary(false), // For copies from existing DSSGPU, is_primary is set to false to avoid unintended destruction
      cpu_dss_ptr(other.cpu_dss_ptr),
      m_name(other.m_name)
   {
#ifndef NDEBUG
      is_read_only =other.is_read_only;
#endif
   }

   ~dataSetStorageGPU()
   {
      // By the time this destructor is called, the data members should already be cleaned up and set to nullptr
#ifndef DISABLE_GPU_KERNEL_ASSERTS
      if(is_primary)
      {
         assert(!this->m_gpu_data);
         assert(!this->m_offsets);
         assert(this->m_num_offsets == 0);
         assert(this->m_size == 0);
      }
#endif
   }

   const GDF::GPUManager_t* get_manager() const
   {
      return m_gpu_manager;
   }

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

   // Both const and non-const objects can call this function
   inline bool exists() const
   {
      // For now, we are doing the simple check if the GPU data exists
      return this->void_data();
      // GPU_TODO: We probably have to think about checking the GPU statuses too!
   }

private:
   const GDF::GPUManager_t* m_gpu_manager = nullptr;

   /*
    * This bool would only be set true for DSSGPU that GPU SILO backend constructs using the
    * dataSetStorageGPU(const dataSetStorage& other, const GDF::GPUManager_t* manager)
    *
    * This is because SYCL implementation copies the kernel class a lot by value and also calls the destructor on these temp copies.
    * This has the unintended consequence of calling sycl::free on offsets which would be avoided with the use of this variable which only
    * allows the "primary" object to sycl::free the offsets
   */
   const bool is_primary = false;

   /*
    * Stores a pointer of type DSS in DSB as the pointer object of this class is stored inside of GPUInstance
    * which in turn is held inside of StorageInfo which cannot be templated
    *
    * We would be returning this by casting it up using the templates of the GPU variable which might not be the same as the original CPU DSS's templates
   */
public:
   const dataSetBase* cpu_dss_ptr = nullptr;
   const char* m_name = nullptr;
};

#include "detail/datasetstoragegpu.hpp"

template <class T, CDF::StorageType TYPE, uint8_t DIMS>
struct sycl::is_device_copyable<dataSetStorageGPU<T, TYPE, DIMS>> : std::true_type {};

#endif //DATASETSTORAGE_GPU_H
