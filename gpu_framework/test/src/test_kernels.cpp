#include <cmath> // For pow to take square
#include "test_kernels.h"
#include "gpu_globals.h" // For gpu_stream_data
#include "gpu_atomics.h" // For atomics
#include "gpu_api_functions.h"


void kg_scale_pressure_and_change_scale::operator() (sycl::nd_item<3> itm) const
{
   size_t idx = GDF::get_1d_index(itm);
   if(idx < pressure.size())
   {
      pressure[idx] *= scale[0];
   }
}

void kg_compute_temperature::operator() (sycl::nd_item<3> itm) const
{
   size_t idx = GDF::get_1d_index(itm);
   if(idx < temperature.size())
   {
      temperature[idx] = (pressure[idx] * volume[idx])/(n * R);
   }
}

void kg_compute_pressure::operator() (sycl::nd_item<3> itm) const
{
   size_t idx = GDF::get_1d_index(itm);
   if(idx < pressure.size())
   {
      pressure[idx] = (n * R * temperature[idx]) / (volume[idx]);
   }
}

void kg_test_kernel::operator() (sycl::nd_item<3> itm) const
{
   size_t idx = GDF::get_1d_index(itm);
   if(idx && x && y)
   {
      idx = x;
   }
}

void kg_set_initial_condition::operator() (sycl::nd_item<3> itm) const
{
   size_t idx = GDF::get_1d_index(itm);
   if(idx < velocity.size())
   {
      velocity(idx,0) = init_val_x;
      velocity(idx,1) = init_val_y;
      velocity(idx,2) = init_val_z;
   }
}

void kg_norm2::operator() (sycl::nd_item<3> itm) const
{
   size_t idx = GDF::get_1d_index(itm);
   size_t stride = GDF::get_1d_stride(itm);
#ifndef DISABLE_GPU_KERNEL_ASSERTS
   // This kernel works under the assumption that there is only one work-group
   assert(itm.get_global_range(2) == itm.get_local_range(2));
#endif
   strict_fp_t sum = 0.0;
   for(size_t ii = idx; ii < size; ii += stride)
   {
      sum += sycl::pow(vec[ii], 2.0);
   }

   if(idx < size)
      vec[idx] = sum;

   itm.barrier();

   if(idx == 0)
   {
      sum = 0.0;
      for(size_t ii = 0; ii < stride; ii++)
      {
         sum += vec[ii];
      }
      vec[1] = sum;
      vec[0] = sycl::sqrt(sum);
   }
}

void kg_atomics::operator() (sycl::nd_item<3> item) const
{
   size_t idx = GDF::get_1d_index(item);
   size_t stride = GDF::get_1d_stride(item);
   for(size_t ii = idx; ii < size; ii += stride)
   {
      GDF::atomic_add(result[0], vec[ii]);
      GDF::atomic_sub(result[1], vec[ii]);
      GDF::atomic_max(result[2], vec[ii]);
      GDF::atomic_min(result[3], vec[ii]);
   }
}

void kg_silo_null::operator ()(sycl::nd_item<3> item) const
{
   size_t idx = GDF::get_1d_index(item);
   // Only do the subtraction if the GPU SILO object exist
   if(gpu_silo_null.exists() && idx == gpu_random_idx)
   {
      gpu_silo_null[gpu_random_idx] -= gpu_subtract_val;
   }
}