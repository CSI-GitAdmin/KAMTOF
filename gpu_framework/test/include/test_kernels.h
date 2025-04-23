#ifndef TEST_KERNELS_H
#define TEST_KERNELS_H

#include "datasetstoragegpu.h" // For DSSGPU
#include "gpu_enums.h" // For GDF namespace
#include "gpu_api_functions.h"

constexpr strict_fp_t tol = 1.0e-10;

class kg_scale_pressure_and_change_scale
{
public:
   kg_scale_pressure_and_change_scale(CellGPU<strict_fp_t>& pressure_in, ParameterGPU<strict_fp_t> scale_in):
      pressure(pressure_in),
      scale(scale_in)
   {}

   SYCL_EXTERNAL void operator() (sycl::nd_item<3> itm) const;

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      GDF::transfer_vars_to_gpu_impl<N>(pressure, scale);
   }

private:
   mutable CellGPU<strict_fp_t> pressure;
   mutable ParameterGPU<strict_fp_t> scale;
};

class kg_compute_temperature
{
public:
   kg_compute_temperature(CellGPU<strict_fp_t>& temperature_in, CellGPURead<strict_fp_t>& pressure_in, const strict_fp_t n_in, const strict_fp_t R_in, CellGPURead<strict_fp_t>& volume_in):
      temperature(temperature_in),
      pressure(pressure_in),
      volume(volume_in),
      n(n_in),
      R(R_in)
   {}

   SYCL_EXTERNAL void operator() (sycl::nd_item<3> itm) const;

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      GDF::transfer_vars_to_gpu_impl<N>(pressure, volume, temperature,n, R);
   }

private:
   mutable CellGPU<strict_fp_t> temperature;
   CellGPURead<strict_fp_t> pressure;
   CellGPURead<strict_fp_t> volume;
   strict_fp_t n;
   strict_fp_t R;
};

class kg_compute_pressure
{
public:
   kg_compute_pressure(CellGPU<strict_fp_t>& pressure_in, CellGPURead<strict_fp_t>& volume_in, const strict_fp_t n_in, const strict_fp_t R_in, CellGPURead<strict_fp_t>& temperature_in):
      pressure(pressure_in),
      volume(volume_in),
      temperature(temperature_in),
      n(n_in),
      R(R_in)
   {}

   SYCL_EXTERNAL void operator() (sycl::nd_item<3> itm) const;

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(pressure, volume, temperature,n, R);
   }

private:
   mutable CellGPU<strict_fp_t> pressure;
   CellGPURead<strict_fp_t> volume;
   CellGPURead<strict_fp_t> temperature;
   strict_fp_t n;
   strict_fp_t R;
};

class kg_test_kernel
{
public:
   kg_test_kernel(const int xx, const strict_fp_t yy):x(xx), y(yy)
   {}

   SYCL_EXTERNAL void operator() (sycl::nd_item<3> itm) const;

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      GDF::transfer_vars_to_gpu_impl<N>(x, y);
   }

private:
   int x;
   strict_fp_t y;

};

class kg_set_initial_condition
{
public:
   kg_set_initial_condition(CellGPU<double,1>& velocity_in, const double init_val_x_in, const double init_val_y_in, const double init_val_z_in):
      velocity(velocity_in), init_val_x(init_val_x_in), init_val_y(init_val_y_in), init_val_z(init_val_z_in)
   {}

   SYCL_EXTERNAL void operator() (sycl::nd_item<3> itm) const;

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      GDF::transfer_vars_to_gpu_impl<N>(velocity, init_val_x, init_val_y, init_val_z);
   }

private:
   mutable CellGPU<double,1> velocity;
   double init_val_x;
   double init_val_y;
   double init_val_z;
};

class kg_norm2
{
public:
   kg_norm2(strict_fp_t* vec_in, const size_t size_in):
      vec(vec_in),
      size(size_in)
   {}

   SYCL_EXTERNAL void operator() (sycl::nd_item<3> itm) const;

private:
   strict_fp_t* vec;
   const size_t size;
};

class kg_atomics
{
public:
   kg_atomics(strict_fp_t* result_in, const strict_fp_t* vec_in, const size_t size_in):
      result(result_in),
      vec(vec_in),
      size(size_in)
   {}

   SYCL_EXTERNAL void operator() (sycl::nd_item<3> itm) const;

private:
   strict_fp_t* result;
   const strict_fp_t* vec;
   const size_t size;
};

class kg_silo_null
{
public:
   kg_silo_null(const size_t random_idx, CellGPU<strict_fp_t> silo_null, const strict_fp_t subtract_val):
      gpu_random_idx(random_idx),
      gpu_silo_null(silo_null),
      gpu_subtract_val(subtract_val)
   {}

   SYCL_EXTERNAL void operator() (sycl::nd_item<3> itm) const;

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      GDF::transfer_vars_to_gpu_impl<N>(gpu_random_idx, gpu_silo_null, gpu_subtract_val);
   }

private:
   const size_t gpu_random_idx;
   mutable CellGPU<strict_fp_t> gpu_silo_null;
   const strict_fp_t gpu_subtract_val;
};

#endif // TESTS_KERNELS_h
