#include "test_funcs.h" // For function defs
#include "test_kernels.h" // For kernels
#include "gpu_globals.h" // For gpu_manager
#include "gpu_silo_fwd.h" // For Cell/CellRead etc.
#include "gpu_api_functions.h" // For GPU API functions
#include "silo.h"

#include <string> // For std::to_string
#include <random> // For random number generation
#include <cstdlib>  // for rand()
#include <ctime>    // for time()
#include <set>      // for std::set
#include <unordered_set>
#include <tuple>
#include <algorithm>


bool a_not_equal_b(const strict_fp_t& a, const strict_fp_t& b, const strict_fp_t& tol_percent)
{
   if(a == b) // If both are exactly equal to zero (or) specific strict_fp_t, no need to calculate percentage error
      return false;
   strict_fp_t err = a - b;
   strict_fp_t percent_err[2];
   if(a != 0.0)
   {
      percent_err[0] = fabs(100* (err/a));
   }
   if(b != 0.0)
   {
      percent_err[1] = fabs(100* (err/b));
   }
   percent_err[0] = (percent_err[0] >= percent_err[1]) ? percent_err[0] : percent_err[1];

   return (percent_err[0] <= tol_percent) ? false : true;
}

void setup_gpu_globals_test()
{
   // Setup GPU Manager
   assert(!gpu_manager);
   gpu_manager = new GDF::GPUManager_t(GDF::sycl_device_t::DEFAULT,
                                       {1, 1, 1024 },
                                       {1, 1,  256 } );

   log_msg("Device type selected : DEFAULT");
}

void finalize_gpu_globals_test()
{
   assert(gpu_manager);
   GDF::gpu_barrier();

   // Finalize GPU Manager
   delete gpu_manager;
   gpu_manager = nullptr;
}

void test_dss_gpu()
{
   // ***------- Check a kernel with a SILO vars -------*** //
   const strict_fp_t n = 1.0;
   const strict_fp_t R = 8.314;
   Cell<strict_fp_t> pressure = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Pressure");
   CellRead<strict_fp_t> volume = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Volume");
   Cell<strict_fp_t> temperature = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Temperature");
   const strict_fp_t init_temp = temperature[0];
   Parameter<strict_fp_t> scale = m_silo.register_entry<strict_fp_t, CDF::StorageType::PARAMETER>("Pressure_scaling_factor");
   scale[0] = R;

   const strict_fp_t pr_val = ((n * R * init_temp)/volume[0]) * scale[0];

   GDF::submit_to_gpu<kg_compute_pressure>(pressure, volume, n, R, temperature);

#ifdef GPU_DEVELOP // If GPU_DEVELOP mode is ON, we do not need to explicitly bring the data back
   pressure[4] = volume[4];
   pressure[4] = ((n * R * init_temp)/volume[0]);
#endif

   GDF::submit_to_gpu<kg_scale_pressure_and_change_scale>(pressure, scale);

   GDF::submit_to_gpu<kg_compute_temperature>(temperature, pressure, n, R, volume);

#ifndef GPU_DEVELOP
   GDF::transfer_to_cpu_move(pressure, temperature, volume);
#endif

   // Check if the values are correct
   for(int ii = 0; ii < pressure.size(); ii++)
   {
      assert(pressure.size() ==  temperature.size());
      if(a_not_equal_b(pressure[ii], pr_val, tol))
      {
         std::string error = "Pressure[" + std::to_string(ii) + "] = " + std::to_string(pressure[ii]) + " instead of " + std::to_string(pr_val) +
                             ". Kernel compute_pressure that does PV=nRT on GPU did NOT match CPU values!";
         log_msg<CDF::LogLevel::ERROR>(error);
      }
      if(a_not_equal_b(temperature[ii], init_temp*scale[0], tol))
      {
         std::string error = "Temperature[" + std::to_string(ii) + "] = " + std::to_string(temperature[ii]) + " instead of " + std::to_string(init_temp*scale[0]) +
                             ". Kernel compute_temperature that does PV=nRT on GPU did NOT match CPU values!";
         log_msg<CDF::LogLevel::ERROR>(error);
      }
   }
#ifndef NDEBUG
   log_msg("Kernel compute_pressure and compute_temperature that does P=nRT/V on GPU matched CPU values!");
#endif

   // ***------- Check a kernel without a SILO var -------*** //
   GDF::submit_to_gpu<kg_test_kernel>(10, pr_val);
#ifndef NDEBUG
   log_msg("Kernel without SILO variables executed successfully!");
#endif

   // ***------- Check the operators for multi-dimensional data in DSSGPU, Also a check of update_gpu_offsets -------*** //
   const strict_fp_t vel_mag = 14.0; // 1^2 + 2^2 + 3^2
   const uint8_t vel_shape[1] = {3};
   Cell<strict_fp_t, 1> velocity = m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL, 1>("velocity", vel_shape);

   GDF::submit_to_gpu<kg_set_initial_condition>(velocity, 1.0, 2.0, 3.0);

#ifndef GPU_DEVELOP
   GDF::transfer_to_cpu_copy(velocity);
#endif

   // Check if the values are correct
   for(int ii = 0; ii < velocity.size(); ii++)
   {
      strict_fp_t vel_mag_kk = pow(velocity(ii,0), 2) + pow(velocity(ii,1), 2) + pow(velocity(ii,2), 2);
      if(a_not_equal_b(vel_mag_kk, vel_mag, tol))
      {
         std::string error = "VelocityMagnitude[" + std::to_string(ii) + "] = " + std::to_string(vel_mag_kk) + "instead of " + std::to_string(vel_mag) +
                             ". Operator() check for multi-dimensional GPU data FAILED!";
         log_msg<CDF::LogLevel::ERROR>(error);
      }
   }
   #ifndef NDEBUG
   log_progress("Operator() check for multi-dimensional GPU data passed!");
   #endif
}

void test_dss_gpu_resize()
{
   const strict_fp_t n = 2.0;
   const strict_fp_t R = 8.314;
   const strict_fp_t pr_val = (n * R * 300)/(2e-02);

   Cell<strict_fp_t> pressure = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Pressure");
   Cell<strict_fp_t> volume = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Volume");
   Cell<strict_fp_t> temperature= m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Temperature");

   const int new_cell_count = 512;

   m_silo.resize<CDF::StorageType::CELL>(new_cell_count);

   GDF::transfer_to_cpu_noinit(pressure, volume, temperature);
   for(int ii = 0; ii < pressure.size(); ii++)
   {
      assert(pressure.size() ==  volume.size() && pressure.size() == temperature.size());
      pressure[ii] = 101325.00;
      volume[ii] = 2e-02;
      temperature[ii] = 300.00;
   }

   GDF::set_gpu_global_local_range({1,1,new_cell_count}, {1,1,new_cell_count/16});

   GDF::transfer_to_gpu_noinit(pressure);

   GDF::submit_to_gpu<kg_compute_pressure>(pressure, volume, n, R, temperature);

#ifndef GPU_DEVELOP
   GDF::transfer_to_cpu_move(pressure, volume, temperature);
#endif

   // Check if the values are correct
   for(int ii = 0; ii < pressure.size(); ii++)
   {
      if(a_not_equal_b(pressure[ii], pr_val, tol))
      {
         std::string error = "Pressure[" + std::to_string(ii) + "] = " + std::to_string(pressure[ii]) + "instead of " + std::to_string(pr_val) +
                             ". DSS GPU resize check FAILED!";
         log_error(error);
      }
   }
#ifndef NDEBUG
   log_progress("DSS GPU resize check passed!");
#endif
}

void test_gpu_pointer_api_funcs()
{
   const size_t num_elements = 1024;
   const strict_fp_t val = 5.365;
   std::vector<strict_fp_t> m_cpu_var(num_elements, val);

   // GPU Compute Answer malloc_Device
   strict_fp_t* m_gpu_var = GDF::malloc_gpu_var<strict_fp_t>(num_elements); // Allocate GPU variable
   strict_fp_t gpu_result_device = 0.0;
   GDF::memcpy_gpu_var(m_gpu_var, m_cpu_var.data(), num_elements);
   GDF::set_gpu_global_local_range({1,1,512},{1,1,512});
   GDF::submit_to_gpu<kg_norm2>(m_gpu_var, num_elements);
   GDF::memcpy_gpu_var(&gpu_result_device, m_gpu_var, 1);
   GDF::free_gpu_var(m_gpu_var);

   // GPU Compute Answer malloc_shared
   m_gpu_var = GDF::malloc_gpu_var<strict_fp_t, true>(num_elements);
   for(size_t ii = 0; ii < num_elements; ii++)
   {
      m_gpu_var[ii] = m_cpu_var[ii];
   }
   GDF::submit_to_gpu<kg_norm2>(m_gpu_var, num_elements);
   strict_fp_t gpu_result_shared = 0.0;
   gpu_result_shared = m_gpu_var[0];
   GDF::free_gpu_var(m_gpu_var);

   // CPU Compute Answer
   strict_fp_t cpu_local_result = 0.0;
   for(size_t ii = 0; ii < num_elements; ii++)
   {
      cpu_local_result += pow(m_cpu_var[ii], 2);
   }
   cpu_local_result = sqrt(cpu_local_result);

   // Check results
   if(a_not_equal_b(cpu_local_result, gpu_result_device, tol))
   {
      std::string error = "gpu_result_device = " + std::to_string(gpu_result_device) + " instead of " + std::to_string(cpu_local_result) +
                          ". Raw pointer API functions check FAILED while using malloc_device!";
      log_msg<CDF::LogLevel::ERROR>(error);
   }
   if(a_not_equal_b(cpu_local_result, gpu_result_shared, tol))
   {
      std::string error = "gpu_result_shared = " + std::to_string(gpu_result_shared) + " instead of " + std::to_string(cpu_local_result) +
                          ". Raw pointer API functions check FAILED while using malloc_shared!";
      log_msg<CDF::LogLevel::ERROR>(error);
   }
#ifndef NDEBUG
   log_msg("Raw pointer API functions check passed!");
#endif
}

void test_gpu_atomics()
{
   const size_t num_elements = 1024;
   const strict_fp_t range[2]{0.0,100.0};
   std::vector<strict_fp_t> m_cpu_var(num_elements);

   // Initialize a random number generator with the seed from the random device
   std::random_device rd;
   std::mt19937 generator(rd());
   std::uniform_real_distribution<strict_fp_t> distribution(range[0], range[1]);


   // Generate and print random strict_fp_ts
   for (int ii = 0; ii < num_elements; ++ii)
   {
      m_cpu_var[ii] = distribution(generator);
   }

   // CPU Compute Answer
   strict_fp_t cpu_local_result[4]{0.0, (num_elements*range[1])/2.0, range[0] - 1.0, range[1] + 1.0};
   for(size_t ii = 0; ii < num_elements; ii++)
   {
      cpu_local_result[0] += m_cpu_var[ii];

      cpu_local_result[1] -= m_cpu_var[ii];

      cpu_local_result[2] = (m_cpu_var[ii] >= cpu_local_result[2]) ? m_cpu_var[ii] : cpu_local_result[2];

      cpu_local_result[3] = (m_cpu_var[ii] <= cpu_local_result[3]) ? m_cpu_var[ii] : cpu_local_result[3];
   }

   // GPU Compute Answer
   strict_fp_t* m_gpu_var = GDF::malloc_gpu_var<strict_fp_t>(num_elements);
   strict_fp_t* gpu_result = GDF::malloc_gpu_var<strict_fp_t, true>(4);

   gpu_result[0] = 0.0;
   gpu_result[1] = (num_elements*range[1])/2.0;
   gpu_result[2] = range[0] - 1.0;
   gpu_result[3] = range[1] + 1.0;
   GDF::memcpy_gpu_var(m_gpu_var, m_cpu_var.data(), num_elements);

   GDF::submit_to_gpu<kg_atomics>(gpu_result, m_gpu_var, num_elements);


   // Check results
   for(size_t ii = 0; ii < 4; ii++)
   {
      if(a_not_equal_b(cpu_local_result[ii], gpu_result[ii], 1e-08))
      {
         std::string error = "gpu_result[" + std::to_string(ii) + "] = " + std::to_string(gpu_result[ii]) + " instead of "
                             + std::to_string(cpu_local_result[ii]) + ". Atomic API functions check FAILED";
         log_msg<CDF::LogLevel::ERROR>(error);
      }
   }

   GDF::free_gpu_var(m_gpu_var);
   GDF::free_gpu_var(gpu_result);

#ifndef NDEBUG
   log_msg("Atomic API functions check passed!");
#endif
}

// Function to generate a random CSR matrix
int generate_random_CSR(int rows, int cols, strict_fp_t non_zero_percentage, std::vector<strict_fp_t>& values,
                        std::vector<int>& col_indices, std::vector<int>& row_ptr, std::vector<strict_fp_t>& vec)
{
   // Calculate total number of elements in the matrix
   int total_elements = rows * cols;

   // Calculate the total number of non-zero elements based on the given percentage
   int nnz = static_cast<int>(non_zero_percentage * total_elements / 100);

   assert((rows > 0) && (cols > 0) && (nnz > 0));

   int min_dim = std::min(rows, cols);

   std::vector<std::tuple<int, int, strict_fp_t>> entries;
   entries.reserve(nnz);

   std::unordered_set<int> used_global_idx;
   used_global_idx.reserve(nnz);

   // The diagonal value in every row is non zero
   // First set all the diagonal entries to a random value
   for(int ii = 0; ii < min_dim; ii++)
   {
      entries.push_back({ii, ii, static_cast<strict_fp_t>(std::rand()) / RAND_MAX});
      used_global_idx.insert((ii * cols) + ii);
   }

   int remaining_nnz = nnz - min_dim;
   while(remaining_nnz > 0)
   {
      int cur_row = std::rand() % rows;
      int cur_col = std::rand() % cols;
      if(used_global_idx.insert((cur_row * cols) + cur_col).second)
      {
         entries.push_back({cur_row, cur_col, static_cast<strict_fp_t>(std::rand()) / RAND_MAX});
         remaining_nnz--;
      }
   }
   std::sort(entries.begin(), entries.end());

   // Fill CSR format arrays
   row_ptr.resize(rows+1, 0);
   col_indices.resize(nnz, 0);
   values.resize(nnz, 0);
   int current_row = 0;
   int cummulative_nnz = 0;
   for(std::tuple<int, int, strict_fp_t>& cur_entry : entries)
   {
      if(std::get<0>(cur_entry) == current_row)
      {
         cummulative_nnz++;
      }
      else
      {
         row_ptr[current_row+1] = cummulative_nnz++;
         current_row++;
      }

      col_indices[cummulative_nnz-1] = std::get<1>(cur_entry);
      values[cummulative_nnz-1] = std::get<2>(cur_entry);
   }

   row_ptr[rows] = cummulative_nnz;

   vec.reserve(rows);
   for(int ii = 0; ii < rows; ii++)
   {
      vec.push_back(static_cast<strict_fp_t>(std::rand()) / RAND_MAX);
   }

   return nnz;
}

void test_silo_null()
{
   std::string silo_str = "RANDOM_STRING_FOR_SILO_VARIABLE";
   const size_t random_idx = 10;
   const strict_fp_t init_val = 43.35;
   const strict_fp_t subtract_val = 10.33;
   Cell<strict_fp_t> silo_null = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>(silo_str);
   assert(!silo_null.exists());  // The variable should not exist

   // The variable can be safely passed on to the transfer call
   GDF::submit_to_gpu<kg_silo_null>(random_idx, silo_null, subtract_val);
#ifndef GPU_DEVELOP
   GDF::transfer_to_cpu_move(silo_null);
#endif
   CellRead<strict_fp_t> silo_null_registered = m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>(silo_str);

   assert(silo_null_registered.exists() && silo_null.exists());
   silo_null[random_idx] = init_val;

   GDF::submit_to_gpu<kg_silo_null>(random_idx, silo_null, subtract_val);

#ifndef GPU_DEVELOP
   GDF::transfer_to_cpu_move(silo_null);
#endif

   // Check results
   if(a_not_equal_b(silo_null[random_idx], (init_val - subtract_val), tol))
   {
      std::string error = std::string("silo_null[") + std::to_string(random_idx) + std::string("] = ") + std::to_string(silo_null[random_idx])
      + std::string(" instead of ") + std::to_string(init_val - subtract_val) + std::string(". GPU SILO NULL check FAILED");
      log_error(error);
   }
#ifndef NDEBUG
   log_progress("GPU SILO NULL check passed!");
#endif

}

void test_ncpu_ngpu()
{
   const uint64_t vec_size = 1024;

   std::random_device rd;
   std::mt19937 generator(rd());
   std::uniform_real_distribution<strict_fp_t> distribution(rank, rank+1);

   // Generate and print random strict_fp_ts
   std::vector<strict_fp_t> my_vec_cpu;
   strict_fp_t cpu_local_result = 0;
   strict_fp_t cpu_global_result = 0;
   my_vec_cpu.reserve(vec_size);
   for(uint64_t ii = 0; ii < vec_size; ii++)
   {
      my_vec_cpu.push_back(distribution(generator));
      cpu_local_result += (my_vec_cpu[ii]*my_vec_cpu[ii]);
   }
   MPI_Allreduce(&cpu_local_result, &cpu_global_result, 1, MPI_STRICT_FP_T, MPI_SUM, MPI_COMM_WORLD);

   log_msg("CPU results : total = " + std::to_string(cpu_local_result) +" and the gloabl total = " + std::to_string(cpu_global_result));

   // Allocate two variables on different ranks
   strict_fp_t* local_result = GDF::malloc_gpu_var<strict_fp_t, true>(1);
   strict_fp_t* global_result = GDF::malloc_gpu_var<strict_fp_t, true>(1);
   strict_fp_t* my_vec = GDF::malloc_gpu_var<strict_fp_t>(vec_size);

   GDF::memcpy_gpu_var(my_vec, my_vec_cpu.data(), vec_size);
   GDF::dot_product(vec_size, my_vec, my_vec, local_result);
   MPI_Allreduce(local_result, global_result, 1, MPI_STRICT_FP_T, MPI_SUM, MPI_COMM_WORLD);

   log_msg("GPU results : local total = " + std::to_string(local_result[0]) +" and the global total = " + std::to_string(global_result[0]));

   if(a_not_equal_b(cpu_global_result, global_result[0], tol))
   {
      log_msg<CDF::LogLevel::ERROR>("Error in GPU MPI_Allreduce test!");
   }

   GDF::free_gpu_var(local_result);
   GDF::free_gpu_var(global_result);
   GDF::free_gpu_var(my_vec);
}
