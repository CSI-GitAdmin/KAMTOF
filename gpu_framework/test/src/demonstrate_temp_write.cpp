#include "gpu_api_functions.h"
#include <random>
#include <fstream>
#include <iostream>
#include <sys/time.h>

const uint64_t vec_size = 1 << 25;
const strict_fp_t tolerance = 1e-06;
const uint64_t max_iter = 10000;
std::ofstream log_file;

static void setup_problem();
static void finalize_problem();
static strict_fp_t f_x(const strict_fp_t& x);
static strict_fp_t f_dash_x(const strict_fp_t& x);
static void compute_error_norm();
static void compute_x_n_plus_1_async();
static void compute_x_n_plus_1();
static void write_output_on_cpu(const uint64_t iter);

/*
 * This function demonstrates the use of the TEMP_WRITE functionality
 *
 * We solve for e^(-x) - x = 0
 *
 * Let f(x) = e^(-x) - x
 *
 * f'(x) = -(e^(-x) + 1)
 *
 * To iteratively find the root, x_(n+1) = x_(n) - f(x_(n))/f'(x_(n))
 *
 * After every iteration, we write out the min and max value of f(x) and their mean on CPU as an output setup
 *
*/
void demonstrate_temp_write_impl(bool async)
{
   if(rank != 0)
      return;

   setup_problem();

   Vector<strict_fp_t> x = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("solution_vector");
   Vector<strict_fp_t> x_prev = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("prev_solution_vector");
   Parameter<strict_fp_t> err_nrm = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::PARAMETER>("Error_norm");
   VectorRead<strict_fp_t> fx = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("function_eval_of_solution_vector");

   assert(x.exists() && x_prev.exists() && fx.exists() && err_nrm.exists());

   uint64_t iter = 0;

   struct timeval s_time, e_time;

   gettimeofday(&s_time, NULL);

   // Compute the error norm for the inital solution (f(x) is the error in this case)
   compute_error_norm();

   while((err_nrm >= tolerance) && (iter <= max_iter)) // Iterate until we reach tolerance (or) max_iters
   {
      // Set x_(n) = x_(n+1)
      GDF::memcpy_gpu_var(x_prev, x);

      // Copy the values of f(x_(n)) to the CPU for writing output. This sets the cpu_data_status to TEMP_WRITE for the variable 'fx'
      GDF::transfer_to_cpu_copy(fx);

      // Compute the values for x_(n+1) and f(x_(n+1))
      if(async)
      {
         // The CPU thread will MOVE ON to "write_output_on_cpu" immediately
         compute_x_n_plus_1_async();
      }
      else
      {
         // The CPU thread will WAIT until this call is complted to move on to "write_output_on_cpu"
         compute_x_n_plus_1();
      }

      // Normalize and write out the error to Newton_iter.log file
      write_output_on_cpu(iter);

      // Move the ownership of the 'fx' variable back to GPU without copying the data over as it was on TEMP_WRITE status
      GDF::transfer_to_gpu_move(fx);

      // Wait for the ASYNC operation to finish before computing the error norm
      GDF::gpu_barrier();

      compute_error_norm();

      iter++;
   }
   gettimeofday(&e_time, NULL);

   write_output_on_cpu(iter);

   const strict_fp_t time_elapsed_cpu = (e_time.tv_sec - s_time.tv_sec) + ((e_time.tv_usec - s_time.tv_usec) * 1e-6);

   log_progress("Converged in " + std::to_string(iter) + " iteration(s) with L2 error norm = " + std::to_string(err_nrm));
   if(async)
      log_progress("Time taken to for the Newton-Ralphson iterations (WITH async): " + std::to_string(time_elapsed_cpu) + " seconds.");
   else
      log_progress("Time taken to for the Newton-Ralphson iterations (WITHOUT aysnc): " + std::to_string(time_elapsed_cpu) + " seconds.");
}

static void setup_problem()
{
   std::mt19937 generator(42); // Mersenne Twister engine seeded (Use the same seed to compare runtimes)
   std::uniform_real_distribution<strict_fp_t> distribution(-10.0, 10.0); // range [-10.0, 10.0]

   Vector<strict_fp_t> x = m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("solution_vector");
   Vector<strict_fp_t> x_prev = m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("prev_solution_vector");
   Parameter<strict_fp_t> err_nrm = m_silo.register_entry<strict_fp_t, CDF::StorageType::PARAMETER>("Error_norm");
   Vector<strict_fp_t> fx = m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("function_eval_of_solution_vector");

   x.resize(vec_size);
   x_prev.resize(vec_size);
   fx.resize(vec_size);

   err_nrm = 0;

   for(size_t kk = 0; kk < vec_size; kk++)
   {
      x[kk] = distribution(generator);
      fx[kk] = f_x(x[kk]);
   }

   log_file.open("Newton_iters.log");
}

static void finalize_problem()
{
   log_file.close();
}

static inline strict_fp_t f_x(const strict_fp_t& x)
{
   return (sycl::exp(-1.0 * x) - x);
}

static inline strict_fp_t f_dash_x(const strict_fp_t& x)
{
   return ((-1.0*sycl::exp(-1.0 * x)) - 1.0);
}

static void compute_error_norm()
{
   Parameter<strict_fp_t> err_nrm = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::PARAMETER>("Error_norm");
   VectorRead<strict_fp_t> fx = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("function_eval_of_solution_vector");

   GDF::transfer_to_gpu_noinit(err_nrm);
   GDF::transfer_to_gpu_readonly(fx);
   GDF::l2_norm(vec_size, fx.gpu_data(), err_nrm.gpu_data());
}

class kg_compute_x_n_plus_1
{
public:
   kg_compute_x_n_plus_1(VectorGPU<strict_fp_t> x, VectorGPURead<strict_fp_t> x_prev, VectorGPU<strict_fp_t> fx):
      gpu_x(x),
      gpu_x_prev(x_prev),
      gpu_fx(fx)
   {}

   void operator() (sycl::nd_item<3> itm) const
   {
      size_t idx = GDF::get_1d_index(itm);
      size_t stride = GDF::get_1d_stride(itm);
      for(size_t kk = idx; kk < gpu_x.size(); kk += stride)
      {
         gpu_x[kk] = gpu_x_prev[kk] - (f_x(gpu_x_prev[kk])/f_dash_x(gpu_x_prev[kk]));
         gpu_fx[kk] = f_x(gpu_x_prev[kk]);
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      GDF::transfer_vars_to_gpu_impl<N>(gpu_x, gpu_x_prev, gpu_fx);
   }

private:
   mutable VectorGPU<strict_fp_t> gpu_x;
   VectorGPURead<strict_fp_t> gpu_x_prev;
   mutable VectorGPU<strict_fp_t> gpu_fx;
};

static void compute_x_n_plus_1_async()
{
   Vector<strict_fp_t> x = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("solution_vector");
   Vector<strict_fp_t> fx = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("function_eval_of_solution_vector");
   VectorRead<strict_fp_t> x_prev = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("prev_solution_vector");

   GDF::submit_to_gpu_async<kg_compute_x_n_plus_1>(x, x_prev, fx);
}

static void compute_x_n_plus_1()
{
   Vector<strict_fp_t> x = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("solution_vector");
   Vector<strict_fp_t> fx = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("function_eval_of_solution_vector");
   VectorRead<strict_fp_t> x_prev = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("prev_solution_vector");

   GDF::submit_to_gpu<kg_compute_x_n_plus_1>(x, x_prev, fx);
}

static void write_output_on_cpu(const uint64_t iter)
{
   Vector<strict_fp_t> fx = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("function_eval_of_solution_vector");
   ParameterRead<strict_fp_t> err_nrm = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::PARAMETER>("Error_norm");

   std::string it_out("Iteration " + std::to_string(iter) + " || L2 norm of error = " + std::to_string(err_nrm));

   strict_fp_t max_error = -1.0*std::numeric_limits<strict_fp_t>::max();
   strict_fp_t normalized_avg_error = 0.0;
   for(size_t kk = 0; kk < fx.size(); kk++)
   {
      if(max_error < fx[kk])
         max_error = fabs(fx[kk]);
   }
   for(size_t kk = 0; kk < fx.size(); kk++)
   {
      fx[kk] = fx[kk] / max_error;
      normalized_avg_error += fx[kk];
   }
   normalized_avg_error = normalized_avg_error / fx.size();

   log_file << it_out << std::endl;
   log_file << "Normalized_avg_error = " << normalized_avg_error << std::endl;
}
