#include "gpu_api_functions.h"
#include <random>
#include <fstream>
#include <iostream>

const uint64_t vec_size = 1 << 21;
const strict_fp_t tolerance = 1e-06;
const uint64_t max_iter = 1000;
std::ofstream log_file;

static void setup_problem();
static void finalize_problem();
static strict_fp_t f_x(const strict_fp_t& x);
static strict_fp_t f_dash_x(const strict_fp_t& x);
static void compute_error_norm();
static void compute_x_n_plus_1_async();
static void write_output(const uint64_t iter);

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
void demonstrate_temp_write(bool run)
{
   if(!run || (rank != 0))
      return;

   setup_problem();

   Vector<strict_fp_t> x = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("solution_vector");
   Vector<strict_fp_t> x_prev = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("prev_solution_vector");
   Parameter<strict_fp_t> err_nrm = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::PARAMETER>("Error_norm");
   VectorRead<strict_fp_t> fx = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("solution_vector");

   uint64_t iter = 0;

   compute_error_norm();

   while((err_nrm >= tolerance) && (iter <= max_iter))
   {
      GDF::memcpy_gpu_var(x_prev, x);
      GDF::transfer_to_cpu_copy(fx);
      compute_x_n_plus_1_async();
      write_output(iter);
      GDF::transfer_to_gpu_noinit(fx);

      GDF::gpu_barrier();
      compute_error_norm();

      iter++;
   }
   log_progress("Converged in " + std::to_string(iter) + " iteration(s) with L2 error norm = " + std::to_string(err_nrm));
   write_output(iter);
}

static void setup_problem()
{
   std::random_device rand_dev; // get a random seed from the OS
   std::mt19937 generator(rand_dev()); // Mersenne Twister engine seeded
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
         gpu_x[kk] = gpu_x_prev[kk] - (gpu_fx[kk]/f_dash_x(gpu_x_prev[kk]));
         gpu_fx[kk] = f_x(gpu_x[kk]);
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

static void write_output(const uint64_t iter)
{
   Vector<strict_fp_t> fx = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("solution_vector");
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