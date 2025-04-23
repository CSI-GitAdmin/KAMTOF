#ifndef ONEMATH_SPMV_H
#define ONEMATH_SPMV_H

#include "oneapi/math/sparse_blas.hpp" // Includes sycl header

/*
 * Perform sparse matrix - dense vector multiplication
 * y = (alpha * (mat_op(A) * x)) + (beta * y)
 * A -> Sparse matrix
 * x -> Dense vector to be multiplied with
 * y -> Solution dense vector
 * alpha,beta -> Scalar coefficients
 */

namespace GDF
{

struct oneMathSPMV
{
   oneMathSPMV();

   // Function to initialize the handles - does init, spmv_buffer_size and optimize
   void init_system(const int64_t nrows, const int64_t ncols, const int64_t nnz, const double alpha_in, const double beta_in, int* const ia,
                    int* const ja, double* const matval, double* const vec, double* const result);

   // Do the sparse matrix - dense vector multiplication
   void compute();

   // Reset the data
   void update_matrix(const int64_t nrows, const size_t ncols, const size_t nnz, int * const ia, int * const ja, double * const matval);
   void update_x(const int64_t size, double* const value);
   void update_y(const int64_t size, double* const value);

   // Free/Finalize the
   void release_system();

   // Check to see if the system is already setup
   bool is_setup()
   {
      return descr;
   }

   // SYCL QUEUE
   sycl::queue& m_que;

   // Scalar coefficients
   double alpha = 0.0;
   double beta = 0.0;

   // Sparse matrix data
   oneapi::math::transpose mat_op = oneapi::math::transpose::nontrans;
   oneapi::math::sparse::spmv_alg alg = oneapi::math::sparse::spmv_alg::default_alg;
   oneapi::math::sparse::matrix_view A_view;
   oneapi::math::sparse::matrix_handle_t A_handle = nullptr;

   // Dense vector data
   oneapi::math::sparse::dense_vector_handle_t x_handle = nullptr;
   oneapi::math::sparse::dense_vector_handle_t y_handle = nullptr;

   // System description
   oneapi::math::sparse::spmv_descr_t descr = nullptr;

   // Temproary workspace required by oneMath
   void* workspace = nullptr;
};

} // namespace GDF

#endif // ONEMATH_SPMV_H