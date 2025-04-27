#include "oneMathSPMV.h"
#include "gpu_api_functions.h"

namespace GDF
{

oneMathSPMV::oneMathSPMV():
   m_que(GDF::get_gpu_queue()),
   alpha(0.0),
   beta(0.0),
   mat_op(oneapi::math::transpose::nontrans),
   alg(oneapi::math::sparse::spmv_alg::default_alg),
   A_handle(nullptr),
   x_handle(nullptr),
   y_handle(nullptr),
   descr(nullptr),
   workspace(nullptr)
{}

void oneMathSPMV::init_system(const int64_t nrows, const int64_t ncols, const int64_t nnz, const double alpha_in, const double beta_in, int * const ia,
                              int * const ja, double * const matval, double * const vec, double * const result)
{
   // The system should not be setup before this
   assert(!is_setup());

   // Set the scalar coefficients
   alpha = alpha_in;
   beta = beta_in;

   oneapi::math::sparse::init_csr_matrix(m_que, &A_handle, nrows, ncols, nnz,
                                       oneapi::math::index_base::zero, ia, ja, matval);
   
   // rocSPARSE backend requires that the property sorted is set when using matrices in CSR format.
   // Setting this property is also the best practice to get best performance.
   oneapi::math::sparse::set_matrix_property(m_que, A_handle,
                                             oneapi::math::sparse::matrix_property::sorted);
   
   // Create and initialize dense vector handles
   oneapi::math::sparse::init_dense_vector(m_que, &x_handle, ncols, vec);
   oneapi::math::sparse::init_dense_vector(m_que, &y_handle, nrows, result);
   
   // Create operation descriptor
   oneapi::math::sparse::init_spmv_descr(m_que, &descr);
   
   // Allocate external workspace
   std::size_t workspace_size = 0;
   oneapi::math::sparse::spmv_buffer_size(m_que, mat_op, &alpha, A_view, A_handle, x_handle,
                                          &beta, y_handle, alg, descr, workspace_size);
   workspace = GDF::malloc_gpu_var<void>(workspace_size);

   // Optimize spmv
   oneapi::math::sparse::spmv_optimize(m_que, mat_op, &alpha, A_view, A_handle, x_handle,
                                       &beta, y_handle, alg, descr, workspace);

   GDF::gpu_barrier();
}

void oneMathSPMV::update_matrix(const int64_t nrows, const size_t ncols, const size_t nnz, int * const ia, int * const ja, double * const matval)
{
   oneapi::math::sparse::set_csr_matrix_data(m_que, A_handle, nrows, ncols, nnz, oneapi::math::index_base::zero, ia, ja, matval);
   GDF::gpu_barrier();
}

void oneMathSPMV::update_x(const int64_t size, double* const value)
{
   oneapi::math::sparse::set_dense_vector_data(m_que, x_handle, size, value);
   GDF::gpu_barrier();
}

void oneMathSPMV::update_y(const int64_t size, double* const value)
{
   oneapi::math::sparse::set_dense_vector_data(m_que, y_handle, size, value);
   GDF::gpu_barrier();
}

void oneMathSPMV::compute()
{
   assert(A_handle);
   assert(x_handle);
   assert(y_handle);
   assert(descr);

   // Perform the matrix multiplication
   oneapi::math::sparse::spmv(m_que, mat_op, &alpha, A_view, A_handle, x_handle, &beta, y_handle, alg, descr);
   GDF::gpu_barrier();
}

void oneMathSPMV::release_system()
{
   assert(A_handle);
   assert(x_handle);
   assert(y_handle);
   assert(descr);

   oneapi::math::sparse::release_dense_vector(m_que, x_handle);
   oneapi::math::sparse::release_dense_vector(m_que, y_handle);
   oneapi::math::sparse::release_sparse_matrix(m_que, A_handle);
   oneapi::math::sparse::release_spmv_descr(m_que, descr);
   GDF::gpu_barrier();
   if(workspace) // Workspace can be nullptr for AMD gpus
      GDF::free_gpu_var(workspace);

   A_handle = nullptr;
   x_handle = nullptr;
   y_handle = nullptr;
   descr = nullptr;
   workspace = nullptr;
}

} // namespace GDF
