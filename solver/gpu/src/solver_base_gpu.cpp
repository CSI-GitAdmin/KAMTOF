#include <vector>
#include <cmath>
#include <cstdio>
#include <set>

#include "mpiclass.h"
#include <mpi.h>

#include "grid.h"
#include "solver_gpu.h"
#include "silo.h"
#include "silo_fwd.h"
#include "fp_data_types.h"

#include "gpu_api_functions.h"
#include "oneMathSPMV.h"
#include "oneapi/math/blas.hpp"

static void mpi_nbnb_transfer_gpu(strict_fp_t *vec);

Solver_base_gpu::Solver_base_gpu()
{
   residual_norm = 1e30;

   m_spmv_sys = new GDF::oneMathSPMV();
}

void Solver_base_gpu::allocate_variables()
{
   m_silo.register_entry<int, CDF::StorageType::VECTOR>("ia_local");
   m_silo.register_entry<int, CDF::StorageType::VECTOR>("ja_local");
   m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("A_data_local");

   m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("rhs_local");
   m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_local");
   m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_old_local");
   m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("Q_cell_local");
   m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("residual_local");
   m_silo.register_entry<int, CDF::StorageType::CELL>("csr_diag_idx_local");

   m_silo.register_entry<strict_fp_t, CDF::StorageType::FACE>("rdista_local");
   m_silo.register_entry<int, CDF::StorageType::FACE>("csr_idx_local");

   m_silo.register_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_rdista_local");
   m_silo.register_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("Q_boundary_local");
}

class kg_set_boundary_conditions
{
public:
   kg_set_boundary_conditions(const int start_idx, const int end_idx, const strict_fp_t QL, BoundaryGPU<strict_fp_t>& Q_boundary):
      gpu_start_idx(start_idx),
      gpu_end_idx(end_idx),
      gpu_QL(QL),
      gpu_Q_boundary(Q_boundary)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for(int i = gpu_start_idx+idx; i < gpu_end_idx; i += stride)
      {
         gpu_Q_boundary[i] = gpu_QL;
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_start_idx, gpu_end_idx, gpu_QL, gpu_Q_boundary);
   }

private:
   const int gpu_start_idx;
   const int gpu_end_idx;
   const strict_fp_t gpu_QL;
   mutable BoundaryGPU<strict_fp_t> gpu_Q_boundary;
};

void Solver_base_gpu::set_boundary_conditions (const strict_fp_t QL, const strict_fp_t QR,
                                          const strict_fp_t QB, const strict_fp_t QT)
{
   VectorRead<int> boundary_type_start_and_end_index = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("boundary_type_start_and_end_index");
   Boundary<strict_fp_t> Q_boundary_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("Q_boundary_local");

   // Left boundary
   GDF::submit_to_gpu<kg_set_boundary_conditions>(boundary_type_start_and_end_index[2],
                                                  boundary_type_start_and_end_index[3],
                                                  QL,
                                                  Q_boundary_local);
   
   // Right boundary
   GDF::submit_to_gpu<kg_set_boundary_conditions>(boundary_type_start_and_end_index[3],
                                                  boundary_type_start_and_end_index[4],
                                                  QR,
                                                  Q_boundary_local);
   
   // Bottom boundary
   GDF::submit_to_gpu<kg_set_boundary_conditions>(boundary_type_start_and_end_index[0],
                                                  boundary_type_start_and_end_index[1],
                                                  QB,
                                                  Q_boundary_local);
                                                  
   // Top boundary
   GDF::submit_to_gpu<kg_set_boundary_conditions>(boundary_type_start_and_end_index[1],
                                                  boundary_type_start_and_end_index[2],
                                                  QT,
                                                  Q_boundary_local);
}

class kg_initialize_solution
{
public:
   kg_initialize_solution(const int num_solved,
                          const strict_fp_t Q_initial,
                          CellGPU<strict_fp_t>& Q_cell):
      gpu_num_solved(num_solved),
      gpu_Q_initial(Q_initial),
      gpu_Q_cell(Q_cell)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for(int i = idx; i < gpu_num_solved; i += stride)
      {
         gpu_Q_cell[i] = gpu_Q_initial;
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_solved, gpu_Q_initial, gpu_Q_cell);
   }

private:
   const int gpu_num_solved;
   const strict_fp_t gpu_Q_initial;
   mutable CellGPU<strict_fp_t> gpu_Q_cell;
};

void Solver_base_gpu::initialize_solution (const int num_solved, const strict_fp_t Q_initial)
{
   Cell<strict_fp_t> Q_cell_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Q_cell_local");

   GDF::submit_to_gpu<kg_initialize_solution>(num_solved,
                                              Q_initial,
                                              Q_cell_local);

   mpi_nbnb_transfer_gpu(Q_cell_local.gpu_data());
}

class kg_update_solution_explicit
{
public:
   kg_update_solution_explicit(const int num_solved,
                               const strict_fp_t delta_t,
                               CellGPURead<strict_fp_t>& volume,
                               CellGPURead<strict_fp_t>& residual,
                               CellGPU<strict_fp_t>& Q_cell):
      gpu_num_solved(num_solved),
      gpu_delta_t(delta_t),
      gpu_volume(volume),
      gpu_residual(residual),
      gpu_Q_cell(Q_cell)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (unsigned int i = idx; i < gpu_num_solved; i += stride)
      {
         gpu_Q_cell[i] -= gpu_residual[i] * gpu_delta_t / gpu_volume[i];
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_solved, gpu_delta_t,
      gpu_volume, gpu_residual, gpu_Q_cell);
   }

private:
   const int gpu_num_solved;
   const strict_fp_t gpu_delta_t;
   CellGPURead<strict_fp_t> gpu_volume;
   CellGPURead<strict_fp_t> gpu_residual;
   mutable CellGPU<strict_fp_t> gpu_Q_cell;
};

class kg_update_solution_add_dQ_to_Q_cell
{
public:
   kg_update_solution_add_dQ_to_Q_cell(const int num_solved,
                                       CellGPURead<strict_fp_t>& dQ,
                                       CellGPU<strict_fp_t>& Q_cell):
      gpu_num_solved(num_solved),
      gpu_dQ(dQ),
      gpu_Q_cell(Q_cell)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (unsigned int i = idx; i < gpu_num_solved; i += stride)
      {
         gpu_Q_cell[i] += gpu_dQ[i];
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_solved, gpu_dQ, gpu_Q_cell);
   }

private:
   const int gpu_num_solved;
   CellGPURead<strict_fp_t> gpu_dQ;
   mutable CellGPU<strict_fp_t> gpu_Q_cell;
};

void Solver_base_gpu::update_solution(const int num_solved)
{
   Cell<strict_fp_t> Q_cell_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Q_cell_local");
   
   if(implicit_solver == false)
   {
      CellRead<strict_fp_t> volume_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("volume_local");
      CellRead<strict_fp_t> residual_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("residual_local");

      GDF::submit_to_gpu<kg_update_solution_explicit>(num_solved,
                                                      delta_t,
                                                      volume_local,
                                                      residual_local,
                                                      Q_cell_local);
   }
   else
   {
      if(solver_type == 0)
      {
         jacobi_linear_solver(num_solved);
      }
      else
      {
         setup_m_spmv_system();

         bicgstab_linear_solver();
      }

      CellRead<strict_fp_t> dQ_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_local");
      
      GDF::submit_to_gpu<kg_update_solution_add_dQ_to_Q_cell>(num_solved,
                                                              dQ_local,
                                                              Q_cell_local);
   }

   mpi_nbnb_transfer_gpu(Q_cell_local.gpu_data());
}

void Solver_base_gpu::write_solution(const int num_solved, const int num_cells, std::string file_name)
{
   int* recv_count = nullptr;
   int* recv_disp = nullptr;
   strict_fp_t* xcen_total = nullptr;
   strict_fp_t* Q_cell_total = nullptr;

   if(rank == 0)
   {
      recv_disp = new int[numprocs];
      recv_count = new int[numprocs];
      xcen_total = new strict_fp_t[2 * num_cells];
      Q_cell_total = new strict_fp_t[num_cells];
   }

   MPI_Gather(&num_solved, 1, MPI_INT, recv_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

   if(rank == 0)
   {
      recv_disp[0] = 0;
      for(int i = 1; i < numprocs; i++)
      {
         recv_disp[i] = recv_disp[i-1] + recv_count[i-1];
      }
   }

   CellRead<strict_fp_t> Q_cell_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Q_cell_local");
   GDF::transfer_to_cpu_move(Q_cell_local);

   MPI_Gatherv(Q_cell_local.cpu_data(), num_solved, MPI_DOUBLE, Q_cell_total, recv_count, recv_disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if(rank == 0)
   {
      for(int i = 0; i < numprocs; i++)
      {
         recv_count[i] *= 2;
         recv_disp[i] *= 2;
      }
   }

   VectorRead<strict_fp_t> xcen_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("xcen_local");
   GDF::transfer_to_cpu_move(xcen_local);

   MPI_Gatherv(xcen_local.cpu_data(), 2*num_solved, MPI_DOUBLE, xcen_total, recv_count, recv_disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if(rank == 0)
   {
      FILE* file = fopen(file_name.c_str(), "w");

      for(int i = 0; i < num_cells; i++)
      {
         fprintf(file, "%0.16f, %.16f, %.16f\n", xcen_total[2*i], xcen_total[2*i+1], Q_cell_total[i]);
      }

      fclose(file);

      delete[] xcen_total;
      delete[] Q_cell_total;
   }
}

void Solver_base_gpu::write_residual(std::string file_name, std::vector<strict_fp_t>& residual_norm)
{
   FILE* file = fopen(file_name.c_str(), "w");

   for(int i = 0; i < residual_norm.size(); i++)
   {
      fprintf(file, "%0.16f\n", residual_norm[i]);
   }

   fclose(file);
}

class kg_jacobi_linear_solver
{
public:
   kg_jacobi_linear_solver(const int num_solved,
                           CellGPURead<strict_fp_t>& rhs,
                           VectorGPURead<strict_fp_t>& A_data,
                           VectorGPURead<int>& ia,
                           VectorGPURead<int>& ja,
                           CellGPURead<strict_fp_t>& dQ_old,
                           CellGPU<strict_fp_t>& dQ):
      gpu_num_solved(num_solved),
      gpu_rhs(rhs),
      gpu_A_data(A_data),
      gpu_ia(ia),
      gpu_ja(ja),
      gpu_dQ_old(dQ_old),
      gpu_dQ(dQ)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for(unsigned int i = idx; i < gpu_num_solved; i += stride)
      {
         strict_fp_t temp = gpu_rhs[i];
         // First entry is the diagonal.
         const strict_fp_t diag_value = gpu_A_data[gpu_ia[i]];
         // Transfer all the off diagonals to right hand side
         for (int j = gpu_ia[i] + 1; j < gpu_ia[i + 1]; j++)     // Skipping first one as it is the diagonal
         {
            temp -=  gpu_A_data[j] * gpu_dQ_old[gpu_ja[j]];
         }

         temp = temp / diag_value;

         gpu_dQ[i] = temp;
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_solved, gpu_rhs, gpu_A_data, gpu_ia, gpu_ja, gpu_dQ_old, gpu_dQ);
   }

private:
   const int gpu_num_solved;
   CellGPURead<strict_fp_t> gpu_rhs;
   VectorGPURead<strict_fp_t> gpu_A_data;
   VectorGPURead<int> gpu_ia;
   VectorGPURead<int> gpu_ja;
   CellGPURead<strict_fp_t> gpu_dQ_old;
   mutable CellGPU<strict_fp_t> gpu_dQ;
};

void Solver_base_gpu::jacobi_linear_solver(const int num_solved)
{
   CellRead<strict_fp_t> rhs_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("rhs_local");
   VectorRead<int> ia_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ia_local");
   VectorRead<int> ja_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ja_local");
   VectorRead<strict_fp_t> A_data_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("A_data_local");
   Cell<strict_fp_t> dQ_old_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_old_local");
   Cell<strict_fp_t> dQ_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_local");
   
   GDF::transfer_to_gpu_noinit(dQ_old_local);
   GDF::memset_gpu_var(dQ_old_local.gpu_data(), 0, num_solved);

   for (unsigned int iter = 0; iter < num_iter; iter++)
   {
      mpi_nbnb_transfer_gpu(dQ_old_local.gpu_data());
      
      GDF::submit_to_gpu<kg_jacobi_linear_solver>(num_solved,
                                                  rhs_local,
                                                  A_data_local,
                                                  ia_local,
                                                  ja_local,
                                                  dQ_old_local,
                                                  dQ_local);
      
      
      GDF::memcpy_gpu_var(dQ_old_local.gpu_data(), dQ_local.gpu_data(), num_solved);
   }
}

void Solver_base_gpu::setup_m_spmv_system()
{
   Vector<int> ia_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ia_local");
   Vector<int> ja_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ja_local");
   Vector<strict_fp_t> A_data_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("A_data_local");
   Cell<strict_fp_t> rhs_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("rhs_local");
   Cell<strict_fp_t> dQ_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_local");

   assert(m_spmv_sys);
   if(m_spmv_sys->is_setup())
      m_spmv_sys->release_system();

   GDF::transfer_to_gpu_noinit(dQ_local);
   GDF::transfer_to_gpu_move(ia_local, ja_local, A_data_local);
   m_spmv_sys->init_system(nrow_local, ncol_local, nnz_local, 1.0, 0.0, ia_local.gpu_data(),
               ja_local.gpu_data(), A_data_local.gpu_data(), rhs_local.gpu_data(), dQ_local.gpu_data());
}

void Solver_base_gpu::sparse_matvec(strict_fp_t* const vec_in, strict_fp_t* const vec_out)
{
   mpi_nbnb_transfer_gpu(vec_in);
   assert(m_spmv_sys && m_spmv_sys->is_setup());
   strict_fp_t* const x = const_cast<strict_fp_t* const>(vec_in);
   strict_fp_t* const y = const_cast<strict_fp_t* const>(vec_out);
   m_spmv_sys->update_x(ncol_local, x);
   m_spmv_sys->update_y(nrow_local, y);
   m_spmv_sys->compute();
}

void Solver_base_gpu::dot_product(const size_t num_elements, const strict_fp_t* const x, const strict_fp_t* const y, strict_fp_t* const result)
{
   assert(x);
   assert(y);

   oneapi::math::blas::column_major::dot(GDF::get_gpu_queue(), num_elements, x, 1, y, 1, result);
   // GDF::gpu_barrier();
   MPI_Allreduce(result, result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

class kg_bicgstab_axpby
{
public:
   kg_bicgstab_axpby(const int size,
                     const strict_fp_t a,
                     const strict_fp_t* const x,
                     const strict_fp_t b,
                     const strict_fp_t* const y,
                     strict_fp_t* const result):
      gpu_size(size),
      gpu_a(a),
      gpu_x(x),
      gpu_b(b),
      gpu_y(y),
      gpu_result(result)
   {}

   void operator()(sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for(int ii = idx; ii < gpu_size; ii += stride)
      {
         gpu_result[ii] = gpu_a*gpu_x[ii] + gpu_b*gpu_y[ii];
      }
   }

private:
   const int gpu_size;
   const strict_fp_t gpu_a;
   const strict_fp_t* const gpu_x;
   const strict_fp_t gpu_b;
   const strict_fp_t* const gpu_y;
   strict_fp_t* const gpu_result;
};

class kg_bicgstab_xpby
{
public:
   kg_bicgstab_xpby(const int size,
                     const strict_fp_t* const x,
                     const strict_fp_t b,
                     const strict_fp_t* const y,
                     strict_fp_t* const result):
      gpu_size(size),
      gpu_x(x),
      gpu_b(b),
      gpu_y(y),
      gpu_result(result)
   {}

   void operator()(sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for(int ii = idx; ii < gpu_size; ii += stride)
      {
         gpu_result[ii] = gpu_x[ii] + gpu_b*gpu_y[ii];
      }
   }

private:
   const int gpu_size;
   const strict_fp_t* const gpu_x;
   const strict_fp_t gpu_b;
   const strict_fp_t* const gpu_y;
   strict_fp_t* const gpu_result;
};

void Solver_base_gpu::bicgstab_linear_solver()
{
   CellRead<strict_fp_t> rhs_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("rhs_local");
   Cell<strict_fp_t> dQ_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_local");

   GDF::transfer_to_gpu_noinit(dQ_local);
   GDF::memset_gpu_var(dQ_local.gpu_data(), 0, nrow_local);

   //Memory Allocations
   strict_fp_t* const r0 = GDF::malloc_gpu_var<strict_fp_t>(ncol_local);
   strict_fp_t* const r  = GDF::malloc_gpu_var<strict_fp_t>(ncol_local);
   strict_fp_t* const p  = GDF::malloc_gpu_var<strict_fp_t>(ncol_local);
   strict_fp_t* const Ap = GDF::malloc_gpu_var<strict_fp_t>(ncol_local);
   strict_fp_t* const s  = GDF::malloc_gpu_var<strict_fp_t>(ncol_local);
   strict_fp_t* const As = GDF::malloc_gpu_var<strict_fp_t>(ncol_local);

   strict_fp_t* const p1 = GDF::malloc_gpu_var<strict_fp_t>(ncol_local);
   strict_fp_t* const s1 = GDF::malloc_gpu_var<strict_fp_t>(ncol_local);

   //Constants
   strict_fp_t* const alpha1 = GDF::malloc_gpu_var<strict_fp_t, true>(1);
   strict_fp_t* const omega1 = GDF::malloc_gpu_var<strict_fp_t, true>(1);
   
   strict_fp_t* const alpha = GDF::malloc_gpu_var<strict_fp_t, true>(1);
   strict_fp_t* const beta = GDF::malloc_gpu_var<strict_fp_t, true>(1);

   strict_fp_t* const temp = GDF::malloc_gpu_var<strict_fp_t, true>(1);

   // r0 = b-Ax
   strict_fp_t* const Ax = GDF::malloc_gpu_var<strict_fp_t>(nrow_local);
   sparse_matvec(dQ_local.gpu_data(), Ax);
   GDF::submit_to_gpu<kg_bicgstab_xpby>(nrow_local, rhs_local.gpu_data(), -1.0, Ax, r0);
   GDF::free_gpu_var(Ax);

   // r = r0, p = r0
   GDF::memcpy_gpu_var(r, r0, nrow_local);
   GDF::memcpy_gpu_var(p, r0, nrow_local);

   for(int iter = 0; iter < num_iter; iter++)
   {
      // p1 = p
      GDF::memcpy_gpu_var(p1, p, nrow_local);

      // alpha1 = r . r0
      dot_product(nrow_local, r, r0, alpha1);

      // Ap = A * p1
      sparse_matvec(p1, Ap);

      // alpha = Ap . r0
      dot_product(nrow_local, Ap, r0, alpha);

      alpha[0] = alpha1[0] / alpha[0];
      
      // s = r - alpha * Ap
      GDF::submit_to_gpu<kg_bicgstab_xpby>(nrow_local, r, -alpha[0], Ap, s);

      // s1 = s
      GDF::memcpy_gpu_var(s1, s, nrow_local);

      // As = A * s1
      sparse_matvec(s1, As);

      // omega1 = (As . s) / (As . As)
      dot_product(nrow_local, As, s, omega1);
      dot_product(nrow_local, As, As, temp);
      omega1[0] /= temp[0];

      // x = x + alpha * p1 + omega1 * s1
      GDF::submit_to_gpu<kg_bicgstab_xpby>(nrow_local, dQ_local.gpu_data(), alpha[0], p1, dQ_local.gpu_data());
      GDF::submit_to_gpu<kg_bicgstab_xpby>(nrow_local, dQ_local.gpu_data(), omega1[0], s1, dQ_local.gpu_data());
      // r = s - omega1 * As
      GDF::submit_to_gpu<kg_bicgstab_xpby>(nrow_local, s, -omega1[0], As, r);

      // beta = (r . r0) * alpha / alpha1 / omega1
      dot_product(nrow_local, r, r0, beta);
      beta[0] *= alpha[0] / alpha1[0] / omega1[0];

      // p = r + beta * (p - omega1 * Ap)
      GDF::submit_to_gpu<kg_bicgstab_xpby>(nrow_local, r, beta[0], p, p);
      GDF::submit_to_gpu<kg_bicgstab_xpby>(nrow_local, p, -(beta[0]*omega1[0]), Ap, p);
   }

   GDF::free_gpu_var(r);
   GDF::free_gpu_var(r0);
   GDF::free_gpu_var(p);
   GDF::free_gpu_var(Ap);
   GDF::free_gpu_var(s);
   GDF::free_gpu_var(As);
   GDF::free_gpu_var(s1);
   GDF::free_gpu_var(p1);
   GDF::free_gpu_var(alpha1);
   GDF::free_gpu_var(alpha);
   GDF::free_gpu_var(omega1);
   GDF::free_gpu_var(beta);
   GDF::free_gpu_var(temp);
}

void Solver_base_gpu::setup_matrix_struct(const int num_solved, const int num_involved)
{
   if(implicit_solver == true)
   {
      Vector<int> ia_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ia_local");
      Vector<int> ja_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ja_local");
      Vector<strict_fp_t> A_data_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("A_data_local");
      Cell<int> csr_diag_idx_local = m_silo.retrieve_entry<int, CDF::StorageType::CELL>("csr_diag_idx_local");
      Face<int> csr_idx_local = m_silo.retrieve_entry<int, CDF::StorageType::FACE>("csr_idx_local");
      VectorRead<int> number_of_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors_local");
      FaceRead<int> cell_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::FACE>("cell_neighbors_local");

      nrow_local = num_solved;
      ncol_local = num_involved;

      ia_local.resize(num_solved + 1);
      for (unsigned int i = 0; i < num_solved; i++)
      {
         ia_local[i + 1] = number_of_neighbors_local[i + 1] - number_of_neighbors_local[i] + 1;
      }
      for (unsigned int i = 1; i < num_solved + 1; i++)
      {
         ia_local[i] = ia_local[i] + ia_local[i - 1];
      }

      this->nnz_local = ia_local[num_solved];

      ja_local.resize(this->nnz_local);
      A_data_local.resize(this->nnz_local);

      std::vector<int> temp(num_solved, 0);

      // Fill the ja array. First is the diagonal entry
      for (unsigned int i = 0; i < num_solved; i++)
      {
         const int left_index = ia_local[i] + temp[i];
         temp[i]++;
         ja_local[left_index] = i;
      }

      for (unsigned int i = 0; i < num_solved; i++)
      {
         for (int j = number_of_neighbors_local[i]; j < number_of_neighbors_local[i + 1]; j++)
         {
            const int left = i;
            const int right = cell_neighbors_local[j];

            const int left_index = ia_local[left] + temp[left];
            temp[left]++;
            ja_local[left_index] = right;

            if (left_index >= ia_local[left + 1])
            {
               printf("Problem in filling the ja array for the matrix.\n");
               printf("Cell %d: left_index %d ia %d ia %d\n", left, left_index, ia_local[left], ia_local[left + 1]);
            }
         }
      }

      for(unsigned int i = 0; i < num_solved; i++)
      {
         csr_diag_idx_local[i] = ia_local[i];
         for (int j = number_of_neighbors_local[i]; j < number_of_neighbors_local[i + 1]; j++)
         {
            const int nbr_local = cell_neighbors_local[j];
            for(unsigned int nn = ia_local[i]+1; nn < ia_local[i + 1]; nn++)
            {
               if(ja_local[nn] == nbr_local)
               {
                  csr_idx_local[j] = nn;
               }
            }
         }
      }
   }
}

class kg_copy_from_vec_to_buf
{
public:
      kg_copy_from_vec_to_buf(const int num_elements,
                              const int* const list,
                              const strict_fp_t* const vec,
                              strict_fp_t* const buf):
      gpu_num_elements(num_elements),
      gpu_list(list),
      gpu_vec(vec),
      gpu_buf(buf)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for(int ii = idx; ii < gpu_num_elements; ii += stride)
      {
         gpu_buf[ii] = gpu_vec[gpu_list[ii]];
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_elements, gpu_list, gpu_vec, gpu_buf);
   }

private:
   const int gpu_num_elements;
   const int* const gpu_list;
   const strict_fp_t* const gpu_vec;
   strict_fp_t* const gpu_buf;
};

class kg_copy_from_buf_to_vec
{
public:
      kg_copy_from_buf_to_vec(const int num_elements,
                              const int* const list,
                              const strict_fp_t* const buf,
                              strict_fp_t* const vec):
      gpu_num_elements(num_elements),
      gpu_list(list),
      gpu_buf(buf),
      gpu_vec(vec)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for(int ii = idx; ii < gpu_num_elements; ii += stride)
      {
         gpu_vec[gpu_list[ii]] = gpu_buf[ii];
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_elements, gpu_list, gpu_buf, gpu_vec);
   }

private:
   const int gpu_num_elements;
   const int* const gpu_list;
   const strict_fp_t* const gpu_buf;
   strict_fp_t* const gpu_vec;
};

void mpi_nbnb_transfer_gpu(strict_fp_t *vec)
{
   GDF::submit_to_gpu<kg_copy_from_vec_to_buf>(mpiparams->ssize,
                                               mpiparams->gpu_slist,
                                               vec,
                                               mpiparams->gpu_sbuf);
   
   MPI_Alltoallv(mpiparams->gpu_sbuf, mpiparams->scounts, mpiparams->sdisp, MPI_DOUBLE, 
   mpiparams->gpu_rbuf, mpiparams->rcounts, mpiparams->rdisp, MPI_DOUBLE, MPI_COMM_WORLD);

   GDF::submit_to_gpu<kg_copy_from_buf_to_vec>(mpiparams->rsize,
                                               mpiparams->gpu_rlist,
                                               mpiparams->gpu_rbuf,
                                               vec);
}

class kg_sum_interior_face_areas
{
public:
   kg_sum_interior_face_areas(const int num_solved, VectorGPURead<int>& number_of_neighbors, FaceGPURead<strict_fp_t>& area, strict_fp_t* const data):
      gpu_num_solved(num_solved),
      gpu_number_of_neighbors(number_of_neighbors),
      gpu_area(area),
      gpu_data(data)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (int il = idx; il < gpu_num_solved; il += stride)
      {
         for (unsigned int jl = gpu_number_of_neighbors[il]; jl < gpu_number_of_neighbors[il + 1]; jl++)
         {
            const int left = il;
            gpu_data[left] += gpu_area[jl];
         }
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_solved, gpu_number_of_neighbors, gpu_area, gpu_data);
   }

private:
   const int gpu_num_solved;
   VectorGPURead<int> gpu_number_of_neighbors;
   FaceGPURead<strict_fp_t> gpu_area;
   strict_fp_t* const gpu_data;
};

class kg_sum_boundary_face_areas
{
public:
   kg_sum_boundary_face_areas(const int num_boundary_faces,
                              BoundaryGPURead<int>& boundary_face_to_cell,
                              BoundaryGPURead<strict_fp_t>& boundary_area,
                              strict_fp_t* const data):
      gpu_num_boundary_faces(num_boundary_faces),
      gpu_boundary_face_to_cell(boundary_face_to_cell),
      gpu_boundary_area(boundary_area),
      gpu_data(data)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for(unsigned int i = idx; i < gpu_num_boundary_faces; i += stride)
      {
         const int cell = gpu_boundary_face_to_cell[i];
         GDF::atomic_add(gpu_data[cell], gpu_boundary_area[i]);
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_boundary_faces,
      gpu_boundary_face_to_cell, gpu_boundary_area, gpu_data);
   }

private:
   const int gpu_num_boundary_faces;
   BoundaryGPURead<int> gpu_boundary_face_to_cell;
   BoundaryGPURead<strict_fp_t> gpu_boundary_area;
   strict_fp_t* const gpu_data;
};

class kg_find_min_inverse_length_scale
{
public:
   kg_find_min_inverse_length_scale(const int num_solved,
                                    CellGPURead<strict_fp_t>& volume,
                                    strict_fp_t* const data,
                                    strict_fp_t* const data_min):
      gpu_num_solved(num_solved),
      gpu_volume(volume),
      gpu_data(data),
      gpu_data_min(data_min)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for(unsigned int il = idx; il < gpu_num_solved; il += stride)
      {
         gpu_data[il] = gpu_data[il] / (gpu_volume[il] * 2);
         GDF::atomic_min(*gpu_data_min, gpu_data[il]);
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_solved, gpu_volume, gpu_data, gpu_data_min);
   }

private:
   const int gpu_num_solved;
   CellGPURead<strict_fp_t> gpu_volume;
   strict_fp_t* const gpu_data;
   strict_fp_t* const gpu_data_min;
};

void Solver_base_gpu::compute_time_step(const int num_solved, const int num_attached)
{
   strict_fp_t* data = GDF::malloc_gpu_var<strict_fp_t>(num_solved);
   GDF::memset_gpu_var(data, 0, num_solved);

   VectorRead<int> number_of_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors_local");
   FaceRead<strict_fp_t> area_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::FACE>("area_local");
   
   GDF::submit_to_gpu<kg_sum_interior_face_areas>(num_solved,
                                                  number_of_neighbors_local,
                                                  area_local,
                                                  data);
   
   BoundaryRead<int> boundary_face_to_cell_local = m_silo.retrieve_entry<int, CDF::StorageType::BOUNDARY>("boundary_face_to_cell_local");
   BoundaryRead<strict_fp_t> boundary_area_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_area_local");
   
   GDF::submit_to_gpu<kg_sum_boundary_face_areas>(num_attached,
                                                  boundary_face_to_cell_local,
                                                  boundary_area_local,
                                                  data);
   
   strict_fp_t* min_value = GDF::malloc_gpu_var<strict_fp_t, true>(1);
   min_value[0] = 1.e30;

   CellRead<strict_fp_t> volume_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("volume_local");
   
   GDF::submit_to_gpu<kg_find_min_inverse_length_scale>(num_solved,
                                                        volume_local,
                                                        data,
                                                        min_value);

   MPI_Allreduce(min_value, min_value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

   // This is the limit. (dt / dx2) + (dt / dy2) < 0.5
   delta_t = 1.0 / (min_value[0] * min_value[0] * 2.0);     // 2.0 added to reduce from limit.

   if(implicit_solver == true)
      delta_t = delta_t * 10.0;

   if(rank == 0)
      printf("GPU: Value of dt %e\n", delta_t);

   GDF::free_gpu_var(data);
   GDF::free_gpu_var(min_value);
}

class kg_compute_interior_faces_rdist
{
public:
   kg_compute_interior_faces_rdist(const int num_solved,
                                   VectorGPURead<int>& number_of_neighbors,
                                   FaceGPURead<int>& cell_neighbors,
                                   VectorGPURead<strict_fp_t>& normal,
                                   VectorGPURead<strict_fp_t>& xcen,
                                   FaceGPURead<strict_fp_t>& area,
                                   FaceGPU<strict_fp_t>& rdista):
      gpu_num_solved(num_solved),
      gpu_number_of_neighbors(number_of_neighbors),
      gpu_cell_neighbors(cell_neighbors),
      gpu_normal(normal),
      gpu_xcen(xcen),
      gpu_area(area),
      gpu_rdista(rdista)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for(unsigned int il = idx; il < gpu_num_solved; il += stride)
      {
         for (int jl = gpu_number_of_neighbors[il]; jl < gpu_number_of_neighbors[il + 1]; jl++)
         {
            const int left = il;
            const int right = gpu_cell_neighbors[jl];
      
            strict_fp_t distance;
            if(std::abs(gpu_normal[2 * jl]) > 0.1)      // x face
            {
               distance = std::abs(gpu_xcen[2 * right] - gpu_xcen[2 * left]);
            }
            else
            {
               distance = std::abs(gpu_xcen[(2 * right) + 1] - gpu_xcen[(2 * left) + 1]);
            }
            
            gpu_rdista[jl] = 1.0 / distance * gpu_area[jl];
         }
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_solved, gpu_number_of_neighbors,
      gpu_cell_neighbors, gpu_normal, gpu_xcen, gpu_area, gpu_rdista);
   }

private:
   const int gpu_num_solved;
   VectorGPURead<int> gpu_number_of_neighbors;
   FaceGPURead<int> gpu_cell_neighbors;
   VectorGPURead<strict_fp_t> gpu_normal;
   VectorGPURead<strict_fp_t> gpu_xcen;
   FaceGPURead<strict_fp_t> gpu_area;
   mutable FaceGPU<strict_fp_t> gpu_rdista;
};

class kg_compute_boundary_faces_rdist
{
public:
   kg_compute_boundary_faces_rdist(const int boundary_faces,
                                   BoundaryGPURead<int>& boundary_face_to_cell,
                                   BoundaryGPURead<strict_fp_t>& boundary_area,
                                   VectorGPURead<strict_fp_t>& boundary_normal,
                                   VectorGPURead<strict_fp_t>& boundary_xcen,
                                   VectorGPURead<strict_fp_t>& xcen,
                                   BoundaryGPU<strict_fp_t>& boundary_rdista):
      gpu_boundary_faces(boundary_faces),
      gpu_boundary_face_to_cell(boundary_face_to_cell),
      gpu_boundary_area(boundary_area),
      gpu_boundary_normal(boundary_normal),
      gpu_boundary_xcen(boundary_xcen),
      gpu_xcen(xcen),
      gpu_boundary_rdista(boundary_rdista)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for(unsigned int i = idx; i < gpu_boundary_faces; i += stride)
      {
         const int cell = gpu_boundary_face_to_cell[i];

         strict_fp_t distance;
         if (std::abs(gpu_boundary_normal[2 * i]) > 0.1)      // x face
         {
            distance = std::abs(gpu_boundary_xcen[2 * i] - gpu_xcen[2 * cell]);
         }
         else
         {
            distance = std::abs(gpu_boundary_xcen[(2 * i) + 1] - gpu_xcen[(2 * cell) + 1]);
         }
         gpu_boundary_rdista[i] = 1.0 / distance * gpu_boundary_area[i];
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_boundary_faces, gpu_boundary_face_to_cell,
      gpu_boundary_area, gpu_boundary_normal, gpu_boundary_xcen, gpu_xcen, gpu_boundary_rdista);
   }

private:
   const int gpu_boundary_faces;
   BoundaryGPURead<int> gpu_boundary_face_to_cell;
   BoundaryGPURead<strict_fp_t> gpu_boundary_area;
   VectorGPURead<strict_fp_t> gpu_boundary_normal;
   VectorGPURead<strict_fp_t> gpu_boundary_xcen;
   VectorGPURead<strict_fp_t> gpu_xcen;
   mutable BoundaryGPU<strict_fp_t> gpu_boundary_rdista;
};

void Solver_base_gpu::compute_rdist(const int num_solved, const int num_attached)
{
   // Diffusion term for interior cells
   VectorRead<int> number_of_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors_local");
   FaceRead<int> cell_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::FACE>("cell_neighbors_local");
   FaceRead<strict_fp_t> area_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::FACE>("area_local");
   VectorRead<strict_fp_t> xcen_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("xcen_local");
   VectorRead<strict_fp_t> normal_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("normal_local");
   Face<strict_fp_t> rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::FACE>("rdista_local");
   
   GDF::submit_to_gpu<kg_compute_interior_faces_rdist>(num_solved,
                                                       number_of_neighbors_local,
                                                       cell_neighbors_local,
                                                       normal_local,
                                                       xcen_local,
                                                       area_local,
                                                       rdista_local);
   
   // Diffusion term for boundary cells
   VectorRead<int> ranks_list = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ranks_list");
   VectorRead<int> global_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("global_local");
   BoundaryRead<int> boundary_face_to_cell_local = m_silo.retrieve_entry<int, CDF::StorageType::BOUNDARY>("boundary_face_to_cell_local");
   BoundaryRead<strict_fp_t> boundary_area_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_area_local");
   VectorRead<strict_fp_t> boundary_xcen_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_xcen_local");
   VectorRead<strict_fp_t> boundary_normal_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_normal_local");
   Boundary<strict_fp_t> boundary_rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_rdista_local");
   
   GDF::submit_to_gpu<kg_compute_boundary_faces_rdist>(num_attached,
                                                       boundary_face_to_cell_local,
                                                       boundary_area_local,
                                                       boundary_normal_local,
                                                       boundary_xcen_local,
                                                       xcen_local,
                                                       boundary_rdista_local);
}

class kg_residual_interior_diffusion
{
public:
   kg_residual_interior_diffusion(const int num_solved,
                                  VectorGPURead<int>& number_of_neighbors,
                                  FaceGPURead<int>& cell_neighbors,
                                  FaceGPURead<strict_fp_t>& rdista,
                                  CellGPURead<strict_fp_t>& Q_cell,
                                  CellGPU<strict_fp_t>& residual):
      gpu_num_solved(num_solved),
      gpu_number_of_neighbors(number_of_neighbors),
      gpu_cell_neighbors(cell_neighbors),
      gpu_rdista(rdista),
      gpu_Q_cell(Q_cell),
      gpu_residual(residual)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (unsigned int il = idx; il < gpu_num_solved; il += stride)
      {
         gpu_residual[il] = 0.0;
         for (int jl = gpu_number_of_neighbors[il]; jl < gpu_number_of_neighbors[il + 1]; jl++)
         {
            const int left = il;
            const int right = gpu_cell_neighbors[jl];
            strict_fp_t face_value = (gpu_Q_cell[right] - gpu_Q_cell[left]) * gpu_rdista[jl];
            gpu_residual[left] -= face_value;
         }
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_solved, gpu_number_of_neighbors,
      gpu_cell_neighbors, gpu_rdista, gpu_Q_cell, gpu_residual);
   }

private:
   const int gpu_num_solved;
   VectorGPURead<int> gpu_number_of_neighbors;
   FaceGPURead<int> gpu_cell_neighbors;
   FaceGPURead<strict_fp_t> gpu_rdista;
   CellGPURead<strict_fp_t> gpu_Q_cell;
   mutable CellGPU<strict_fp_t> gpu_residual;
};

class kg_residual_boundary_diffusion
{
public:
   kg_residual_boundary_diffusion(const int num_boundary_faces,
                                  BoundaryGPURead<int>& boundary_face_to_cell,
                                  BoundaryGPURead<strict_fp_t>& Q_boundary,
                                  CellGPURead<strict_fp_t>& Q_cell,
                                  BoundaryGPURead<strict_fp_t>& boundary_rdista,
                                  CellGPU<strict_fp_t>& residual):
      gpu_num_boundary_faces(num_boundary_faces),
      gpu_boundary_face_to_cell(boundary_face_to_cell),
      gpu_Q_boundary(Q_boundary),
      gpu_Q_cell(Q_cell),
      gpu_boundary_rdista(boundary_rdista),
      gpu_residual(residual)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (unsigned int i = idx; i < gpu_num_boundary_faces; i += stride)
      {
         const int cell = gpu_boundary_face_to_cell[i];

         const strict_fp_t face_value = (gpu_Q_boundary[i] - gpu_Q_cell[cell]) * gpu_boundary_rdista[i];

         GDF::atomic_sub(gpu_residual[cell], face_value);
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_boundary_faces, gpu_boundary_face_to_cell,
      gpu_Q_boundary, gpu_Q_cell, gpu_boundary_rdista, gpu_residual);
   }

private:
   const int gpu_num_boundary_faces;
   BoundaryGPURead<int> gpu_boundary_face_to_cell;
   BoundaryGPURead<strict_fp_t> gpu_Q_boundary;
   CellGPURead<strict_fp_t> gpu_Q_cell;
   BoundaryGPURead<strict_fp_t> gpu_boundary_rdista;
   mutable CellGPU<strict_fp_t> gpu_residual;
};

class kg_compute_system_compute_residual_norm
{
public:
   kg_compute_system_compute_residual_norm(const int num_solved,
                                           CellGPURead<strict_fp_t>& residual,
                                           strict_fp_t* const residual_norm):
      gpu_num_solved(num_solved),
      gpu_residual(residual),
      gpu_residual_norm(residual_norm)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (unsigned int i = idx; i < gpu_num_solved; i += stride)
      {
         GDF::atomic_add(gpu_residual_norm[0], sycl::fabs(gpu_residual[i]));
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_solved, gpu_residual, gpu_residual_norm);
   }

private:
   const int gpu_num_solved;
   CellGPURead<strict_fp_t> gpu_residual;
   strict_fp_t* const gpu_residual_norm;
};

class kg_compute_system_add_residual_to_rhs
{
public:
   kg_compute_system_add_residual_to_rhs(const int num_solved,
                                         CellGPURead<strict_fp_t>& residual,
                                         CellGPU<strict_fp_t>& rhs):
      gpu_num_solved(num_solved),
      gpu_residual(residual),
      gpu_rhs(rhs)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (unsigned int i = idx; i < gpu_num_solved; i += stride)
      {
         gpu_rhs[i] = -gpu_residual[i];
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_solved, gpu_residual, gpu_rhs);
   }

private:
   const int gpu_num_solved;
   CellGPURead<strict_fp_t> gpu_residual;
   mutable CellGPU<strict_fp_t> gpu_rhs;
};

void Solver_base_gpu::compute_residual(const int num_solved, const int num_attached)
{
   // Diffusion term for internal cells
   VectorRead<int> number_of_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors_local");
   FaceRead<int> cell_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::FACE>("cell_neighbors_local");
   FaceRead<strict_fp_t> rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::FACE>("rdista_local");
   CellRead<strict_fp_t> Q_cell_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Q_cell_local");
   Cell<strict_fp_t> residual_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("residual_local");
   
   GDF::submit_to_gpu<kg_residual_interior_diffusion>(num_solved,
                                                      number_of_neighbors_local,
                                                      cell_neighbors_local,
                                                      rdista_local,
                                                      Q_cell_local,
                                                      residual_local);
   
   //Diffusion term for boundary cells
   BoundaryRead<int> boundary_face_to_cell_local = m_silo.retrieve_entry<int, CDF::StorageType::BOUNDARY>("boundary_face_to_cell_local");
   BoundaryRead<strict_fp_t> Q_boundary_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("Q_boundary_local");
   BoundaryRead<strict_fp_t> boundary_rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_rdista_local");
   
   GDF::submit_to_gpu<kg_residual_boundary_diffusion>(num_attached,
                                                      boundary_face_to_cell_local,
                                                      Q_boundary_local,
                                                      Q_cell_local,
                                                      boundary_rdista_local,
                                                      residual_local);
   
   oneapi::math::blas::column_major::asum(GDF::get_gpu_queue(), residual_local.size(), residual_local.gpu_data(), 1, &this->residual_norm);

   if(implicit_solver == true)
   {
      Cell<strict_fp_t> rhs_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("rhs_local");

      GDF::submit_to_gpu<kg_compute_system_add_residual_to_rhs>(num_solved,
                                                                residual_local,
                                                                rhs_local);
      
   }
}

class kg_compute_system_time_term
{
public:
   kg_compute_system_time_term(const int num_solved,
                               const strict_fp_t delta_t,
                               CellGPURead<int>& csr_diag_idx,
                               CellGPURead<strict_fp_t>& volume,
                               VectorGPU<strict_fp_t>& A_data):
      gpu_num_solved(num_solved),
      gpu_delta_t(delta_t),
      gpu_csr_diag_idx(csr_diag_idx),
      gpu_volume(volume),
      gpu_A_data(A_data)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (unsigned int il = idx; il < gpu_num_solved; il += stride)
      {
         const int crs_index = gpu_csr_diag_idx[il];
         gpu_A_data[crs_index] += gpu_volume[il] / gpu_delta_t;
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_solved, gpu_delta_t, gpu_csr_diag_idx, gpu_volume, gpu_A_data);
   }

private:
   const int gpu_num_solved;
   const strict_fp_t gpu_delta_t;
   CellGPURead<int> gpu_csr_diag_idx;
   CellGPURead<strict_fp_t> gpu_volume;
   mutable VectorGPU<strict_fp_t> gpu_A_data;
};

class kg_compute_system_interior_diffusion
{
public:
   kg_compute_system_interior_diffusion(const int num_solved,
                                        VectorGPURead<int>& number_of_neighbors,
                                        FaceGPURead<strict_fp_t>& rdista,
                                        CellGPURead<int>& csr_diag_idx,
                                        FaceGPURead<int>& csr_idx,
                                        VectorGPU<strict_fp_t>& A_data):
      gpu_num_solved(num_solved),
      gpu_number_of_neighbors(number_of_neighbors),
      gpu_rdista(rdista),
      gpu_csr_diag_idx(csr_diag_idx),
      gpu_csr_idx(csr_idx),
      gpu_A_data(A_data)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (unsigned int il = idx; il < gpu_num_solved; il += stride)
      {
         for (int jl = gpu_number_of_neighbors[il]; jl < gpu_number_of_neighbors[il + 1]; jl++)
         {
            const int left = il;

            const strict_fp_t temp = gpu_rdista[jl];

            // Left cell
            int crs_index = gpu_csr_diag_idx[left];
            GDF::atomic_add(gpu_A_data[crs_index], temp);

            crs_index = gpu_csr_idx[jl];
            GDF::atomic_sub(gpu_A_data[crs_index], temp);
         }
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_solved, gpu_number_of_neighbors,
      gpu_rdista, gpu_csr_diag_idx, gpu_csr_idx, gpu_A_data);
   }

private:
   const int gpu_num_solved;
   VectorGPURead<int> gpu_number_of_neighbors;
   FaceGPURead<strict_fp_t> gpu_rdista;
   CellGPURead<int> gpu_csr_diag_idx;
   FaceGPURead<int> gpu_csr_idx;
   mutable VectorGPU<strict_fp_t> gpu_A_data;
};

class kg_compute_system_boundary_diffusion
{
public:
   kg_compute_system_boundary_diffusion(const int num_boundary_faces,
                                        BoundaryGPURead<int>& boundary_face_to_cell,
                                        BoundaryGPURead<strict_fp_t>& boundary_rdista,
                                        CellGPURead<int>& csr_diag_idx,
                                        VectorGPU<strict_fp_t>& A_data):
      gpu_num_boundary_faces(num_boundary_faces),
      gpu_boundary_face_to_cell(boundary_face_to_cell),
      gpu_boundary_rdista(boundary_rdista),
      gpu_csr_diag_idx(csr_diag_idx),
      gpu_A_data(A_data)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (unsigned int i = idx; i < gpu_num_boundary_faces; i += stride)
      {
         const int cell = gpu_boundary_face_to_cell[i];

         const strict_fp_t temp = gpu_boundary_rdista[i];

         // Left cell
         int crs_index = gpu_csr_diag_idx[cell];
         GDF::atomic_add(gpu_A_data[crs_index], temp);
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_boundary_faces, gpu_boundary_face_to_cell,
      gpu_boundary_rdista, gpu_csr_diag_idx, gpu_A_data);
   }

private:
   const int gpu_num_boundary_faces;
   BoundaryGPURead<int> gpu_boundary_face_to_cell;
   BoundaryGPURead<strict_fp_t> gpu_boundary_rdista;
   CellGPURead<int> gpu_csr_diag_idx;
   mutable VectorGPU<strict_fp_t> gpu_A_data;
};

void Solver_base_gpu::compute_system(const int num_solved, const int num_attached)
{
   if(implicit_solver == true)
   {
      Vector<strict_fp_t> A_data_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("A_data_local");

      GDF::transfer_to_gpu_noinit(A_data_local);
      GDF::memset_gpu_var(A_data_local.gpu_data(), 0, this->nnz_local);

      // Time term
      CellRead<strict_fp_t> volume_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("volume_local");
      CellRead<int> csr_diag_idx_local = m_silo.retrieve_entry<int, CDF::StorageType::CELL>("csr_diag_idx_local");
      
      GDF::submit_to_gpu<kg_compute_system_time_term>(num_solved,
                                                      delta_t,
                                                      csr_diag_idx_local,
                                                      volume_local,
                                                      A_data_local);
      
      // Diffusion term for internal cells
      VectorRead<int> number_of_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors_local");
      FaceRead<int> csr_idx_local = m_silo.retrieve_entry<int, CDF::StorageType::FACE>("csr_idx_local");
      FaceRead<strict_fp_t> rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::FACE>("rdista_local");
      
      GDF::submit_to_gpu<kg_compute_system_interior_diffusion>(num_solved,
                                                               number_of_neighbors_local,
                                                               rdista_local,
                                                               csr_diag_idx_local,
                                                               csr_idx_local,
                                                               A_data_local);
      
      // Diffusion term for boundary cells
      VectorRead<int> ranks_list = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ranks_list");
      VectorRead<int> global_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("global_local");
      BoundaryRead<int> boundary_face_to_cell_local = m_silo.retrieve_entry<int, CDF::StorageType::BOUNDARY>("boundary_face_to_cell_local");
      BoundaryRead<strict_fp_t> boundary_rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_rdista_local");
      
      GDF::submit_to_gpu<kg_compute_system_boundary_diffusion>(num_attached,
                                                               boundary_face_to_cell_local,
                                                               boundary_rdista_local,
                                                               csr_diag_idx_local,
                                                               A_data_local);
   }
}

strict_fp_t Solver_base_gpu::print_residual_norm(const int time_iter)
{
   if(rank == 0) printf("Time iter: %d, Residual %0.16e\n", time_iter, residual_norm);

   return residual_norm;
}

strict_fp_t Solver_base_gpu::get_residual_norm()
{
   return residual_norm;
}
