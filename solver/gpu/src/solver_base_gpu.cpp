 #include <vector>
#include <cmath>
#include <cstdio>

#include "solver_gpu.h"
#include "silo.h"
#include "silo_fwd.h"
#include "fp_data_types.h"

#include "gpu_api_functions.h"
#include "oneMathSPMV.h"
#include "oneapi/math/blas.hpp"

Solver_base_gpu::Solver_base_gpu():
   ia(m_silo.register_entry<int, CDF::StorageType::VECTOR>("ia")),
   ja(m_silo.register_entry<int, CDF::StorageType::VECTOR>("ja")),
   csr_idx(m_silo.register_entry<int, CDF::StorageType::FACE>("csr_idx")),
   csr_diag_idx(m_silo.register_entry<int, CDF::StorageType::CELL>("csr_diag_idx")),
   A_data(m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("A_data")),
   area(m_silo.register_entry<strict_fp_t, CDF::StorageType::FACE>("area")),
   normal(m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("normal")),
   xcen(m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("xcen")),
   boundary_normal(m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_normal")),
   boundary_xcen(m_silo.register_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_xcen")),
   number_of_neighbors(m_silo.register_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors")),
   cell_neighbors(m_silo.register_entry<int, CDF::StorageType::FACE>("cell_neighbors")),
   boundary_type_start_and_end_index(m_silo.register_entry<int, CDF::StorageType::VECTOR>("boundary_type_start_and_end_index")),
   volume(m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("volume")),
   boundary_face_to_cell(m_silo.register_entry<int, CDF::StorageType::BOUNDARY>("boundary_face_to_cell")),
   boundary_area(m_silo.register_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_area")),
   rhs(m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("rhs")),
   dQ(m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("dQ")),
   dQ_old(m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_old")),
   Q_cell(m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("Q_cell")),
   Q_boundary(m_silo.register_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("Q_boundary")),
   residual(m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("residual"))
{
   residual_norm = 1e30;

   m_spmv_sys = new GDF::oneMathSPMV();
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
   GDF::submit_to_gpu<kg_set_boundary_conditions>(boundary_type_start_and_end_index[2],
                                                  boundary_type_start_and_end_index[3],
                                                  QL,
                                                  Q_boundary);
   
   GDF::submit_to_gpu<kg_set_boundary_conditions>(boundary_type_start_and_end_index[3],
                                                  boundary_type_start_and_end_index[4],
                                                  QR,
                                                  Q_boundary);
   
   GDF::submit_to_gpu<kg_set_boundary_conditions>(boundary_type_start_and_end_index[0],
                                                  boundary_type_start_and_end_index[1],
                                                  QB,
                                                  Q_boundary);
   
   GDF::submit_to_gpu<kg_set_boundary_conditions>(boundary_type_start_and_end_index[1],
                                                  boundary_type_start_and_end_index[2],
                                                  QT,
                                                  Q_boundary);
}

class kg_initialize_solution
{
public:
   kg_initialize_solution(const int num_cells, const strict_fp_t Q_initial, CellGPU<strict_fp_t>& Q_cell):
      gpu_num_cells(num_cells),
      gpu_Q_initial(Q_initial),
      gpu_Q_cell(Q_cell)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for(int i = idx; i < gpu_num_cells; i += stride)
      {
         gpu_Q_cell[i] = gpu_Q_initial;
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_cells, gpu_Q_initial, gpu_Q_cell);
   }

private:
   const int gpu_num_cells;
   const strict_fp_t gpu_Q_initial;
   mutable CellGPU<strict_fp_t> gpu_Q_cell;
};

void Solver_base_gpu::initialize_solution (const strict_fp_t Q_initial)
{
   GDF::submit_to_gpu<kg_initialize_solution>(this->num_cells, Q_initial, Q_cell);
}

void Solver_base_gpu::print_residual ()
{
   for (unsigned int i = 0; i < this->num_cells; i++)
   {
      printf("Cell %d, residual %e\n", i, residual[i]);
   }
}

void Solver_base_gpu::print_solution ()
{
   for (unsigned int i = 0; i < this->num_cells; i++)
   {
      printf("Cell %d, solution %e\n", i, Q_cell[i]);
   }
}

class kg_update_solution_add_dQ_to_Q_cell
{
public:
   kg_update_solution_add_dQ_to_Q_cell(const int num_cells, CellGPURead<strict_fp_t>& dQ, CellGPU<strict_fp_t>& Q_cell):
      gpu_num_cells(num_cells),
      gpu_dQ(dQ),
      gpu_Q_cell(Q_cell)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (unsigned int i = idx; i < gpu_num_cells; i += stride)
      {
         gpu_Q_cell[i] += gpu_dQ[i];
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_cells, gpu_dQ, gpu_Q_cell);
   }

private:
   const int gpu_num_cells;
   CellGPURead<strict_fp_t> gpu_dQ;
   mutable CellGPU<strict_fp_t> gpu_Q_cell;
};

void Solver_base_gpu::update_solution(const int solver_type, const int num_iter)
{
   if (m_implicit == false)
   {
      for (unsigned int i = 0; i < this->num_cells; i++)
      {
         Q_cell[i] -= residual[i] * delta_t / volume[i];
      }
   }
   else
   {
      if(solver_type == 0)
      {
         jacobi_linear_solver(num_iter);
      }
      else
      {
         setup_m_spmv_system();

         bicgstab_linear_solver(num_iter);
      }

      GDF::submit_to_gpu<kg_update_solution_add_dQ_to_Q_cell>(this->num_cells, dQ, Q_cell);
   }
}

void Solver_base_gpu::write_solution(std::string file_name)
{
   FILE* file = fopen(file_name.c_str(), "w");

   for(int i = 0; i < this->num_cells; i++)
   {
      fprintf(file, "%0.16f, %.16f, %.16f\n", this->xcen[2*i], this->xcen[2*i+1], this->Q_cell[i]);
   }

   fclose(file);
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
   kg_jacobi_linear_solver(const int num_cells,
                           CellGPURead<strict_fp_t>& rhs,
                           VectorGPURead<strict_fp_t>& A_data,
                           VectorGPURead<int>& ia,
                           VectorGPURead<int>& ja,
                           CellGPURead<strict_fp_t>& dQ_old,
                           CellGPU<strict_fp_t>& dQ):
      gpu_num_cells(num_cells),
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
      for(unsigned int i = idx; i < gpu_num_cells; i += stride)
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
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_cells, gpu_rhs, gpu_A_data, gpu_ia, gpu_ja, gpu_dQ_old, gpu_dQ);
   }

private:
   const int gpu_num_cells;
   CellGPURead<strict_fp_t> gpu_rhs;
   VectorGPURead<strict_fp_t> gpu_A_data;
   VectorGPURead<int> gpu_ia;
   VectorGPURead<int> gpu_ja;
   CellGPURead<strict_fp_t> gpu_dQ_old;
   mutable CellGPU<strict_fp_t> gpu_dQ;
};

void Solver_base_gpu::jacobi_linear_solver(const int num_iter)
{
   GDF::transfer_to_gpu_noinit(dQ_old);
   GDF::memset_gpu_var(dQ_old.gpu_data(), 0, this->num_cells);

   for (unsigned int iter = 0; iter < num_iter; iter++)
   {
      GDF::submit_to_gpu<kg_jacobi_linear_solver>(this->num_cells, rhs, A_data, ia, ja, dQ_old, dQ);
      GDF::memcpy_gpu_var(dQ_old.gpu_data(), dQ.gpu_data(), this->num_cells);
   }
}

void Solver_base_gpu::setup_m_spmv_system()
{
   assert(m_spmv_sys);
   if(m_spmv_sys->is_setup())
      m_spmv_sys->release_system();
   
   GDF::transfer_to_gpu_noinit(dQ);
   GDF::transfer_to_gpu_move(ia, ja);
   m_spmv_sys->init_system(num_cells, num_cells, nnz, 1.0, 0.0, ia.gpu_data(),
               ja.gpu_data(), A_data.gpu_data(), rhs.gpu_data(), dQ.gpu_data());
}

void Solver_base_gpu::sparse_matvec(const strict_fp_t* const vec_in, strict_fp_t* const vec_out)
{
   assert(m_spmv_sys && m_spmv_sys->is_setup());
   double* const x = const_cast<double* const>(vec_in);
   double* const y = const_cast<double* const>(vec_out);
   m_spmv_sys->update_x(num_cells, x);
   m_spmv_sys->update_y(num_cells, y);
   m_spmv_sys->compute();
}

void Solver_base_gpu::dot_product(const strict_fp_t* const x, const strict_fp_t* const y, strict_fp_t* const result)
{
   assert(x);
   assert(y);

   oneapi::math::blas::column_major::dot(GDF::get_gpu_queue(), this->num_cells, x, 1, y, 1, result);
   GDF::gpu_barrier();
}

class kg_bicgstab_axpby
{
public:
   kg_bicgstab_axpby(const int num_cells,
                     const strict_fp_t a,
                     const strict_fp_t* const x,
                     const strict_fp_t b,
                     const strict_fp_t* const y,
                     strict_fp_t* const result):
      gpu_num_cells(num_cells),
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
      for(int ii = idx; ii < gpu_num_cells; ii += stride)
      {
         gpu_result[ii] = gpu_a*gpu_x[ii] + gpu_b*gpu_y[ii];
      }
   }

private:
   const int gpu_num_cells;
   const strict_fp_t gpu_a;
   const strict_fp_t* const gpu_x;
   const strict_fp_t gpu_b;
   const strict_fp_t* const gpu_y;
   strict_fp_t* const gpu_result;
};

void Solver_base_gpu::bicgstab_linear_solver(const int num_iter)
{
   GDF::memset_gpu_var(dQ.gpu_data(), 0, this->num_cells);

   //Memory Allocations
   strict_fp_t* const r0 = GDF::malloc_gpu_var<strict_fp_t>(this->num_cells);
   strict_fp_t* const r  = GDF::malloc_gpu_var<strict_fp_t>(this->num_cells);
   strict_fp_t* const p  = GDF::malloc_gpu_var<strict_fp_t>(this->num_cells);
   strict_fp_t* const Ap = GDF::malloc_gpu_var<strict_fp_t>(this->num_cells);
   strict_fp_t* const s  = GDF::malloc_gpu_var<strict_fp_t>(this->num_cells);
   strict_fp_t* const As = GDF::malloc_gpu_var<strict_fp_t>(this->num_cells);

   strict_fp_t* const p1 = GDF::malloc_gpu_var<strict_fp_t>(this->num_cells);
   strict_fp_t* const s1 = GDF::malloc_gpu_var<strict_fp_t>(this->num_cells);

   //Constants
   strict_fp_t* const alpha1 = GDF::malloc_gpu_var<strict_fp_t, true>(1);
   strict_fp_t* const omega1 = GDF::malloc_gpu_var<strict_fp_t, true>(1);
   
   strict_fp_t* const alpha = GDF::malloc_gpu_var<strict_fp_t, true>(1);
   strict_fp_t* const beta = GDF::malloc_gpu_var<strict_fp_t, true>(1);

   strict_fp_t* const temp = GDF::malloc_gpu_var<strict_fp_t, true>(1);

   // r0 = b-Ax
   strict_fp_t* const Ax = GDF::malloc_gpu_var<strict_fp_t>(this->num_cells);
   sparse_matvec(dQ.gpu_data(), Ax);
   GDF::submit_to_gpu<kg_bicgstab_axpby>(this->num_cells, 1.0, this->rhs.gpu_data(), -1.0, Ax, r0);
   GDF::free_gpu_var(Ax);

   // r = r0, p = r0
   GDF::memcpy_gpu_var(r, r0, this->num_cells);
   GDF::memcpy_gpu_var(p, r0, this->num_cells);

   for(int iter = 0; iter < num_iter; iter++)
   {
      // p1 = p
      GDF::memcpy_gpu_var(p1, p, this->num_cells);

      // alpha1 = r . r0
      dot_product(r, r0, alpha1);

      // Ap = A * p1
      sparse_matvec(p1, Ap);

      // alpha = Ap . r0
      dot_product(Ap, r0, alpha);

      alpha[0] = alpha1[0] / alpha[0];
      
      // s = r - alpha * Ap
      GDF::submit_to_gpu<kg_bicgstab_axpby>(this->num_cells, 1.0, r, -alpha[0], Ap, s);

      // s1 = s
      GDF::memcpy_gpu_var(s1, s, this->num_cells);

      // As = A * s1
      sparse_matvec(s1, As);

      // omega1 = (As . s) / (As . As)
      dot_product(As, s, omega1);
      dot_product(As, As, temp);
      omega1[0] /= temp[0];

      // x = x + alpha * p1 + omega1 * s1
      GDF::submit_to_gpu<kg_bicgstab_axpby>(this->num_cells, 1.0, dQ.gpu_data(), alpha[0], p1, dQ.gpu_data());
      GDF::submit_to_gpu<kg_bicgstab_axpby>(this->num_cells, 1.0, dQ.gpu_data(), omega1[0], s1, dQ.gpu_data());
      // r = s - omega1 * As
      GDF::submit_to_gpu<kg_bicgstab_axpby>(this->num_cells, 1.0, s, -omega1[0], As, r);

      // beta = (r . r0) * alpha / alpha1 / omega1
      dot_product(r, r0, beta);
      beta[0] *= alpha[0] / alpha1[0] / omega1[0];

      // p = r + beta * (p - omega1 * Ap)
      GDF::submit_to_gpu<kg_bicgstab_axpby>(this->num_cells, 1.0, r, beta[0], p, p);
      GDF::submit_to_gpu<kg_bicgstab_axpby>(this->num_cells, 1.0, p, -(beta[0]*omega1[0]), Ap, p);
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

void Solver_base_gpu::allocate_memory(const bool implicit, const Grid & grid)
{
   // This is the unique face version of the Solver class.
   // Let's first allocate memory in the base class.
   m_implicit = implicit;

   this->num_cells = grid.get_number_of_cells();
   this->num_faces = grid.get_number_of_faces();
   this->num_boundary_faces = grid.get_number_of_boundary_faces();
   this->num_boundary = grid.get_number_of_boundaries();

   xcen.resize(2 * this->num_cells);

   boundary_normal.resize(2 * this->num_boundary_faces);
   boundary_xcen.resize(2 * this->num_boundary_faces);

   boundary_type_start_and_end_index.resize(this->num_boundary + 1);

   number_of_neighbors.resize(this->num_cells + 1);
   normal.resize(2 * this->num_faces);
}

void Solver_base_gpu::copy_grid_data (const Grid & grid)
{
   // Solver_base_gpu::copy_grid_data(grid);
   VectorRead<strict_fp_t>& c_xcen = grid.get_xcen();
   CellRead<strict_fp_t>& c_volume = grid.get_volume();
   BoundaryRead<int>& c_boundary_face_to_cell = grid.get_boundary_face_to_cell();
   VectorRead<strict_fp_t>& c_boundary_xcen = grid.get_boundary_xcen();
   VectorRead<strict_fp_t>& c_boundary_normal = grid.get_boundary_normal();
   BoundaryRead<strict_fp_t>& c_boundary_area = grid.get_boundary_area();
   VectorRead<int>& c_boundary_type_start_and_end_index = grid.get_boundary_type_start_and_end_index();

   for (unsigned int i = 0; i < (2*this->num_cells); i++)
   {
      xcen[i] = c_xcen[i];
   }

   for (unsigned int i = 0; i < this->num_cells; i++)
   {
      volume[i] = c_volume[i];
   }

   for (unsigned int i = 0; i < this->num_boundary_faces; i++)
   {
      boundary_face_to_cell[i] = c_boundary_face_to_cell[i];
      boundary_area[i] = c_boundary_area[i];
      boundary_normal[2 * i] = c_boundary_normal[2 * i];
      boundary_normal[(2 * i) + 1] = c_boundary_normal[(2 * i) + 1];
      boundary_xcen[2 * i] = c_boundary_xcen[2 * i];
      boundary_xcen[(2 * i) + 1] = c_boundary_xcen[(2 * i) + 1];
   }

   for (unsigned int i = 0; i < this->num_boundary; i++)
   {
      boundary_type_start_and_end_index[i] =
         c_boundary_type_start_and_end_index[i];
      boundary_type_start_and_end_index[i+1] =
         c_boundary_type_start_and_end_index[i+1];
   }



   VectorRead<int>& c_number_of_neighbors = grid.get_number_of_neighbors();

   for (unsigned int i = 0; i < (this->num_cells+1); i++)
      number_of_neighbors[i] = c_number_of_neighbors[i];

   FaceRead<int>& c_cell_neighbors = grid.get_cell_neighbors();

   for (unsigned int i = 0; i < this->num_faces; i++)
      cell_neighbors[i] = c_cell_neighbors[i];

   FaceRead<strict_fp_t>& c_area = grid.get_area();
   VectorRead<strict_fp_t>& c_normal = grid.get_normal();

   for (unsigned int i = 0; i < this->num_faces; i++)
   {
      area[i] = c_area[i];

      normal[2 * i] = c_normal[2 * i];
      normal[(2 * i) + 1] = c_normal[(2 * i) + 1];
   }

   if (m_implicit == true)
   {
      const int nrow = grid.get_number_of_cells();

      ia.resize(nrow + 1);

      // Count the neighbors or non-zeros for each cell or row (number_of_neighbors already has this information)
      for (unsigned int i = 0; i < nrow; i++)
      {
         ia[i + 1] = number_of_neighbors[i + 1] - number_of_neighbors[i] + 1; // +1 for diagonal
      }
      // ia array contains number of non-zeros for each. Those need to be summed-up to get the ia array values.
      for (unsigned int i = 1; i < nrow + 1; i++)
      {
         ia[i] = ia[i] + ia[i - 1];
      }

      this->nnz = ia[nrow];

      ja.resize(this->nnz);
      A_data.resize(this->nnz);

      std::vector<int> temp(nrow, 0);

      // Fill the ja array. First is the diagonal entry
      for (unsigned int i = 0; i < nrow; i++)
      {
         const int left_index = ia[i] + temp[i];
         temp[i]++;
         ja[left_index] = i;
      }

      for (unsigned int i = 0; i < nrow; i++)
      {
         for (int j = number_of_neighbors[i]; j < number_of_neighbors[i + 1]; j++)
         {
            const int left = i;
            const int right = cell_neighbors[j];

            const int left_index = ia[left] + temp[left];
            temp[left]++;
            ja[left_index] = right;

            if (left_index >= ia[left + 1])
            {
               printf("Problem in filling the ja array for the matrix.\n");
               printf("Cell %d: left_index %d ia %d ia %d\n", left, left_index, ia[left], ia[left + 1]);
            }
         }
      }

      for(unsigned int kk = 0; kk < this->num_cells; kk++)
      {
         csr_diag_idx[kk] = ia[kk];
         for (int nbr_count = number_of_neighbors[kk]; nbr_count < number_of_neighbors[kk + 1]; nbr_count++)
         {
            const int nbr_kk = cell_neighbors[nbr_count];
            for(unsigned int i = ia[kk]+1; i < ia[kk + 1]; i++)
            {
               if(ja[i] == nbr_kk)
               {
                  csr_idx[nbr_count] = i;
               }
            }
         }
      }
   }
}

class kg_sum_interior_face_areas
{
public:
   kg_sum_interior_face_areas(const int num_cells, VectorGPURead<int>& number_of_neighbors, FaceGPURead<strict_fp_t>& area, strict_fp_t* const data):
      gpu_num_cells(num_cells),
      gpu_number_of_neighbors(number_of_neighbors),
      gpu_area(area),
      gpu_data(data)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (int i = idx; i < gpu_num_cells; i += stride)
      {
         for (unsigned int j = gpu_number_of_neighbors[i]; j < gpu_number_of_neighbors[i + 1]; j++)
         {
            const int left = i;
            gpu_data[left] += gpu_area[j];
         }
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_cells, gpu_number_of_neighbors, gpu_area, gpu_data);
   }

private:
   const int gpu_num_cells;
   VectorGPURead<int> gpu_number_of_neighbors;
   FaceGPURead<strict_fp_t> gpu_area;
   strict_fp_t* const gpu_data;
};

class kg_sum_boundary_face_areas
{
public:
   kg_sum_boundary_face_areas(const int num_boundary_faces, BoundaryGPURead<int>& boundary_face_to_cell, BoundaryGPURead<strict_fp_t>& boundary_area, strict_fp_t* const data):
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
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_boundary_faces, gpu_boundary_face_to_cell, gpu_boundary_area, gpu_data);
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
   kg_find_min_inverse_length_scale(const int num_cells, CellGPURead<strict_fp_t>& volume, strict_fp_t* const data, strict_fp_t* const data_min):
      gpu_num_cells(num_cells),
      gpu_volume(volume),
      gpu_data(data),
      gpu_data_min(data_min)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for(unsigned int i = idx; i < gpu_num_cells; i += stride)
      {
         gpu_data[i] = gpu_data[i] / (gpu_volume[i] * 2);
         GDF::atomic_min(*gpu_data_min, gpu_data[i]);
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_cells, gpu_volume, gpu_data, gpu_data_min);
   }

private:
   const int gpu_num_cells;
   CellGPURead<strict_fp_t> gpu_volume;
   strict_fp_t* const gpu_data;
   strict_fp_t* const gpu_data_min;
};

void Solver_base_gpu::compute_time_step ()
{
   strict_fp_t* data = GDF::malloc_gpu_var<strict_fp_t>(this->num_cells);
   GDF::memset_gpu_var(data, 0, this->num_cells);

   GDF::submit_to_gpu<kg_sum_interior_face_areas>(this->num_cells, number_of_neighbors, area, data);

   GDF::submit_to_gpu<kg_sum_boundary_face_areas>(this->num_boundary_faces, boundary_face_to_cell, boundary_area, data);

   strict_fp_t* const min_value = GDF::malloc_gpu_var<strict_fp_t, true>(1);
   min_value[0] = 1.e30;
   GDF::submit_to_gpu<kg_find_min_inverse_length_scale>(this->num_cells, volume, data, min_value);

   // This is the limit. (dt / dx2) + (dt / dy2) < 0.5
   delta_t = 1.0 / (min_value[0] * min_value[0] * 2.0);     // 2.0 added to reduce from limit.

   if (m_implicit == true)
      delta_t = delta_t * 10.0;

   printf("GPU: Value of dt %e\n", delta_t);

   GDF::free_gpu_var(data);
   GDF::free_gpu_var(min_value);
}

class kg_compute_system_time_term
{
public:
   kg_compute_system_time_term(const int num_cells, const strict_fp_t delta_t, CellGPURead<int>& csr_diag_idx, CellGPURead<strict_fp_t>& volume, VectorGPU<strict_fp_t>& A_data):
      gpu_num_cells(num_cells),
      gpu_delta_t(delta_t),
      gpu_csr_diag_idx(csr_diag_idx),
      gpu_volume(volume),
      gpu_A_data(A_data)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (unsigned int i = idx; i < gpu_num_cells; i += stride)
      {
         const int crs_index = gpu_csr_diag_idx[i];
         gpu_A_data[crs_index] += gpu_volume[i] / gpu_delta_t;
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_cells, gpu_delta_t, gpu_csr_diag_idx, gpu_volume, gpu_A_data);
   }

private:
   const int gpu_num_cells;
   const strict_fp_t gpu_delta_t;
   CellGPURead<int> gpu_csr_diag_idx;
   CellGPURead<strict_fp_t> gpu_volume;
   mutable VectorGPU<strict_fp_t> gpu_A_data;
};

class kg_compute_system_interior_diffution
{
public:
   kg_compute_system_interior_diffution(const int num_cells,
                                        VectorGPURead<int>& number_of_neighbors,
                                        FaceGPURead<int>& cell_neighbors,
                                        VectorGPURead<strict_fp_t>& normal,
                                        VectorGPURead<strict_fp_t>& xcen,
                                        CellGPURead<strict_fp_t>& Q_cell,
                                        FaceGPURead<strict_fp_t>& area,
                                        CellGPURead<int>& csr_diag_idx,
                                        FaceGPURead<int>& csr_idx,
                                        VectorGPU<strict_fp_t>& A_data,
                                        CellGPU<strict_fp_t>& residual):
      gpu_num_cells(num_cells),
      gpu_number_of_neighbors(number_of_neighbors),
      gpu_cell_neighbors(cell_neighbors),
      gpu_normal(normal),
      gpu_xcen(xcen),
      gpu_Q_cell(Q_cell),
      gpu_area(area),
      gpu_csr_diag_idx(csr_diag_idx),
      gpu_csr_idx(csr_idx),
      gpu_A_data(A_data),
      gpu_residual(residual)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (unsigned int i = idx; i < gpu_num_cells; i += stride)
      {
         for (int j = gpu_number_of_neighbors[i]; j < gpu_number_of_neighbors[i + 1]; j++)
         {
            const int left = i;
            const int right = gpu_cell_neighbors[j];

            strict_fp_t distance;
            if (std::abs(gpu_normal[2 * j]) > 0.1)      // x face
            {
               distance = std::abs(gpu_xcen[2 * right] - gpu_xcen[2 * left]);
            }
            else
            {
               distance = std::abs(gpu_xcen[(2 * right) + 1] - gpu_xcen[(2 * left) + 1]);
            }
            const strict_fp_t grad_Q = (gpu_Q_cell[right] - gpu_Q_cell[left]) / distance;

            strict_fp_t face_value = grad_Q * gpu_area[j];

            gpu_residual[left] -= face_value;

            const strict_fp_t temp = 1.0 / distance * gpu_area[j];

            // Left cell
            int crs_index = gpu_csr_diag_idx[left];
            GDF::atomic_add(gpu_A_data[crs_index], temp);
            // gpu_A_data[crs_index] += temp;

            crs_index = gpu_csr_idx[j];
            GDF::atomic_sub(gpu_A_data[crs_index], temp);
            // gpu_A_data[crs_index] -= temp;
         }
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_cells, gpu_number_of_neighbors, gpu_cell_neighbors, gpu_normal, gpu_xcen,
                                               gpu_Q_cell, gpu_area, gpu_csr_diag_idx, gpu_csr_idx, gpu_A_data, gpu_residual);
   }

private:
   const int gpu_num_cells;
   VectorGPURead<int> gpu_number_of_neighbors;
   FaceGPURead<int> gpu_cell_neighbors;
   VectorGPURead<strict_fp_t> gpu_normal;
   VectorGPURead<strict_fp_t> gpu_xcen;
   CellGPURead<strict_fp_t> gpu_Q_cell;
   FaceGPURead<strict_fp_t> gpu_area;
   CellGPURead<int> gpu_csr_diag_idx;
   FaceGPURead<int> gpu_csr_idx;
   mutable VectorGPU<strict_fp_t> gpu_A_data;
   mutable CellGPU<strict_fp_t> gpu_residual;
};

class kg_compute_system_boundary_diffution
{
public:
   kg_compute_system_boundary_diffution(const int num_boundary_faces,
                                        BoundaryGPURead<int>& boundary_face_to_cell,
                                        VectorGPURead<strict_fp_t>& boundary_normal,
                                        VectorGPURead<strict_fp_t>& boundary_xcen,
                                        VectorGPURead<strict_fp_t>& xcen,
                                        BoundaryGPURead<strict_fp_t>& Q_boundary,
                                        CellGPURead<strict_fp_t>& Q_cell,
                                        BoundaryGPURead<strict_fp_t>& boundary_area,
                                        CellGPURead<int>& csr_diag_idx,
                                        VectorGPU<strict_fp_t>& A_data,
                                        CellGPU<strict_fp_t>& residual):
      gpu_num_boundary_faces(num_boundary_faces),
      gpu_boundary_face_to_cell(boundary_face_to_cell),
      gpu_boundary_normal(boundary_normal),
      gpu_boundary_xcen(boundary_xcen),
      gpu_xcen(xcen),
      gpu_Q_boundary(Q_boundary),
      gpu_Q_cell(Q_cell),
      gpu_boundary_area(boundary_area),
      gpu_csr_diag_idx(csr_diag_idx),
      gpu_A_data(A_data),
      gpu_residual(residual)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (unsigned int i = idx; i < gpu_num_boundary_faces; i += stride)
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
         const strict_fp_t grad_Q = (gpu_Q_boundary[i] - gpu_Q_cell[cell]) / distance;

         const strict_fp_t face_value = grad_Q * gpu_boundary_area[i];

         GDF::atomic_sub(gpu_residual[cell], face_value);

         const strict_fp_t temp = 1.0 / distance * gpu_boundary_area[i];

         // Left cell
         int crs_index = gpu_csr_diag_idx[cell];
         GDF::atomic_add(gpu_A_data[crs_index], temp);
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_boundary_faces, gpu_boundary_face_to_cell, gpu_boundary_normal, gpu_boundary_xcen, gpu_xcen,
                                               gpu_Q_boundary, gpu_Q_cell, gpu_boundary_area, gpu_csr_diag_idx, gpu_A_data, gpu_residual);
   }

private:
   const int gpu_num_boundary_faces;
   BoundaryGPURead<int> gpu_boundary_face_to_cell;
   VectorGPURead<strict_fp_t> gpu_boundary_normal;
   VectorGPURead<strict_fp_t> gpu_boundary_xcen;
   VectorGPURead<strict_fp_t> gpu_xcen;
   BoundaryGPURead<strict_fp_t> gpu_Q_boundary;
   CellGPURead<strict_fp_t> gpu_Q_cell;
   BoundaryGPURead<strict_fp_t> gpu_boundary_area;
   CellGPURead<int> gpu_csr_diag_idx;
   mutable VectorGPU<strict_fp_t> gpu_A_data;
   mutable CellGPU<strict_fp_t> gpu_residual;
};

class kg_compute_system_compute_residual_norm
{
public:
   kg_compute_system_compute_residual_norm(const int num_cells, CellGPURead<strict_fp_t>& residual, strict_fp_t* const residual_norm):
      gpu_num_cells(num_cells),
      gpu_residual(residual),
      gpu_residual_norm(residual_norm)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (unsigned int i = idx; i < gpu_num_cells; i += stride)
      {
         GDF::atomic_add(gpu_residual_norm[0], sycl::fabs(gpu_residual[i]));
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_cells, gpu_residual, gpu_residual_norm);
   }

private:
   const int gpu_num_cells;
   CellGPURead<strict_fp_t> gpu_residual;
   strict_fp_t* const gpu_residual_norm;
};

class kg_compute_system_add_residual_to_rhs
{
public:
   kg_compute_system_add_residual_to_rhs(const int num_cells, CellGPURead<strict_fp_t>& residual, CellGPU<strict_fp_t>& rhs):
      gpu_num_cells(num_cells),
      gpu_residual(residual),
      gpu_rhs(rhs)
   {}

   void operator() (sycl::nd_item<3> item) const
   {
      size_t idx = GDF::get_1d_index(item);
      size_t stride = GDF::get_1d_stride(item);
      for (unsigned int i = idx; i < gpu_num_cells; i += stride)
      {
         gpu_rhs[i] = -gpu_residual[i];
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      return GDF::transfer_vars_to_gpu_impl<N>(gpu_num_cells, gpu_residual, gpu_rhs);
   }

private:
   const int gpu_num_cells;
   CellGPURead<strict_fp_t> gpu_residual;
   mutable CellGPU<strict_fp_t> gpu_rhs;
};

void Solver_base_gpu::compute_system ()
{
   GDF::transfer_to_gpu_noinit(A_data);
   GDF::transfer_to_gpu_noinit(residual);
   GDF::memset_gpu_var(A_data.gpu_data(), 0, this->nnz);
   GDF::memset_gpu_var(residual.gpu_data(), 0, this->num_cells);

   GDF::submit_to_gpu<kg_compute_system_time_term>(this->num_cells, delta_t, csr_diag_idx, volume, A_data);

   GDF::submit_to_gpu<kg_compute_system_interior_diffution>(this->num_cells,
                                                            number_of_neighbors,
                                                            cell_neighbors,
                                                            normal,
                                                            xcen,
                                                            Q_cell,
                                                            area,
                                                            csr_diag_idx,
                                                            csr_idx,
                                                            A_data,
                                                            residual);
   
   
   GDF::submit_to_gpu<kg_compute_system_boundary_diffution>(this->num_boundary_faces,
                                                            boundary_face_to_cell,
                                                            boundary_normal,
                                                            boundary_xcen,
                                                            xcen,
                                                            Q_boundary,
                                                            Q_cell,
                                                            boundary_area,
                                                            csr_diag_idx,
                                                            A_data,
                                                            residual);
   
   strict_fp_t* const residual_norm = GDF::malloc_gpu_var<strict_fp_t, true>(1);
   residual_norm[0] = 0.0;
   GDF::submit_to_gpu<kg_compute_system_compute_residual_norm>(this->num_cells,
                                                               residual,
                                                               residual_norm);
   this->residual_norm = residual_norm[0];

   GDF::submit_to_gpu<kg_compute_system_add_residual_to_rhs>(this->num_cells,
                                                            residual,
                                                            rhs);
   
   GDF::free_gpu_var(residual_norm);
}
