#include <vector>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <set>

#include "grid.h"
#include "solver.h"
#include "mpiclass.h"
#include <mpi.h>
#include "silo.h"
#include "silo_fwd.h"
#include "fp_data_types.h"

Solver_base::Solver_base()
{
   residual_norm = 1e30;
}

void Solver_base::allocate_variables()
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

void Solver_base::set_boundary_conditions (const strict_fp_t QL, const strict_fp_t QR,
                                          const strict_fp_t QB, const strict_fp_t QT)
{
   VectorRead<int> boundary_type_start_and_end_index = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("boundary_type_start_and_end_index");
   Boundary<strict_fp_t> Q_boundary_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("Q_boundary_local");
   
   {  // Left boundary
      const int boundary = 2;
      for (unsigned int i = boundary_type_start_and_end_index[boundary];
           i < boundary_type_start_and_end_index[boundary+1]; i++)
      {
         Q_boundary_local[i] = QL;
      }
   }
   {  // Right boundary
      const int boundary = 3;
      for (unsigned int i = boundary_type_start_and_end_index[boundary];
           i < boundary_type_start_and_end_index[boundary+1]; i++)
      {
         Q_boundary_local[i] = QR;
      }
   }
   {  // Bottom boundary
      const int boundary = 0;
      for (unsigned int i = boundary_type_start_and_end_index[boundary];
           i < boundary_type_start_and_end_index[boundary+1]; i++)
      {
         Q_boundary_local[i] = QB;
      }
   }
   {  // Top boundary
      const int boundary = 1;
      for (unsigned int i = boundary_type_start_and_end_index[boundary];
           i < boundary_type_start_and_end_index[boundary+1]; i++)
      {
         Q_boundary_local[i] = QT;
      }
   }
}

void Solver_base::initialize_solution(const int num_solved, const strict_fp_t Q_initial)
{
   Cell<strict_fp_t> Q_cell_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Q_cell_local");

   for (unsigned int il = 0; il < num_solved; il++)
   {
      Q_cell_local[il] = Q_initial;
   }
   mpi_nbr_communication(Q_cell_local.cpu_data());   
}

void Solver_base::update_solution(const int num_solved)
{
   Cell<strict_fp_t> Q_cell_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Q_cell_local");
   
   if(implicit_solver == false)
   {
      CellRead<strict_fp_t> volume_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("volume_local");
      CellRead<strict_fp_t> residual_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("residual_local");

      for (unsigned int i = 0; i < num_solved; i++)
      {
         Q_cell_local[i] -= residual_local[i] * delta_t / volume_local[i];
      }
   }
   else
   {
      if(solver_type == 0)
      {
         jacobi_linear_solver(num_solved);
      }
      else
      {
         bicgstab_linear_solver();
      }

      CellRead<strict_fp_t> dQ_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_local");
      
      for (unsigned int i = 0; i < num_solved; i++)
      {
         Q_cell_local[i] += dQ_local[i];
      }
   }

   mpi_nbr_communication(Q_cell_local.cpu_data());
}

void Solver_base::write_solution(const int num_solved, const int num_cells, std::string file_name)
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

void Solver_base::write_residual(std::string file_name, std::vector<strict_fp_t>& residual_norm)
{
   if(rank == 0)
   {
      FILE* file = fopen(file_name.c_str(), "w");

      for(int i = 0; i < residual_norm.size(); i++)
      {
         fprintf(file, "%0.16f\n", residual_norm[i]);
      }

      fclose(file);
   }
}

void Solver_base::jacobi_linear_solver(const int num_solved)
{
   CellRead<strict_fp_t> rhs_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("rhs_local");
   VectorRead<int> ia_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ia_local");
   VectorRead<int> ja_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ja_local");
   VectorRead<strict_fp_t> A_data_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("A_data_local");
   Cell<strict_fp_t> dQ_old_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_old_local");
   Cell<strict_fp_t> dQ_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_local");

   for (unsigned int i = 0; i < num_solved; i++)
   {
      dQ_old_local[i] = 0.0;
   }

   for (unsigned int iter = 0; iter < num_iter; iter++)
   {
      mpi_nbr_communication(dQ_old_local.cpu_data());

      for (unsigned int i = 0; i < num_solved; i++)
      {
         strict_fp_t temp = rhs_local[i];
         // First entry is the diagonal.
         const strict_fp_t diag_value = A_data_local[ia_local[i]];
         // Transfer all the off diagonals to right hand side
         for (int j = ia_local[i] + 1; j < ia_local[i + 1]; j++)     // Skipping first one as it is the diagonal
         {
            temp -=  A_data_local[j] * dQ_old_local[ja_local[j]];
         }

         dQ_local[i] = temp / diag_value;
      }

      for (unsigned int i = 0; i < num_solved; i++)
      {
         dQ_old_local[i] = dQ_local[i];
      }
   }
}

void Solver_base::matrix_vector_multiply(const int num_solved, VectorRead<int>& ia_local, VectorRead<int>& ja_local, VectorRead<strict_fp_t>& A_data_local, strict_fp_t* vec_in, strict_fp_t* const vec_out)
{
   memset(vec_out, 0, num_solved * sizeof(strict_fp_t));

   mpi_nbnb_transfer(vec_in);
   for(int i = 0; i < num_solved; i++)
   {
      for(int j = ia_local[i]; j < ia_local[i+1]; j++)
      {
         vec_out[i] += A_data_local[j] * vec_in[ja_local[j]];
      }
   }
}

strict_fp_t Solver_base::dot_product(const size_t num_elements, const strict_fp_t* const x1, const strict_fp_t* const x2)
{
   strict_fp_t result = 0.0;
   strict_fp_t result_global = 0.0;


   for(int i = 0; i < num_elements; i++)
   {
      result += x1[i] * x2[i];
   }

   MPI_Allreduce(&result, &result_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   return result_global;
}

void Solver_base::bicgstab_linear_solver()
{
   VectorRead<int> ia_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ia_local");
   VectorRead<int> ja_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ja_local");
   VectorRead<strict_fp_t> A_data_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("A_data_local");
   CellRead<strict_fp_t> rhs_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("rhs_local");
   Cell<strict_fp_t> dQ_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_local");

   memset(dQ_local.cpu_data(), 0, nrow_local * sizeof(strict_fp_t));

   //Memory Allocations
   strict_fp_t* r0 = new strict_fp_t [ncol_local];
   strict_fp_t* r  = new strict_fp_t [ncol_local];
   strict_fp_t* p  = new strict_fp_t [ncol_local];
   strict_fp_t* Ap = new strict_fp_t [ncol_local];
   strict_fp_t* s  = new strict_fp_t [ncol_local];
   strict_fp_t* As = new strict_fp_t [ncol_local];

   strict_fp_t* p1 = new strict_fp_t [ncol_local];
   strict_fp_t* s1 = new strict_fp_t [ncol_local];

   //Constants
   strict_fp_t alpha  = 0;
   strict_fp_t alpha1 = 0;
   strict_fp_t omega1  = 0;
   strict_fp_t beta   = 0;

   strict_fp_t* const Ax = new strict_fp_t[nrow_local];
   matrix_vector_multiply(nrow_local, ia_local, ja_local, A_data_local, dQ_local.cpu_data(), Ax);
   for(int i = 0; i < nrow_local; i++)
   {
      r0[i] = rhs_local[i]-Ax[i];
   }
   delete[] Ax;

   memcpy(r, r0, nrow_local * sizeof(strict_fp_t));
   memcpy(p, r0, nrow_local * sizeof(strict_fp_t));

   for(int iter = 0; iter < num_iter; iter++)
   {
      memcpy(p1, p, nrow_local * sizeof(strict_fp_t)); // no preconditioner

      alpha1 = dot_product(nrow_local, r, r0);

      matrix_vector_multiply(nrow_local, ia_local, ja_local, A_data_local, p1, Ap);

      alpha = dot_product(nrow_local, Ap, r0);
      alpha = alpha1/alpha;

      for(int i = 0; i < nrow_local; i++)
      {
         s[i] = r[i] - alpha * Ap[i];
      }

      memcpy(s1, s, nrow_local * sizeof(strict_fp_t)); // no preconditoner

      matrix_vector_multiply(nrow_local, ia_local, ja_local, A_data_local, s1, As);

      omega1 = dot_product(nrow_local, As, s);
      omega1 /= dot_product(nrow_local, As, As);

      for(int i = 0; i < nrow_local; i++)
      {
         dQ_local[i] = dQ_local[i] + alpha*p1[i] + omega1*s1[i];
         r[i] = s[i] - omega1*As[i];
      }

      beta = dot_product(nrow_local, r, r0)/alpha1;
      beta *= alpha/omega1;

      for(int i = 0; i < nrow_local; i++)
      {
         p[i] = r[i] + beta*(p[i] - omega1*Ap[i]);
      }
   }

   delete[] r;
   delete[] r0;
   delete[] p;
   delete[] Ap;
   delete[] s;
   delete[] As;
   delete[] s1;
   delete[] p1;
}

void Solver_base::setup_matrix_struct(const int num_solved, const int num_involved)
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

void Solver_base::compute_time_step(const int num_solved, const int num_attached)
{
   std::vector<strict_fp_t> data(num_solved, 0.0);

   VectorRead<int> number_of_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors_local");
   FaceRead<strict_fp_t> area_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::FACE>("area_local");
   
   for (unsigned int il = 0; il < num_solved; il++)
   {
      for (int jl = number_of_neighbors_local[il]; jl < number_of_neighbors_local[il + 1]; jl++)
      {
         const int left = il;
         data[left] += area_local[jl];
      }
   }

   BoundaryRead<int> boundary_face_to_cell_local = m_silo.retrieve_entry<int, CDF::StorageType::BOUNDARY>("boundary_face_to_cell_local");
   BoundaryRead<strict_fp_t> boundary_area_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_area_local");
   
   for (unsigned int i = 0; i < num_attached; i++)
   {
      const int cell = boundary_face_to_cell_local[i];
      data[cell] += boundary_area_local[i];
   }

   strict_fp_t min_value = 1.e30;

   CellRead<strict_fp_t> volume_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("volume_local");
   
   for (unsigned int il = 0; il < num_solved; il++)
   {
      data[il] = data[il] / (volume_local[il] * 2);

      if (data[il] < min_value)
         min_value = data[il];
   }
   MPI_Allreduce(&min_value, &min_value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   
   // This is the limit. (dt / dx2) + (dt / dy2) < 0.5
   delta_t = 1.0 / (min_value * min_value * 2.0);     // 2.0 added to reduce from limit.

   if(implicit_solver == true)
      delta_t = delta_t * 10.0;

   if(rank == 0)
      printf("CPU: Value of dt %e\n", delta_t);
}

void Solver_base::compute_system(const int num_solved, const int num_attached)
{
   if(implicit_solver == true)
   {
      Vector<strict_fp_t> A_data_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("A_data_local");
      
      for (unsigned int i = 0; i < nnz_local; i++)
      {
         A_data_local[i] = 0.0;
      }

      // Time term
      CellRead<strict_fp_t> volume_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("volume_local");
      CellRead<int> csr_diag_idx_local = m_silo.retrieve_entry<int, CDF::StorageType::CELL>("csr_diag_idx_local");
      
      for (unsigned int i = 0; i < num_solved; i++)
      {
         const int crs_index = csr_diag_idx_local[i];
         A_data_local[crs_index] += volume_local[i] / delta_t;
      }

      // Diffusion term for internal cells
      VectorRead<int> number_of_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors_local");
      FaceRead<int> csr_idx_local = m_silo.retrieve_entry<int, CDF::StorageType::FACE>("csr_idx_local");
      FaceRead<strict_fp_t> rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::FACE>("rdista_local");
      
      for (unsigned int il = 0; il < num_solved; il++)
      {
         for (int jl = number_of_neighbors_local[il]; jl < number_of_neighbors_local[il + 1]; jl++)
         {
            const int left = il;

            const strict_fp_t temp = rdista_local[jl];

            // Left cell
            int crs_index = csr_diag_idx_local[left];
            A_data_local[crs_index] += temp;
   
            crs_index = csr_idx_local[jl];
            A_data_local[crs_index] -= temp;
         }
      }

      // Diffusion term for boundary cells
      VectorRead<int> ranks_list = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ranks_list");
      VectorRead<int> global_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("global_local");
      BoundaryRead<int> boundary_face_to_cell_local = m_silo.retrieve_entry<int, CDF::StorageType::BOUNDARY>("boundary_face_to_cell_local");
      BoundaryRead<strict_fp_t> boundary_rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_rdista_local");
      
      for (unsigned int i = 0; i < num_attached; i++)
      {
         const int cell = boundary_face_to_cell_local[i];
         
         const strict_fp_t temp = boundary_rdista_local[i];;

         // Left cell
         int crs_index = csr_diag_idx_local[cell];
         A_data_local[crs_index] += temp;
      }
   }
}

void Solver_base::compute_residual(const int num_solved, const int num_attached)
{
   // Diffusion term for internal cells
   VectorRead<int> number_of_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors_local");
   FaceRead<int> cell_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::FACE>("cell_neighbors_local");
   FaceRead<strict_fp_t> rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::FACE>("rdista_local");
   CellRead<strict_fp_t> Q_cell_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Q_cell_local");
   Cell<strict_fp_t> residual_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("residual_local");
   
   for (unsigned int il = 0; il < num_solved; il++)
   {
      residual_local[il] = 0.0;
      for (int jl = number_of_neighbors_local[il]; jl < number_of_neighbors_local[il + 1]; jl++)
      {
         const int left = il;
         const int right = cell_neighbors_local[jl];
         strict_fp_t face_value = (Q_cell_local[right] - Q_cell_local[left]) * rdista_local[jl];
         residual_local[left] -= face_value;
      }
   }
   
   //Diffusion term for boundary cells
   BoundaryRead<int> boundary_face_to_cell_local = m_silo.retrieve_entry<int, CDF::StorageType::BOUNDARY>("boundary_face_to_cell_local");
   BoundaryRead<strict_fp_t> Q_boundary_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("Q_boundary_local");
   BoundaryRead<strict_fp_t> boundary_rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_rdista_local");
   
   for (unsigned int i = 0; i < num_attached; i++)
   {
      const int cell = boundary_face_to_cell_local[i];

      const strict_fp_t face_value = (Q_boundary_local[i] - Q_cell_local[cell]) * boundary_rdista_local[i];

      residual_local[cell] -= face_value;
   }
   
   compute_residual_norm(num_solved);
   
   if(implicit_solver == true)
   {
      Cell<strict_fp_t> rhs_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("rhs_local");
   
      for (unsigned int il = 0; il < num_solved; il++)
      {
         rhs_local[il] = -residual_local[il];
      }
   }
}

void Solver_base::compute_residual_norm(const int num_solved)
{
   CellRead<strict_fp_t> residual_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("residual_local");

   residual_norm = 0.0;
   strict_fp_t residual_norm_global = 0.0;
   for (unsigned int i = 0; i < num_solved; i++)
   {
      residual_norm += std::abs(residual_local[i]);
   }
   MPI_Allreduce(&residual_norm, &residual_norm_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   residual_norm = residual_norm_global;
}

void Solver_base::compute_rdist(const int num_solved, const int num_attached)
{
   // Diffusion term for interior cells
   VectorRead<int> number_of_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors_local");
   FaceRead<int> cell_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::FACE>("cell_neighbors_local");
   FaceRead<strict_fp_t> area_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::FACE>("area_local");
   VectorRead<strict_fp_t> xcen_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("xcen_local");
   VectorRead<strict_fp_t> normal_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("normal_local");
   Face<strict_fp_t> rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::FACE>("rdista_local");
   
   for (unsigned int il = 0; il < num_solved; il++)
   {
      for (int jl = number_of_neighbors_local[il]; jl < number_of_neighbors_local[il + 1]; jl++)
      {
         const int left = il;
         const int right = cell_neighbors_local[jl];

         strict_fp_t distance;
         if(std::abs(normal_local[2 * jl]) > 0.1)      // x face
         {
            distance = std::abs(xcen_local[2 * right] - xcen_local[2 * left]);
         }
         else
         {
            distance = std::abs(xcen_local[(2 * right) + 1] - xcen_local[(2 * left) + 1]);
         }
         
         rdista_local[jl] = 1.0 / distance * area_local[jl];
      }
   }

   // Diffusion term for boundary cells
   VectorRead<int> ranks_list = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ranks_list");
   VectorRead<int> global_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("global_local");
   BoundaryRead<int> boundary_face_to_cell_local = m_silo.retrieve_entry<int, CDF::StorageType::BOUNDARY>("boundary_face_to_cell_local");
   BoundaryRead<strict_fp_t> boundary_area_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_area_local");
   VectorRead<strict_fp_t> boundary_xcen_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_xcen_local");
   VectorRead<strict_fp_t> boundary_normal_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_normal_local");
   Boundary<strict_fp_t> boundary_rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_rdista_local");
   
   for (unsigned int i = 0; i < num_attached; i++)
   {
      const int cell = boundary_face_to_cell_local[i];

      strict_fp_t distance;
      if (std::abs(boundary_normal_local[2 * i]) > 0.1)      // x face
      {
         distance = std::abs(boundary_xcen_local[2 * i] - xcen_local[2 * cell]);
      }
      else
      {
         distance = std::abs(boundary_xcen_local[(2 * i) + 1] - xcen_local[(2 * cell) + 1]);
      }
      boundary_rdista_local[i] = 1.0 / distance * boundary_area_local[i];
   }
}

void Solver_base::mpi_nbr_communication(strict_fp_t* vecg)
{
   mpi_nbnb_transfer(vecg);
}

strict_fp_t Solver_base::print_residual_norm(const int time_iter)
{
   if(rank == 0)
      printf("Time iter: %d, Residual %0.16e\n", time_iter, residual_norm);

   return residual_norm;
}

strict_fp_t Solver_base::get_residual_norm()
{
   return residual_norm;
}