 #include <vector>
#include <cmath>
#include <cstdio>

#include "solver.h"
#include "silo.h"
#include "silo_fwd.h"
#include "fp_data_types.h"

Solver_base::Solver_base():
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
{}

void Solver_base::set_boundary_conditions (const strict_fp_t QL, const strict_fp_t QR,
                                          const strict_fp_t QB, const strict_fp_t QT)
{
   {  // Left boundary
      const int boundary = 2;
      for (unsigned int i = boundary_type_start_and_end_index[boundary];
           i < boundary_type_start_and_end_index[boundary+1]; i++)
      {
         Q_boundary[i] = QL;
      }
   }
   {  // Right boundary
      const int boundary = 3;
      for (unsigned int i = boundary_type_start_and_end_index[boundary];
           i < boundary_type_start_and_end_index[boundary+1]; i++)
      {
         Q_boundary[i] = QR;
      }
   }
   {  // Bottom boundary
      const int boundary = 0;
      for (unsigned int i = boundary_type_start_and_end_index[boundary];
           i < boundary_type_start_and_end_index[boundary+1]; i++)
      {
         Q_boundary[i] = QB;
      }
   }
   {  // Top boundary
      const int boundary = 1;
      for (unsigned int i = boundary_type_start_and_end_index[boundary];
           i < boundary_type_start_and_end_index[boundary+1]; i++)
      {
         Q_boundary[i] = QT;
      }
   }
}

void Solver_base::initialize_solution (const strict_fp_t Q_initial)
{
   for (unsigned int i = 0; i < this->num_cells; i++)
   {
      Q_cell[i] = Q_initial;
   }
}

void Solver_base::print_residual ()
{
   for (unsigned int i = 0; i < this->num_cells; i++)
   {
      printf("Cell %d, residual %e\n", i, residual[i]);
   }
}

void Solver_base::print_solution ()
{
   for (unsigned int i = 0; i < this->num_cells; i++)
   {
      printf("Cell %d, solution %e\n", i, Q_cell[i]);
   }
}

void Solver_base::update_solution (const int solver_type, const int num_iter)
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
         bicgstab_linear_solver(num_iter);
      }
      
      for (unsigned int i = 0; i < this->num_cells; i++)
      {
         Q_cell[i] += dQ[i];
      }
   }
}

void Solver_base::write_solution(std::string file_name)
{
   FILE* file = fopen(file_name.c_str(), "w");

   for(int i = 0; i < this->num_cells; i++)
   {
      fprintf(file, "%0.16f, %.16f, %.16f\n", this->xcen[2*i], this->xcen[2*i+1], this->Q_cell[i]);
   }

   fclose(file);
}

void Solver_base::write_residual(std::string file_name, std::vector<strict_fp_t>& residual_norm)
{
   FILE* file = fopen(file_name.c_str(), "w");

   for(int i = 0; i < residual_norm.size(); i++)
   {
      fprintf(file, "%0.16f\n", residual_norm[i]);
   }

   fclose(file);
}

void Solver_base::jacobi_linear_solver(const int num_iter)
{
   for (unsigned int i = 0; i < this->num_cells; i++)
   {
      dQ_old[i] = 0.0;
   }

   for (unsigned int iter = 0; iter < num_iter; iter++)
   {
      for (unsigned int i = 0; i < this->num_cells; i++)
      {
         strict_fp_t temp = rhs[i];
         // First entry is the diagonal.
         const strict_fp_t diag_value = A_data[ia[i]];
         // Transfer all the off diagonals to right hand side
         for (int j = ia[i] + 1; j < ia[i + 1]; j++)     // Skipping first one as it is the diagonal
         {
            temp -=  A_data[j] * dQ_old[ja[j]];
         }

         dQ[i] = temp / diag_value;
      }

      for (unsigned int i = 0; i < this->num_cells; i++)
      {
         dQ_old[i] = dQ[i];
      }
   }
}

void Solver_base::matrix_vector_multiply(const strict_fp_t* const vec_in, strict_fp_t* const vec_out)
{
   memset(vec_out, 0, this->num_cells * sizeof(strict_fp_t));

   for(int i = 0; i < this->num_cells; i++)
   {
      for(int j = this->ia[i]; j < this->ia[i+1]; j++)
      {
         vec_out[i] += this->A_data[j] * vec_in[this->ja[j]];
      }
   }
}

strict_fp_t Solver_base::dot_product(const strict_fp_t* const x1, const strict_fp_t* const x2)
{
   strict_fp_t result = 0.0;

   for(int i = 0; i < this->num_cells; i++)
   {
      result += x1[i] * x2[i];
   }

   return result;
}

void Solver_base::bicgstab_linear_solver(const int num_iter)
{
   memset(dQ.cpu_data(), 0, this->num_cells * sizeof(strict_fp_t));

   //Memory Allocations
   strict_fp_t* r0 = new strict_fp_t [this->num_cells];
   strict_fp_t* r  = new strict_fp_t [this->num_cells];
   strict_fp_t* p  = new strict_fp_t [this->num_cells];
   strict_fp_t* Ap = new strict_fp_t [this->num_cells];
   strict_fp_t* s  = new strict_fp_t [this->num_cells];
   strict_fp_t* As = new strict_fp_t [this->num_cells];

   strict_fp_t* p1 = new strict_fp_t [this->num_cells];
   strict_fp_t* s1 = new strict_fp_t [this->num_cells];

   //Constants
   strict_fp_t alpha  = 0;
   strict_fp_t alpha1 = 0;
   strict_fp_t omega1  = 0;
   strict_fp_t beta   = 0;


   strict_fp_t* const Ax = new strict_fp_t[this->num_cells];
   matrix_vector_multiply(dQ.cpu_data(), Ax);
   for(int i = 0; i < this->num_cells; i++)
   {
      r0[i] = this->rhs[i]-Ax[i];
   }
   delete[] Ax;

   memcpy(r, r0, this->num_cells * sizeof(strict_fp_t));
   memcpy(p, r0, this->num_cells * sizeof(strict_fp_t));

   for(int iter = 0; iter < num_iter; iter++)
   {
      memcpy(p1, p, this->num_cells * sizeof(strict_fp_t)); // no preconditioner

      alpha1 = dot_product(r, r0);

      matrix_vector_multiply(p1, Ap);

      alpha = dot_product(Ap, r0);

      alpha = alpha1/alpha;

      for(int i = 0; i < this->num_cells; i++)
      {
         s[i] = r[i] - alpha * Ap[i];
      }

      memcpy(s1, s, this->num_cells * sizeof(strict_fp_t)); // no preconditoner

      matrix_vector_multiply(s1, As);

      omega1 = dot_product(As, s);
      omega1 /= dot_product(As, As);

      for(int i = 0; i < this->num_cells; i++)
      {
         dQ[i] = dQ[i] + alpha*p1[i] + omega1*s1[i];
         r[i] = s[i] - omega1*As[i];
      }

      beta = dot_product(r, r0)/alpha1;
      beta *= alpha/omega1;

      for(int i = 0; i < this->num_cells; i++)
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

void Solver_base::allocate_memory(const bool implicit, const Grid & grid)
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

void Solver_base::copy_grid_data (const Grid & grid)
{
   // Solver_base::copy_grid_data(grid);
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

void Solver_base::compute_time_step ()
{
   std::vector<strict_fp_t> data (this->num_cells, 0.0);

   for (unsigned int i = 0; i < this->num_cells; i++)
   {
      for (int j = number_of_neighbors[i]; j < number_of_neighbors[i + 1]; j++)
      {
         const int left = i;
         //         const int right = cell_neighbors[j];
         data[left] += area[j];
      }
   }

   for (unsigned int i = 0; i < this->num_boundary_faces; i++)
   {
      const int cell = boundary_face_to_cell[i];

      data[cell] += boundary_area[i];
   }

   strict_fp_t min_value = 1.e30;

   for (unsigned int i = 0; i < this->num_cells; i++)
   {
      data[i] = data[i] / (volume[i] * 2);

      if (data[i] < min_value)
         min_value = data[i];
   }

   // This is the limit. (dt / dx2) + (dt / dy2) < 0.5
   delta_t = 1.0 / (min_value * min_value * 2.0);     // 2.0 added to reduce from limit.

   if (m_implicit == true)
      delta_t = delta_t * 10.0;

   printf("CPU: Value of dt %e\n", delta_t);
}

void Solver_base::compute_system ()
{
   if (m_implicit == true)
   {
      for (unsigned int i = 0; i < this->nnz; i++)
      {
         A_data[i] = 0.0;
      }
      for (unsigned int i = 0; i < this->num_cells; i++)
      {
         residual[i] = 0.0;
      }

      // //Time term
      for (unsigned int i = 0; i < this->num_cells; i++)
      {
         const int crs_index = csr_diag_idx[i];
         A_data[crs_index] += volume[i] / delta_t;
      }

      //Diffusion term for internal cells
      for (unsigned int i = 0; i < this->num_cells; i++)
      {
         for (int j = number_of_neighbors[i]; j < number_of_neighbors[i + 1]; j++)
         {
            const int left = i;
            const int right = cell_neighbors[j];

            strict_fp_t distance;
            if (std::abs(normal[2 * j]) > 0.1)      // x face
            {
               distance = std::abs(xcen[2 * right] - xcen[2 * left]);
            }
            else
            {
               distance = std::abs(xcen[(2 * right) + 1] - xcen[(2 * left) + 1]);
            }
            const strict_fp_t grad_Q = (Q_cell[right] - Q_cell[left]) / distance;

            strict_fp_t face_value = grad_Q * area[j];

            residual[left] -= face_value;

            const strict_fp_t temp = 1.0 / distance * area[j];

            // Left cell
            int crs_index = csr_diag_idx[left];
            A_data[crs_index] += temp;

            crs_index = csr_idx[j];
            A_data[crs_index] -= temp;
         }
      }

      //Diffusion term for boundary cells
      for (unsigned int i = 0; i < this->num_boundary_faces; i++)
      {
         const int cell = boundary_face_to_cell[i];

         strict_fp_t distance;
         if (std::abs(boundary_normal[2 * i]) > 0.1)      // x face
         {
            distance = std::abs(boundary_xcen[2 * i] - xcen[2 * cell]);
         }
         else
         {
            distance = std::abs(boundary_xcen[(2 * i) + 1] - xcen[(2 * cell) + 1]);
         }
         const strict_fp_t grad_Q = (Q_boundary[i] - Q_cell[cell]) / distance;

         const strict_fp_t face_value = grad_Q * boundary_area[i];

         residual[cell] -= face_value;

         const strict_fp_t temp = 1.0 / distance * boundary_area[i];

         // Left cell
         int crs_index = csr_diag_idx[cell];
         A_data[crs_index] += temp;
      }

      compute_residual_norm();
      for (unsigned int i = 0; i < this->num_cells; i++)
      {
         rhs[i] = -residual[i];
      }
   }
}