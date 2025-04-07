#ifndef SOLVER_H
#define SOLVER_H

#include "grid.h"

#include <vector>
#include <array>
#include <cstdio>
#include <cstdlib>

#include "silo.h"
#include "silo_fwd.h"
#include "fp_data_types.h"


class Solver_base
{
public:
   void allocate_memory(const bool implicit, const Grid & grid);
   
   void copy_grid_data(const Grid & grid);
   
   void set_boundary_conditions(const strict_fp_t QL, const strict_fp_t QR, const strict_fp_t QB, const strict_fp_t QT);
   
   void initialize_solution(const strict_fp_t Q_initial);
   
   void compute_time_step();
   
   void compute_system();
   
   void update_solution(const int solver_type, const int num_iter);

   void write_solution(std::string file_name);

   void write_residual(std::string file_name, std::vector<strict_fp_t>& residual_norm);
   
   void print_residual();
   
   void print_solution();
   
   strict_fp_t print_residual_norm(const int time_iter)
   {
      printf("Time iter: %d, Residual %0.16e\n", time_iter, residual_norm);

      return residual_norm;
   }

   Solver_base();
   
protected:
   int num_cells;
   int num_faces;
   int num_boundary_faces;
   int num_boundary;

   // Grid data
   Vector<strict_fp_t> xcen;          // 2 * num_cells
   Cell<strict_fp_t> volume;        // num_cells
   
   Boundary<int> boundary_face_to_cell;     // num_boundary_faces
   Boundary<strict_fp_t> boundary_area;          // num_boundary_faces
   Vector<strict_fp_t> boundary_normal;        // 2 * num_boundary_faces
   Vector<strict_fp_t> boundary_xcen;          // 2 * num_boundary_faces
   Vector<int> boundary_type_start_and_end_index;     // 2 * num_boundary

   Vector<int> number_of_neighbors;  // number_of_cells + 1
   Face<int> cell_neighbors;       // num_faces or number_of_neighbors[end]
   Face<strict_fp_t> area;            // num_faces
   Vector<strict_fp_t> normal;          // 2 * num_faces
   
   // Matrix data (for implicit)
   int nnz;
   Vector<int> ia;
   Vector<int> ja;
   Face<int> csr_idx;
   Cell<int> csr_diag_idx;
   Vector<strict_fp_t> A_data;
   Cell<strict_fp_t> rhs;
   Cell<strict_fp_t> dQ;
   Cell<strict_fp_t> dQ_old;
   
   Cell<strict_fp_t> Q_cell;
   Boundary<strict_fp_t> Q_boundary;
   Cell<strict_fp_t> residual;
   
   strict_fp_t residual_norm;
   
   strict_fp_t delta_t;
   bool m_implicit;
   
   void jacobi_linear_solver (const int num_iter);

   void matrix_vector_multiply(const strict_fp_t* const vec_in, strict_fp_t* const vec_out);
   strict_fp_t dot_product(const strict_fp_t* const x1, const strict_fp_t* const x2);
   void bicgstab_linear_solver(const int num_iter);
   
   void compute_residual_norm ()
   {
      residual_norm = 0.0;
      for (unsigned int i = 0; i < this->num_cells; i++)
      {
         residual_norm += std::abs(residual[i]);
         // residual_norm += residual[i] * residual[i];
      }
   }
};

#endif /* SOLVER_H */

