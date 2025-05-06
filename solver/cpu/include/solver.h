#ifndef SOLVER_H
#define SOLVER_H

#include "grid.h"
#include "mpiclass.h"

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
   void allocate_variables();

   void setup_matrix_struct(const int num_solved, const int num_involved);
   
   void set_boundary_conditions(const strict_fp_t QL, const strict_fp_t QR, const strict_fp_t QB, const strict_fp_t QT);
   
   void initialize_solution(const int num_solved, const strict_fp_t Q_initial);
   
   void compute_time_step(const int num_solved, const int num_attached);
   
   void compute_system(const int num_solved, const int num_attached);
   
   void update_solution(const int num_solved);

   void compute_rdist(const int num_solved, const int num_attached);

   void compute_residual(const int num_solved, const int num_attached);

   void write_solution(const int num_solved, const int num_cells, std::string file_name);

   void write_residual(std::string file_name, std::vector<strict_fp_t>& residual_norm);
   
   strict_fp_t print_residual_norm(const int time_iter);

   strict_fp_t get_residual_norm();

   void mpi_nbr_communication(strict_fp_t* vecg);     

   void compute_residual_norm(const int num_solved);

   void jacobi_linear_solver(const int num_solved);

   void matrix_vector_multiply(const int num_solved, VectorRead<int>& ia_local, VectorRead<int>& ja_local, VectorRead<strict_fp_t>& A_data_local, strict_fp_t* vec_in, strict_fp_t* const vec_out);

   strict_fp_t dot_product(const size_t num_elements, const strict_fp_t* const x1, const strict_fp_t* const x2);

   void bicgstab_linear_solver();

   bool continue_iterations(const uint64_t cur_iter);

   Solver_base();

   int nnz_local;
   int nrow_local;
   int ncol_local;

   strict_fp_t residual_norm;
   strict_fp_t inital_residual_norm;
   strict_fp_t delta_t;
};

#endif /* SOLVER_H */

