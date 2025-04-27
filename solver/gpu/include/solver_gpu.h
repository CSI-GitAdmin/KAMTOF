#ifndef SOLVER_GPU_H
#define SOLVER_GPU_H

#include "grid.h"

#include <vector>
#include <array>
#include <cstdio>
#include <cstdlib>

#include "silo.h"
#include "silo_fwd.h"
#include "fp_data_types.h"

#include "oneMathSPMV.h"

class Solver_base_gpu
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

   void write_solution(const int num_solved,const int num_cells, std::string file_name);

   void write_residual(std::string file_name, std::vector<strict_fp_t>& residual_norm);
   
   strict_fp_t print_residual_norm(const int time_iter);

   strict_fp_t get_residual_norm();

   void jacobi_linear_solver(const int num_solved);

   void setup_m_spmv_system();

   void sparse_matvec(strict_fp_t* const vec_in, strict_fp_t* const vec_out);

   void dot_product(const size_t num_elements, const strict_fp_t* const x, const strict_fp_t* const y, strict_fp_t* const result);

   void bicgstab_linear_solver();

   Solver_base_gpu();

   ~Solver_base_gpu(){
      if(m_spmv_sys->is_setup())
         m_spmv_sys->release_system();
      delete m_spmv_sys;
      m_spmv_sys  = nullptr;
   }

   struct GDF::oneMathSPMV* m_spmv_sys;

   int nnz_local;
   int nrow_local;
   int ncol_local;

   strict_fp_t residual_norm;
   strict_fp_t delta_t;
};

#endif /* SOLVER_GPU_H */

