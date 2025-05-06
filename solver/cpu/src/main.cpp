#include "grid.h"
#include "solver.h"
#include "input_parser.h"
#include "logger.hpp"

#include <cstdlib>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "silo.h"

#ifdef ENABLE_GPU
#include "gpu_globals.h"
#include "solver_gpu.h"
#include "pagefault_handler.h"
#endif

void send_vars_to_gpu();

int main (int argc, char** argv)
{
   mpi_init(&argc, &argv);

#ifdef ENABLE_GPU
   #ifdef CPU_AUTO_TRANSFER
      setup_pagefault_handler();
   #endif
#endif

   // default name of input file
   std::string infile;

   // declate input_data_ptr
   InputParser* input_data_ptr;

   // input file name
   if (argc > 2)
   {
      if(rank == 0)
      {
         log_msg<CDF::LogLevel::WARNING>("Only one input line argument is needed. Additional input line arguments will be ignored.");
      }

      infile = argv[1];

      // initialize parser object
      input_data_ptr = new InputParser(infile);
   }
   else if (argc < 2)
   {
      if(rank == 0)
      {
         log_msg<CDF::LogLevel::WARNING>("Missing name of input file for Laplace solver. Using default values from input parser.");
         input_data_ptr  = new InputParser();
      }
   }
   else
   {
      infile = argv[1];

      // initialize parser object
      input_data_ptr = new InputParser(infile);
   }
   
   // print input options being used from the root rank
   if(rank == 0)
   {
      input_data_ptr->print_input_struct();
   }

   // create local variables to store input options
   gpu_solver            = input_data_ptr->use_gpu_solver;   // use_gpu_solver
   implicit_solver       = input_data_ptr->implicit_solver;  // use implicit_solver
   tol                   = input_data_ptr->tol_val;          // tolerance at which solver will stop
   tol_type              = input_data_ptr->tol_type;         // 0: absolute , 1: relative, 2: iteration_count
   solver_type           = input_data_ptr->solver_type;      // 0: jacobi , 1: bicgstab
   num_iter              = input_data_ptr->num_iter;         // number of inner iterations
   gpu_global_range      = input_data_ptr->gpu_global_range; // GPU global range
   gpu_local_range       = input_data_ptr->gpu_local_range;  // GPU local range

   std::vector<strict_fp_t> residual_norm;

#ifdef ENABLE_GPU
   if(gpu_solver)
      setup_gpu_globals();
#endif

   Grid grid(input_data_ptr->Nx, input_data_ptr->Ny);
   grid.allocate_variables();
   grid.generate_grid();

   if(!gpu_solver)
   {
      Solver_base* solver_ptr = new Solver_base;
      solver_ptr->allocate_variables();
      solver_ptr->setup_matrix_struct(grid.num_solved, grid.num_involved);
      solver_ptr->set_boundary_conditions(0, 0, 1e2, 0);
      solver_ptr->initialize_solution(grid.num_solved, 1.0);
      solver_ptr->compute_time_step(grid.num_solved, grid.num_attached);
      solver_ptr->compute_rdist(grid.num_solved, grid.num_attached);
      solver_ptr->compute_system(grid.num_solved, grid.num_attached);

      struct timeval s_time, e_time;
      gettimeofday(&s_time, NULL);

      solver_ptr->compute_residual(grid.num_solved, grid.num_attached);
      residual_norm.push_back(solver_ptr->print_residual_norm(0));
      solver_ptr->inital_residual_norm = solver_ptr->get_residual_norm();

      int count = 1;
      while(solver_ptr->continue_iterations(count))
      {
         solver_ptr->update_solution(grid.num_solved);
         solver_ptr->compute_residual(grid.num_solved, grid.num_attached);
         residual_norm.push_back(solver_ptr->print_residual_norm(count));
         count++;
      }

      gettimeofday(&e_time, NULL);

      const strict_fp_t time_elapsed_cpu = (e_time.tv_sec - s_time.tv_sec) +
                                      ((e_time.tv_usec - s_time.tv_usec) * 1e-6);

      if(rank == 0)
         printf("CPU time to solve %f\n", time_elapsed_cpu);

      std::string soln_file, res_file;
      soln_file = "laplace_solution_cpu_nx_" + std::to_string(input_data_ptr->Nx) + "_ny_" + std::to_string(input_data_ptr->Ny) + ".txt";
      res_file  = "laplace_residual_cpu_nx_" + std::to_string(input_data_ptr->Nx) + "_ny_" + std::to_string(input_data_ptr->Ny) + ".txt";

      solver_ptr->write_solution(grid.num_solved, grid.num_cells, soln_file);
      solver_ptr->write_residual(res_file, residual_norm);

      delete solver_ptr;
      solver_ptr = nullptr;
   }
#ifdef ENABLE_GPU
   else
   {
      Solver_base_gpu* solver_ptr_gpu = new Solver_base_gpu;
      solver_ptr_gpu->allocate_variables();
      solver_ptr_gpu->setup_matrix_struct(grid.num_solved, grid.num_involved);
#ifdef GPU_FULLY_OPTIMIZED
      send_vars_to_gpu();
#endif
      solver_ptr_gpu->set_boundary_conditions(0, 0, 1e2, 0);
      solver_ptr_gpu->initialize_solution(grid.num_solved, 1.0);
      solver_ptr_gpu->compute_time_step(grid.num_solved, grid.num_attached);
      solver_ptr_gpu->compute_rdist(grid.num_solved, grid.num_attached);
      solver_ptr_gpu->compute_system(grid.num_solved, grid.num_attached);

      struct timeval s_time, e_time;
      gettimeofday(&s_time, NULL);

      solver_ptr_gpu->compute_residual(grid.num_solved, grid.num_attached);
      residual_norm.push_back(solver_ptr_gpu->print_residual_norm(0));
      solver_ptr_gpu->inital_residual_norm = solver_ptr_gpu->get_residual_norm();
      
      int count = 1;
      while(solver_ptr_gpu->continue_iterations(count))
      {
         solver_ptr_gpu->update_solution(grid.num_solved);
         solver_ptr_gpu->compute_residual(grid.num_solved, grid.num_attached);
         residual_norm.push_back(solver_ptr_gpu->print_residual_norm(count));
         count++;
      }

      gettimeofday(&e_time, NULL);

      const strict_fp_t time_elapsed_cpu = (e_time.tv_sec - s_time.tv_sec) +
                                    ((e_time.tv_usec - s_time.tv_usec) * 1e-6);
      
      if(rank == 0)
         printf("GPU time to solve %f\n", time_elapsed_cpu);


      std::string soln_file, res_file;
      soln_file = "laplace_solution_gpu_nx_" + std::to_string(input_data_ptr->Nx) + "_ny_" + std::to_string(input_data_ptr->Ny) + ".txt";
      res_file  = "laplace_residual_gpu_nx_" + std::to_string(input_data_ptr->Nx) + "_ny_" + std::to_string(input_data_ptr->Ny) + ".txt";

      solver_ptr_gpu->write_solution(grid.num_solved, grid.num_cells, soln_file);
      solver_ptr_gpu->write_residual(res_file, residual_norm);

      delete solver_ptr_gpu;
      solver_ptr_gpu = nullptr;
   }
#endif

   delete mpiparams;

   m_silo.clear_entries();

#ifdef ENABLE_GPU
      if(gpu_solver)
         finalize_gpu_globals();
#endif

   mpi_finalize();
   
   return 0;
}

void send_vars_to_gpu()
{
   Boundary<strict_fp_t> Q_boundary_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("Q_boundary_local");
   GDF::transfer_to_gpu_noinit(Q_boundary_local);

   Cell<strict_fp_t> Q_cell_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("Q_cell_local");
   GDF::transfer_to_gpu_noinit(Q_cell_local);

   VectorRead<int> number_of_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("number_of_neighbors_local");
   FaceRead<strict_fp_t> area_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::FACE>("area_local");
   GDF::transfer_to_gpu_readonly(number_of_neighbors_local, area_local);

   BoundaryRead<int> boundary_face_to_cell_local = m_silo.retrieve_entry<int, CDF::StorageType::BOUNDARY>("boundary_face_to_cell_local");
   BoundaryRead<strict_fp_t> boundary_area_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_area_local");
   GDF::transfer_to_gpu_readonly(boundary_face_to_cell_local, boundary_area_local);

   CellRead<strict_fp_t> volume_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("volume_local");
   GDF::transfer_to_gpu_readonly(volume_local);

   FaceRead<int> cell_neighbors_local = m_silo.retrieve_entry<int, CDF::StorageType::FACE>("cell_neighbors_local");
   VectorRead<strict_fp_t> xcen_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("xcen_local");
   VectorRead<strict_fp_t> normal_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("normal_local");
   Face<strict_fp_t> rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::FACE>("rdista_local");
   GDF::transfer_to_gpu_readonly(cell_neighbors_local, xcen_local, normal_local);
   GDF::transfer_to_gpu_move(rdista_local);

   VectorRead<strict_fp_t> boundary_xcen_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_xcen_local");
   VectorRead<strict_fp_t> boundary_normal_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("boundary_normal_local");
   Boundary<strict_fp_t> boundary_rdista_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::BOUNDARY>("boundary_rdista_local");
   GDF::transfer_to_gpu_readonly(boundary_xcen_local, boundary_normal_local);
   GDF::transfer_to_gpu_move(boundary_rdista_local);

   Vector<strict_fp_t> A_data_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::VECTOR>("A_data_local");
   CellRead<int> csr_diag_idx_local = m_silo.retrieve_entry<int, CDF::StorageType::CELL>("csr_diag_idx_local");
   GDF::transfer_to_gpu_noinit(A_data_local);
   GDF::transfer_to_gpu_readonly(csr_diag_idx_local);

   FaceRead<int> csr_idx_local = m_silo.retrieve_entry<int, CDF::StorageType::FACE>("csr_idx_local");
   GDF::transfer_to_gpu_readonly(csr_idx_local);

   Cell<strict_fp_t> residual_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("residual_local");
   GDF::transfer_to_gpu_noinit(residual_local);

   Cell<strict_fp_t> rhs_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("rhs_local");
   GDF::transfer_to_gpu_noinit(rhs_local);

   VectorRead<int> ia_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ia_local");
   VectorRead<int> ja_local = m_silo.retrieve_entry<int, CDF::StorageType::VECTOR>("ja_local");
   Cell<strict_fp_t> dQ_old_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_old_local");
   Cell<strict_fp_t> dQ_local = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("dQ_local");
   GDF::transfer_to_gpu_readonly(ia_local, ja_local);
   GDF::transfer_to_gpu_noinit(dQ_old_local, dQ_local);
}
