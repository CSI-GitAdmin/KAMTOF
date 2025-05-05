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


int main (int argc, char** argv)
{
   mpi_init(&argc, &argv);

#ifdef ENABLE_GPU
   setup_pagefault_handler();
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
   tol_type              = input_data_ptr->tol_type;         // 0: absolute , 1: relative
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

      int count = 1;
      while(solver_ptr->get_residual_norm() > tol)
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
      solver_ptr_gpu->set_boundary_conditions(0, 0, 1e2, 0);
      solver_ptr_gpu->initialize_solution(grid.num_solved, 1.0);
      solver_ptr_gpu->compute_time_step(grid.num_solved, grid.num_attached);
      solver_ptr_gpu->compute_rdist(grid.num_solved, grid.num_attached);
      solver_ptr_gpu->compute_system(grid.num_solved, grid.num_attached);

      struct timeval s_time, e_time;
      gettimeofday(&s_time, NULL);

      solver_ptr_gpu->compute_residual(grid.num_solved, grid.num_attached);
      residual_norm.push_back(solver_ptr_gpu->print_residual_norm(0));
      
      int count = 1;
      while(solver_ptr_gpu->get_residual_norm() > tol)
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
