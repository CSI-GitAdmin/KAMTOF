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

   if(rank == 0) // FIXME: Remove once the code is MPI
   {
      // default name of input file
      std::string infile = "laplace.in";
      
      // input file name
      if (argc > 2)
      {
         log_msg<CDF::LogLevel::WARNING>("Only one input line argument is needed. Additional input line arguments will be ignored.");
         infile = argv[1];
      }
      else if (argc < 2)
      {
         log_msg<CDF::LogLevel::WARNING>("Missing name of input file for Laplace solver. Using default value: 'laplace.in'.");
      }
      else
      {
         infile = argv[1];
      }


      // set input file name
      std::string input_filename = infile;
      
      // initialize parser object
      InputParser* input_data_ptr = new InputParser(input_filename);

      // read inputs and store input data into a struct
      input_data_ptr->read_inputs(input_filename);

      // print input options being used from the root rank
      if (rank == 0)
      {
         input_data_ptr->print_input_struct();
      }
      
      // create local variables to store input options
      const bool gpu_solver = input_data_ptr->use_gpu_solver;   // use_gpu_solver
      const int Nx          = input_data_ptr->Nx;               // number of grid points in x
      const int Ny          = input_data_ptr->Ny;               // number of grid points in y
      const bool implicit   = input_data_ptr->implicit_solver;  // use implicit solver
      const strict_fp_t tol = input_data_ptr->tol_val;          // tolerance at which solver will stop
      const int tol_type    = input_data_ptr->tol_type;         // 0: absolute , 1: relative
      const int solver_type = input_data_ptr->solver_type;      // 0: jacobi , 1: bicgstab
      const int num_iter    = input_data_ptr->num_iter;         // number of inner iterations

      std::vector<strict_fp_t> residual_norm;

      m_silo.resize<CDF::StorageType::CELL>(Nx*Ny);
      m_silo.resize<CDF::StorageType::FACE>(4*(Nx-2)*(Ny-2) + 3*(2*(Nx-2)+2*(Ny-2)) + 2*4);
      m_silo.resize<CDF::StorageType::BOUNDARY>(2*Nx + 2*Ny);

      Grid grid;
      grid.generate_grid(Nx, Ny);

      if(!gpu_solver)
      {
         Solver_base* solver_ptr = new Solver_base;
         solver_ptr->allocate_memory(implicit, grid);
         solver_ptr->copy_grid_data(grid);
         solver_ptr->set_boundary_conditions(0, 0, 0, 1);
         solver_ptr->initialize_solution(1.0);
         solver_ptr->compute_time_step();

         struct timeval s_time, e_time;
         gettimeofday(&s_time, NULL);

         int count = 0;
         while(solver_ptr->get_residual_norm() > tol)
         {
            solver_ptr->compute_system();
            residual_norm.push_back(solver_ptr->print_residual_norm(count));
            solver_ptr->update_solution(solver_type, num_iter);
            count++;
         }
         residual_norm.push_back(solver_ptr->print_residual_norm(count));

         gettimeofday(&e_time, NULL);

         const strict_fp_t time_elapsed_cpu = (e_time.tv_sec - s_time.tv_sec) +
                                       ((e_time.tv_usec - s_time.tv_usec) * 1e-6);

         printf("CPU time to solve %f\n", time_elapsed_cpu);

         std::string soln_file, res_file;
         soln_file = "laplace_solution_cpu_nx_" + std::to_string(Nx) + "_ny_" + std::to_string(Ny) + ".txt";
         res_file  = "laplace_residual_cpu_nx_" + std::to_string(Nx) + "_ny_" + std::to_string(Ny) + ".txt";

         solver_ptr->write_solution(soln_file);
         solver_ptr->write_residual(res_file, residual_norm);

         delete solver_ptr;
         solver_ptr = nullptr;
      }
#ifdef ENABLE_GPU
      else
      {
         setup_gpu_globals();
         Solver_base_gpu* solver_ptr_gpu = new Solver_base_gpu;
         solver_ptr_gpu->allocate_memory(implicit, grid);
         solver_ptr_gpu->copy_grid_data(grid);
         solver_ptr_gpu->set_boundary_conditions(0, 0, 0, 1);
         solver_ptr_gpu->initialize_solution(1.0);
         solver_ptr_gpu->compute_time_step();

         struct timeval s_time, e_time;
         gettimeofday(&s_time, NULL);
         
         int count = 0;
         while(solver_ptr_gpu->get_residual_norm() > tol)
         {
            solver_ptr_gpu->compute_system();
            residual_norm.push_back(solver_ptr_gpu->print_residual_norm(count));
            solver_ptr_gpu->update_solution(solver_type, num_iter);
            count++;
         }
         residual_norm.push_back(solver_ptr_gpu->print_residual_norm(count));

         gettimeofday(&e_time, NULL);

         const strict_fp_t time_elapsed_cpu = (e_time.tv_sec - s_time.tv_sec) +
                                       ((e_time.tv_usec - s_time.tv_usec) * 1e-6);
         
         printf("CPU time to solve %f\n", time_elapsed_cpu);

         std::string soln_file, res_file;
         soln_file = "laplace_solution_gpu_nx_" + std::to_string(Nx) + "_ny_" + std::to_string(Ny) + ".txt";
         res_file  = "laplace_residual_gpu_nx_" + std::to_string(Nx) + "_ny_" + std::to_string(Ny) + ".txt";

         solver_ptr_gpu->write_solution(soln_file);
         solver_ptr_gpu->write_residual(res_file, residual_norm);

         delete solver_ptr_gpu;
         solver_ptr_gpu = nullptr;
      }
#endif

      m_silo.clear_entries();

#ifdef ENABLE_GPU
      if(gpu_solver)
         finalize_gpu_globals();
#endif
   }

   mpi_finalize();
   
   return 0;
}
