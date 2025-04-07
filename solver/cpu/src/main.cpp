#include "grid.h"
#include "solver.h"

#include <cstdlib>
#include <sys/time.h>

#include "silo.h"

#ifdef ENABLE_GPU
#include "gpu_globals.h"
#include "solver_gpu.h"
#endif


int main (int argc, char** argv)
{
   mpi_init(&argc, &argv);

   if(rank == 0) // FIXME: Remove once the code is MPI
   {
   const bool gpu_solver = (argc == 5) ? atoi(argv[4]) : false;

   // Assume 2D 1m by 1m box
   const int Nx = (argc == 5) ? atoi(argv[1]) : 100;
   const int Ny = (argc == 5) ? atoi(argv[2]) : 100;
   
   const bool implicit = true;

   const int solver_type = 0; // 0: jacobi , 1: bicgstab
   const double num_iter = 10; // number of inner iterations

   const int number_of_time_steps = (argc == 5) ? atoi(argv[3]) : 1000;
   std::vector<strict_fp_t> residual_norm(number_of_time_steps + 1);

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

      for (int i = 0; i < number_of_time_steps; i++)
      {
         solver_ptr->compute_system();
         residual_norm[i] = solver_ptr->print_residual_norm(i);
         solver_ptr->update_solution(solver_type, num_iter);
      }
      residual_norm[number_of_time_steps] = solver_ptr->print_residual_norm(number_of_time_steps);

      gettimeofday(&e_time, NULL);

      const strict_fp_t time_elapsed_cpu = (e_time.tv_sec - s_time.tv_sec) +
                                      ((e_time.tv_usec - s_time.tv_usec) * 1e-6);

      printf("CPU time to solve %f\n", time_elapsed_cpu);

      solver_ptr->write_solution("laplace_solution.txt");
      solver_ptr->write_residual("laplace_residual.txt", residual_norm);

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
      
      for (int i = 0; i < number_of_time_steps; i++)
      {
         solver_ptr_gpu->compute_system();
         residual_norm[i] = solver_ptr_gpu->print_residual_norm(i);
         solver_ptr_gpu->update_solution();
      }
      residual_norm[number_of_time_steps] = solver_ptr_gpu->print_residual_norm(number_of_time_steps);

      gettimeofday(&e_time, NULL);

      const strict_fp_t time_elapsed_cpu = (e_time.tv_sec - s_time.tv_sec) +
                                    ((e_time.tv_usec - s_time.tv_usec) * 1e-6);
      
      printf("CPU time to solve %f\n", time_elapsed_cpu);

      solver_ptr_gpu->write_solution("laplace_solution.txt");
      solver_ptr_gpu->write_residual("laplace_residual.txt", residual_norm);

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

