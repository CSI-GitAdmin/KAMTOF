#include "mpi_utils.h"
#include "cpu_globals.h"
#include "logger.hpp"

void mpi_init(int* argc_ptr, char*** argv_ptr)
{
   int already_init = 0;
   MPI_Initialized(&already_init);
   if(already_init)
      log_msg<CDF::LogLevel::ERROR>("Redundant call to MPI_Init!");
   else
      MPI_Init(argc_ptr, argv_ptr);

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

   MPI_Comm local_comm;
   MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
   MPI_Comm_rank(local_comm, &local_rank);
   MPI_Comm_size(local_comm, &local_numprocs);
   MPI_Comm_free(&local_comm); // Free the local communicator
}

void mpi_finalize()
{
   int already_finalized = 0;
   MPI_Finalized(&already_finalized);
   if(already_finalized)
      log_msg<CDF::LogLevel::ERROR>("Redundant call to MPI_Finalize!");
   else
      MPI_Finalize();
}

void mpi_abort()
{
   MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

void mpi_barrier()
{
   MPI_Barrier(MPI_COMM_WORLD);
}