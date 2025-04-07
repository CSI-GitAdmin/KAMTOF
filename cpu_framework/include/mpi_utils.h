#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include "mpi.h"

void mpi_init(int* argc_ptr, char*** argv_ptr);

void mpi_finalize();

void mpi_abort();

void mpi_barrier();

#endif // MPI_UTILS_H