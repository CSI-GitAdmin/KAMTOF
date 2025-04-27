#ifndef CPU_GLOBALS_H
#define CPU_GLOBALS_H

#ifndef CDF_GLOBAL
   #define CDF_GLOBAL extern
#endif

#include "fp_data_types.h"

CDF_GLOBAL int rank, numprocs;
CDF_GLOBAL int local_rank, local_numprocs;
CDF_GLOBAL bool gpu_solver;
CDF_GLOBAL bool implicit_solver;
CDF_GLOBAL strict_fp_t tol;
CDF_GLOBAL int tol_type;
CDF_GLOBAL int solver_type;
CDF_GLOBAL int num_iter;

#ifdef ENABLE_GPU
// Global variable to tell if the pagefault call is triggered by the const (or) the non const operator of DSB
CDF_GLOBAL bool in_const_operator;
#endif

#endif // CPU_GLOBALS_H