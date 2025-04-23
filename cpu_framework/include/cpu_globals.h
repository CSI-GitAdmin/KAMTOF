#ifndef CPU_GLOBALS_H
#define CPU_GLOBALS_H

#ifndef CDF_GLOBAL
   #define CDF_GLOBAL extern
#endif

CDF_GLOBAL int rank, numprocs;
CDF_GLOBAL int local_rank, local_numprocs;

#ifdef ENABLE_GPU
// Global variable to tell if the pagefault call is triggered by the const (or) the non const operator of DSB
#include <cstdint>
CDF_GLOBAL uint8_t in_const_operator;
#endif

#endif // CPU_GLOBALS_H