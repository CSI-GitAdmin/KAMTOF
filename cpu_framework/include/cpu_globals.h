#ifndef CPU_GLOBALS_H
#define CPU_GLOBALS_H

#ifndef CDF_GLOBAL
   #define CDF_GLOBAL extern
#endif

CDF_GLOBAL int rank, numprocs;
CDF_GLOBAL int local_rank, local_numprocs;

#endif // CPU_GLOBALS_H