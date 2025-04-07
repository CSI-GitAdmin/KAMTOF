#ifndef FP_DATA_TYPES_H
#define FP_DATA_TYPES_H

// Define the primitive types
#define fp64_t double
#define fp32_t float

// Precision matters for these variables
#ifdef STRICT_FP_32
#define strict_fp_t fp32_t
#define MPI_STRICT_FP_T MPI_FLOAT
#else
#define strict_fp_t fp64_t
#define MPI_STRICT_FP_T MPI_DOUBLE
#endif

// Precision does not matter for these variables
#ifdef LOOSE_FP_64
#define loose_fp_t fp64_t
#define MPI_LOOSE_FP_T MPI_DOUBLE
#else
#define loose_fp_t fp32_t
#define MPI_LOOSE_FP_T MPI_FLOAT
#endif

#endif // FP_DATA_TYPES_H