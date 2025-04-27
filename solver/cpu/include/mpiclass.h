/* 
 * Author: Mandhapati Raju <mandhapati.raju@convergecfd.com>
 *
 */

#ifndef MPICLASS_H
#define MPICLASS_H

#include <vector>
#include <array>
#include <mpi.h>
#include <algorithm>
#include <unordered_map>

#include "silo.h"
#include "silo_fwd.h"
#include "fp_data_types.h"

#ifdef ENABLE_GPU
#include "gpu_api_functions.h"
#endif

class MpiClass 
{
public:
   int ssize;
   int rsize;
   int* slist;
   int* rlist;
   int* scounts;
   int* sdisp;
   int* rcounts;
   int* rdisp;
   strict_fp_t* sbuf;
   strict_fp_t* rbuf;

   int* gpu_slist;
   int* gpu_rlist;
   strict_fp_t* gpu_sbuf;
   strict_fp_t* gpu_rbuf;

   ~MpiClass()
   {
      delete[] slist;
      delete[] rlist;
      delete[] scounts;
      delete[] rcounts;
      delete[] sdisp;
      delete[] rdisp;
      delete[] sbuf;
      delete[] rbuf;

#ifdef ENABLE_GPU
      if(gpu_solver)
      {
         GDF::free_gpu_var<int>(gpu_slist);
         GDF::free_gpu_var<int>(gpu_rlist);
         GDF::free_gpu_var<strict_fp_t>(gpu_sbuf);
         GDF::free_gpu_var<strict_fp_t>(gpu_rbuf);
      }
#endif
   }
};

CDF_GLOBAL MpiClass* mpiparams;

#endif /* MPICLASS_H */

