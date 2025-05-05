#include "test_funcs.h"
#include "gpu_api_functions.h"

static void set_P();
static void update_T();
static void update_V();

/*
 * This function depicts a typical scenario that could occur while a large code base is being ported to GPU
*/
void porting_stage_scenario()
{
   if(rank != 0)
      return;

   /* P = f(V,T)
    * 2 HtoD transfers
   */
   set_P();
   log_progress("Executed set_P on GPU");

   /* T = f(P,T)
    * 1 DtoH transfer
   */
   update_T();
   log_progress("Executed update_T on CPU");

   /* V = f(P,V,T)
    * 1 HtoD transfer
   */
   update_V();
   log_progress("Executed update_V on GPU");
}

/*
 * --------------------------------------------------------------------------------
 * Set_P()
 * --------------------------------------------------------------------------------
*/

class kg_set_P
{
public:
   kg_set_P(CellGPU<strict_fp_t> P, CellGPURead<strict_fp_t> V, CellGPURead<strict_fp_t> T):
      P_gpu(P),
      V_gpu(V),
      T_gpu(T)
   {}

   void operator() (sycl::nd_item<3> itm) const
   {
      size_t idx = GDF::get_1d_index(itm);
      size_t stride = GDF::get_1d_stride(itm);
      for(size_t kk = idx; kk < P_gpu.size(); kk += stride)
      {
         P_gpu[kk] = V_gpu[kk] * sycl::fabs(1.33 - T_gpu[kk]);
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      GDF::transfer_vars_to_gpu_impl<N>(P_gpu, V_gpu, T_gpu);
   }

private:
   mutable CellGPU<strict_fp_t> P_gpu;
   CellGPURead<strict_fp_t> V_gpu;
   CellGPURead<strict_fp_t> T_gpu;
};

static void set_P()
{
   Cell<strict_fp_t> P = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("variable_P");
   CellRead<strict_fp_t> V = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("variable_V");
   CellRead<strict_fp_t> T = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("variable_T");

   GDF::transfer_to_gpu_noinit(P); // The previous values of P are not required as it is completely overwrittern
   GDF::submit_to_gpu<kg_set_P>(P, V, T);
}


/*
 * --------------------------------------------------------------------------------
 * update_T()
 * --------------------------------------------------------------------------------
*/

static void update_T()
{
   CellRead<strict_fp_t> P = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("variable_P");
   Cell<strict_fp_t> T = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("variable_T");

   for(size_t kk = 0; kk < T.size(); kk++)
   {
      T[kk] = T[kk] + (1.58e-04 * P[kk]);
   }

}

/*
 * --------------------------------------------------------------------------------
 * update_T()
 * --------------------------------------------------------------------------------
*/

class kg_update_V
{
public:
   kg_update_V(CellGPURead<strict_fp_t> P, CellGPU<strict_fp_t> V, CellGPURead<strict_fp_t> T):
      P_gpu(P),
      V_gpu(V),
      T_gpu(T)
   {}

   void operator() (sycl::nd_item<3> itm) const
   {
      size_t idx = GDF::get_1d_index(itm);
      size_t stride = GDF::get_1d_stride(itm);
      for(size_t kk = idx; kk < V_gpu.size()/2; kk += stride)
      {
         V_gpu[kk] = P_gpu[kk]/(sycl::fabs(1.33 - T_gpu[kk]));
      }
   }

   template<uint8_t N>
   void transfer_vars_to_gpu()
   {
      GDF::transfer_vars_to_gpu_impl<N>(P_gpu, V_gpu, T_gpu);
   }

private:
   CellGPURead<strict_fp_t> P_gpu;
   mutable CellGPU<strict_fp_t> V_gpu;
   CellGPURead<strict_fp_t> T_gpu;
};

static void update_V()
{
   CellRead<strict_fp_t> P = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("variable_P");
   Cell<strict_fp_t> V = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("variable_V");
   CellRead<strict_fp_t> T = m_silo.retrieve_entry<strict_fp_t, CDF::StorageType::CELL>("variable_T");

   GDF::submit_to_gpu<kg_update_V>(P, V, T); // Only edit the first half of the variable V and hence cannot do 'noinit' transfer optimization
}
