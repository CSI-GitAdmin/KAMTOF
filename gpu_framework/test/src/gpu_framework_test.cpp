#include "silo.h" // For all SILO functionalites
#include "test_funcs.h" // For test_functions
#include "gpu_api_functions.h" // For GPU API functions
#include "mpi_utils.h"

constexpr uint64_t dim = 2;

int main (int argc, char** argv)
{
   mpi_init(&argc, &argv);

#ifndef NDEBUG
   if(rank == 0)
   {
      log_msg("Running in debug mode (NDEBUG not defined)\n");
   }
#endif

   uint64_t cell_count = 256;
   m_silo.resize<CDF::StorageType::CELL>(cell_count);

   Cell<strict_fp_t> pressure = m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("Pressure");
   Cell<strict_fp_t> volume = m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("Volume");
   Cell<strict_fp_t> temperature = m_silo.register_entry<strict_fp_t, CDF::StorageType::CELL>("Temperature");
   for(int ii = 0; ii < pressure.size(); ii++)
   {
      assert(pressure.size() ==  volume.size() && pressure.size() == temperature.size());
      pressure[ii] = 101325.00;
      volume[ii] = 2e-02;
      temperature[ii] = 300.00;
   }

   setup_gpu_globals_test();

   static_assert(dim >= 1, " GPU ND Range dimension has to be 1 or greater");
   uint64_t glob_range[dim], locl_range[dim];
   // In SYCL ND-range, the last dimension is the contigous dimension
   for (int i = 0; i < dim-1; i++)
   {
      glob_range[i] = 1;
      locl_range[i] = 1;
   }
   glob_range[dim-1] = cell_count;
   locl_range[dim-1] = cell_count;

   GDF::set_gpu_global_local_range(glob_range, locl_range);

   if(rank == 0)
   {
      test_dss_gpu();
      // test_dss_gpu_resize();
      test_gpu_pointer_api_funcs();
      test_gpu_atomics();
      // test_silo_null();
   }

   mpi_barrier();

   test_ncpu_ngpu();

   m_silo.clear_entries();

   finalize_gpu_globals_test();

   mpi_finalize();

   return EXIT_SUCCESS;
}
