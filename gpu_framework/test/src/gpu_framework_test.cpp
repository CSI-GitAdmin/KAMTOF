#include "silo.h" // For all SILO functionalites
#include "test_funcs.h" // For test_functions
#include "gpu_api_functions.h" // For GPU API functions
#include "mpi_utils.h"
#include "pagefault_handler.h"

int main (int argc, char** argv)
{
   mpi_init(&argc, &argv);

   setup_pagefault_handler();

   setup_gpu_globals_test();

   setup_cdf_vars_for_gpu_framework_tests();

   if(argc == 2)
   {
      switch (atoi(argv[1]))
      {
         case 1:
         {
            porting_stage_scenario();
            break;
         }
         case 2:
         {
            demonstrate_temp_write_async();
            break;
         }
         case 3:
         {
            demonstrate_temp_write();
            break;
         }
         case 4:
         {
            backend_testing();
            break;
         }
         default:
         {
            log_error("Incorrect CLI passed for gpu_framework_test. Valid options are 1 -> porting_stage_scenario, 2 -> demonstrate_temp_write_async, "
                      "3 -> demonstrate_temp_write, 4 -> backend_testing ");
         }
      }
   }
   else
   {
      backend_testing();
   }

   m_silo.clear_entries();

   finalize_gpu_globals_test();

   mpi_finalize();

   return EXIT_SUCCESS;
}