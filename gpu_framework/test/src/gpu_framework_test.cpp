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

   porting_stage_scenario((argc == 2) && (atoi(argv[1]) == 1));

   demonstrate_temp_write((argc == 2) && (atoi(argv[1]) == 2));

   backend_testing((argc != 2) || ((argc == 2) && (atoi(argv[1]) == 3)));

   m_silo.clear_entries();

   finalize_gpu_globals_test();

   mpi_finalize();

   return EXIT_SUCCESS;
}