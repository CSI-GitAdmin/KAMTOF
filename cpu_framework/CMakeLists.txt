cmake_minimum_required( VERSION 3.20 )

project( "CPU Framework" )

set( CPU_FRAMEWORK_DIR  $ENV{CPU_FRAMEWORK_DIR} )

aux_source_directory( ${CPU_FRAMEWORK_DIR}/src CpuFrameworkSrcTop)

add_library( cpu_framework ${CpuFrameworkSrcTop} )

target_include_directories( cpu_framework PUBLIC
   ${CPU_FRAMEWORK_DIR}/include
   ${CPU_FRAMEWORK_DIR}/include/detail
)

###### Add target and dependencies for unit tests ##############
set( BUILD_TEST $ENV{BUILD_XDF_TESTS})
if (BUILD_TEST)

   set( CPU_FRAMEWORK_TEST_DIR ${CPU_FRAMEWORK_DIR}/test  )

   aux_source_directory( ${CPU_FRAMEWORK_TEST_DIR}/src cpuFrameworkTestSrcTop)

   add_executable( cpu_framework_test ${cpuFrameworkTestSrcTop} )

   if(ENABLE_GPU)
      set_target_properties(cpu_framework_test PROPERTIES EXCLUDE_FROM_ALL TRUE)
   endif()

   target_include_directories( cpu_framework_test PUBLIC
      ${CPU_FRAMEWORK_TEST_DIR}/include)

   target_link_libraries( cpu_framework_test PUBLIC
      cpu_framework MPI::MPI_CXX)

endif()