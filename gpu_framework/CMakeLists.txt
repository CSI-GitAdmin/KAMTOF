cmake_minimum_required( VERSION 3.20 )

project( "GPU Framework" )

set( GPU_FRAMEWORK_DIR  $ENV{GPU_FRAMEWORK_DIR} )

aux_source_directory( ${GPU_FRAMEWORK_DIR}/src GpuFrameworkSrcTop)

add_library( gpu_framework ${GpuFrameworkSrcTop} )

target_include_directories( gpu_framework PUBLIC
   ${GPU_FRAMEWORK_DIR}/include
   ${GPU_FRAMEWORK_DIR}/include/utils
   ${GPU_FRAMEWORK_DIR}/include/detail
   $ENV{CPU_FRAMEWORK_DIR}/include
   $ENV{CPU_FRAMEWORK_DIR}/include/detail
)

find_package(oneMath REQUIRED)
target_link_libraries(gpu_framework ONEMATH::onemath)

###### Add target and dependencies for unit tests ##############
set( BUILD_TEST $ENV{BUILD_XDF_TESTS})
if (BUILD_TEST)

   set( GPU_FRAMEWORK_TEST_DIR ${GPU_FRAMEWORK_DIR}/test  )
   aux_source_directory( ${GPU_FRAMEWORK_TEST_DIR}/src GpuFrameworkTestSrcTop)

   add_executable( gpu_framework_test ${GpuFrameworkTestSrcTop} )

   target_include_directories( gpu_framework_test PUBLIC
      ${GPU_FRAMEWORK_DIR}/test/include)

   target_link_libraries( gpu_framework_test
      cpu_framework
      gpu_framework
   )

endif()
