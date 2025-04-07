#CMake for the Laplace solver

cmake_minimum_required( VERSION 3.20 )

set( CMAKE_CXX_STANDARD 23 )

project( "SOLVER using GPU Framework" )

find_package (MPI REQUIRED )

# Pass the Device flag to the solver source code
if (ENABLE_GPU)
   add_definitions(-DENABLE_GPU)
endif()

# Build in release mode by default
if( CMAKE_BUILD_TYPE STREQUAL "" )
   set( CMAKE_BUILD_TYPE Release )
endif()

# Build the XDF tests by default
if(NOT DEFINED BUILD_XDF_TESTS)
   set( ENV{BUILD_XDF_TESTS} ON)
   message( STATUS "Building XDF tests by default. To turn it OFF, run CMake with -DBUILD_XDF_TESTS=OFF" )
else()
   set( ENV{BUILD_XDF_TESTS} "${BUILD_XDF_TESTS}")
   if(BUILD_XDF_TESTS)
      message(STATUS "Building XDF tests. To turn it OFF, run CMake with -DBUILD_XDF_TESTS=OFF")
   else()
      message(STATUS "NOT Building XDF tests. To turn it ON, run CMake with -DBUILD_XDF_TESTS=ON")
   endif()
endif()

set( CMAKE_CXX_FLAGS_RELEASE " -O3 -DNDEBUG -Werror" )
set( CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fno-inline-functions -Wunused-result" )
set( CMAKE_CXX_FLAGS_ASAN    "-g -O3 -DNDEBUG -fsanitize=address -fno-omit-frame-pointer " )
set( CMAKE_CXX_FLAGS_MAP    "-g -O3 -DNDEBUG " )

if(ENABLE_GPU)

   if(NOT DEFINED GPU_DEVELOP OR GPU_DEVELOP)
      set(GPU_DEVELOP ON)
   endif()

   if(GPU_DEVELOP)
      add_definitions(-DGPU_DEVELOP)
      message(STATUS "Building with GPU_DEVELOP Mode ON. Please compile with -DGPU_DEVELOP=OFF to disable GPU development mode")
   else()
      message(STATUS "Building with GPU_DEVELOP Mode OFF. Please compile with -DGPU_DEVELOP=ON to disable GPU development mode")
   endif()

   # Options to know which devices we are doing AOT for
   option(SYCLALL "Target ALL GPU and CPU for GPU AOT" OFF)
   option(INTGPU "Target INTEL GPU for GPU AOT" OFF)
   option(AMDGPU "Target AMD GPU for GPU AOT" OFF)
   option(NVDGPU "Target NVIDIA GPU for GPU AOT" OFF)
   option(SYCLCPU "Target SYCL CPUs for GPU AOT" OFF)

   # Set the SYCL Compiler flags anf linker flags
   set(SYCL_LINK_FLAGS "-fsycl-max-parallel-link-jobs=12 ") # To link in parallel
   set(SYCL_COMPILE_FLAGS "-fsycl")

   # HW is set to ALL by default is no argument was provided
   if(NOT HW)
      set(HW "ALL")
      message(STATUS "Default HW set to ALL. Choose a subset from -DHW='<INTGPU;NVDGPU;AMDPGU;SYCLCPU;ALL>' to specify the required hardware")
      set(SYCLALL ON)
   else()
      string(TOUPPER "${HW}" HW)
      # Iterate through the list to find all the provided HW options
      foreach(CUR_HW IN LISTS HW)
         if(CUR_HW STREQUAL "ALL")
            if(${SYCLALL}) # Catch duplicates
               message(FATAL_ERROR "${CUR_HW} option repeated in the HW list ${HW}")
            endif()
            set(SYCLALL ON)
         elseif(CUR_HW STREQUAL "INTGPU")
            if(${INTGPU}) # Catch duplicates
               message(FATAL_ERROR "${CUR_HW} option repeated in the HW list ${HW}")
            endif()
            set(INTGPU ON)
         elseif(CUR_HW STREQUAL "AMDGPU")
            if(AMDGPU) # Catch duplicates
               message(FATAL_ERROR "${CUR_HW} option repeated in the HW list ${HW}")
            endif()
            set(AMDGPU ON)
         elseif(CUR_HW STREQUAL "NVDGPU")
            if(${NVDGPU}) # Catch duplicates
               message(FATAL_ERROR "${CUR_HW} option repeated in the HW list ${HW}")
            endif()
            set(NVDGPU ON)
         elseif(CUR_HW STREQUAL "SYCLCPU")
            if(${SYCLCPU}) # Catch duplicates
               message(FATAL_ERROR "${CUR_HW} option repeated in the HW list ${HW}")
            endif()
            set(SYCLCPU ON)
         else() # Unknown Hardware type provided
            message(WARNING "Unkown HW type provided for GPU AOT : ${CUR_HW}")
            message(FATAL_ERROR "USAGE: cmake -DCMAKE_BUILD_TYPE=RELEASE/DEBUG -DENABLE_GPU=ON -DHW=<INTGPU;NVDGPU;AMDPGU;SYCLCPU;ALL> -DSM=70/75/80/86 -DGFX=1100/942a/1030 <path_to_CMakeLists.txt_folder>")
         endif()
      endforeach()
   endif()

   # This is done to unset all the options for the case where ALL was provided not as the first argument for HW list
   # Eg. -DHW='<INTGPU;NVDGPU;AMDPGU;SYCLCPU;ALL>'. Here we would have to unset the NVDGPU and SYCLCPU options
   if(${SYCLALL})
      set(INTGPU ON)
      set(AMDGPU ON)
      set(NVDGPU ON)
      set(SYCLCPU ON)
   endif()

   # Check and set SM
   if(${NVDGPU}) # If NVIDIA GPUs are a part of the AOT target
      if(NOT SM) # Set default value of SM to 70 to handle V100 and higher GPUs
         set(SM "70")
         message(STATUS "NVIDIA SM set to ${SM} (Default). Run with -DSM=<xy> to specify the required compute capability version as per hardware")
      elseif(NOT SM MATCHES "^[0-9]+$") # Make sure that SM is just a number
         message(FATAL_ERROR "SM value must be a number eg. 70/75/80/86. Given value: ${SM}")
      endif()
   else() # If NVIDIA GPUs are NOT a part of the AOT target
      if(SM)
         message(FATAL_ERROR "SM provided but HW list does not contain NVDGPU/ALL")
      endif()
   endif()

   # Check and set GFX
   if(${AMDGPU}) # If AMD GPUs are a part of the AOT target
      if(NOT GFX) # Set default value of GFX to 70 to handle RX 7900 GRE GPUs
         set(GFX "1100")
         message(STATUS "AMD GFX set to ${GFX} (Default). Run with -DGFX=<xy> to specify the required compute capability version as per hardware")
      endif()
   else() # If AMD GPUs are NOT a part of the AOT target
      if(GFX)
         message(FATAL_ERROR "GFX provided but HW list does not contain AMDGPU/ALL")
      endif()
   endif()

   # Set the base flags to start with
   set(SYCL_AOT_TARGETS_BASE " -Wno-unused-command-line-argument -fsycl-targets=")
   set(SYCL_AOT_TARGETS "${SYCL_AOT_TARGETS_BASE}") # All AOT target flags will be added to this list
   set(SYCL_AOT_DEVICES " ") # All AOT device arch flags will be added to this list

   # INTEL GPU (xe-hpc) AOT flags
   set(SYCL_AOT_TARGET_INTEL "intel_gpu_pvc,")

   # AMD GPU (gfx${GFX}) AOT flags
   set(SYCL_AOT_TARGET_AMD "amd_gpu_gfx${GFX},")

   # NVIDIA GPU (sm_${SM}) AOT flags
   set(SYCL_AOT_TARGET_NVIDIA "nvidia_gpu_sm_${SM},")

   # SYCL CPU AOT flags. Does not have a device flag
   set(SYCL_AOT_TARGET_CPU "spir64")

   # SYCL SPECIFIC COMPILE NOTES #
   ###############################
   #
   # In the current method, CPU AOT flag spir64 will be always be added to the end as the
   # AOT target flag should not end with a comma (,). All the other flags should end with a comma (,)
   # to account for any other flags getting added after it
   #
   # As of now, Intel AOT commands "spir64_gen" should always first in -fsycl-targets. This seems to be
   # a bug in which if the compiler sees the target "spir64_gen", it passes the first -Xs option to the
   # OCLOC compiler
   #
   ###############################

   # Append all the needed flags to TARGETS and DEVICES
   if(${SYCLALL}) # If ALL, add all at once
      set(SYCL_AOT_TARGETS "${SYCL_AOT_TARGETS_BASE}${SYCL_AOT_TARGET_INTEL}${SYCL_AOT_TARGET_AMD}${SYCL_AOT_TARGET_NVIDIA}${SYCL_AOT_TARGET_CPU}")
      message(STATUS "SYCL AOT target set to ALL (Supported archs = AMD:gfx${GFX}, INTEL:xe-hpc, NVIDIA:sm_${SM}, CPU:spir64)")
   else() # If NOT ALL, add each vendor AOT flags as applicable
      if(${INTGPU})
         set(SYCL_AOT_TARGETS "${SYCL_AOT_TARGETS}${SYCL_AOT_TARGET_INTEL}")
         message(STATUS "INTEL GPU (arch:xe-hpc) added to AOT target")
      endif()
      if(${AMDGPU})
         set(SYCL_AOT_TARGETS "${SYCL_AOT_TARGETS}${SYCL_AOT_TARGET_AMD}")
         message(STATUS "AMD GPU (arch:gfx${GFX}) added to AOT target")
      endif()
      if(${NVDGPU})
         set(SYCL_AOT_TARGETS "${SYCL_AOT_TARGETS}${SYCL_AOT_TARGET_NVIDIA}")
         message(STATUS "NVIDIA GPU (arch:sm_${SM}) added to AOT target")
      endif()
   endif()

   # Add the CPU target in the end only is HW was not set to ALL (It was added to the end already if HW was set to ALL)
   if(NOT ${SYCLALL})
      set(SYCL_AOT_TARGETS "${SYCL_AOT_TARGETS}${SYCL_AOT_TARGET_CPU}")
      set(SYCLCPU ON)
      message(STATUS "SYCL CPU (arch:spir64) added to AOT target")
   endif()

   # Assemble the final SYCL AOT flags
   set(SYCL_AOT_FLAGS "${SYCL_AOT_TARGETS}")

   # Add all the SYCL flags to CMAKE CXX FLAGS
   set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE}  ${SYCL_COMPILE_FLAGS} ${SYCL_AOT_FLAGS}")
   set(CMAKE_CXX_FLAGS_DEBUG    "${CMAKE_CXX_FLAGS_DEBUG}    ${SYCL_COMPILE_FLAGS} ${SYCL_AOT_FLAGS}")
   set(CMAKE_CXX_FLAGS_ASAN     "${CMAKE_CXX_FLAGS_ASAN}     ${SYCL_COMPILE_FLAGS} ${SYCL_AOT_FLAGS}")
   set(CMAKE_CXX_FLAGS_MAP      "${CMAKE_CXX_FLAGS_MAP}      ${SYCL_COMPILE_FLAGS} ${SYCL_AOT_FLAGS}")
   set(CMAKE_EXE_LINKER_FLAGS   "${CMAKE_EXE_LINKER_FLAGS}   ${SYCL_LINK_FLAGS}")

   # Pass in the device types to the source code to use for device/vendor specific blocks
   if(${SYCLALL})
      add_definitions(-DSYCLALL)
   endif()
   if(${AMDGPU})
      add_definitions(-DAMDGPU)
   endif()
   if(${INTGPU})
      add_definitions(-DINTGPU)
   endif()
   if(${NVDGPU})
      add_definitions(-DNVDGPU)
   endif()
   if(${SYCLCPU})
      add_definitions(-DSYCLCPU)
   endif()

   if(GPU_MEM_LOG)
      add_definitions(-DGPU_MEM_LOG)
   endif()

   # Work around to avoid clang llvm link errors when using asserts in NVIDIA/AMD GPU
   if(${NVDGPU} OR ${AMDGPU})
      add_definitions(-DDISABLE_GPU_KERNEL_ASSERTS)
   endif()

endif()

set( ENV{PROJECT_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set( ENV{CPU_FRAMEWORK_DIR} $ENV{PROJECT_ROOT_DIR}/cpu_framework )
set( ENV{CPU_SOLVER_DIR} $ENV{PROJECT_ROOT_DIR}/solver/cpu )

if (ENABLE_GPU)
   set( ENV{GPU_FRAMEWORK_DIR} $ENV{PROJECT_ROOT_DIR}/gpu_framework )
   add_subdirectory( $ENV{GPU_FRAMEWORK_DIR} )
endif()

add_subdirectory( $ENV{CPU_FRAMEWORK_DIR} )

aux_source_directory( $ENV{CPU_SOLVER_DIR}/src CpuSolverSrcTop)

if(ENABLE_GPU)
   set( ENV{GPU_SOLVER_DIR} $ENV{PROJECT_ROOT_DIR}/solver/gpu )
   aux_source_directory( $ENV{GPU_SOLVER_DIR}/src GpuSolverSrcTop)
   add_executable(solver ${CpuSolverSrcTop} ${GpuSolverSrcTop})
else()
   add_executable(solver ${CpuSolverSrcTop})
endif()

target_include_directories( solver PUBLIC
   $ENV{CPU_SOLVER_DIR}/include
   $ENV{CPU_FRAMEWORK_DIR}/include
   $ENV{CPU_FRAMEWORK_DIR}/include/detail
)

if (ENABLE_GPU)
   target_include_directories(solver PUBLIC
      $ENV{GPU_SOLVER_DIR}/include
      $ENV{GPU_FRAMEWORK_DIR}/include
      $ENV{GPU_FRAMEWORK_DIR}/include/detail
      $ENV{GPU_FRAMEWORK_DIR}/include/util)
   target_link_libraries(solver cpu_framework gpu_framework)
else()
   target_link_libraries(solver cpu_framework)
endif()