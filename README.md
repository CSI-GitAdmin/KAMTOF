# KAMTOF : Kernel-level Access-based Memory Transfer Optimization Framework

KAMTOF is a SYCL based GPU data framework for rapid incremental GPU porting of a large-scale CFD CPU code

## Building KAMTOF

KAMTOF requires the following tools:
- Intel oneAPI toolkit version 2025.1+
- oneMath v0.7 (Built with support for BLAS and SPARSE libraries of all 3 vendors)
- CUDA toolkit v12.2+
- ROCm toolkit v6.2+
- HPCX MPI Library 2.21+
- Intel OCLOC

## Environment Setup script

KAMTOF provides an environment [setup script](scripts/setup_env.sh) in the scripts folder to setup the environment needed for building. The setup script requires the following environment variables to be setup:
- ONEAPI_ROOT
- CUDA_ROOT
- ROCM_ROOT
- ONEMATH_ROOT
- HPCX_ROOT


### CMake options

These are the CMake options provided by KAMTOF. Please preface all these options with -D flag when using in the CMake command.

- CMAKE_BUILD_TYPE : Build type option
  - Options: RELEASE/DEBUG/MAP/ASAN
  - Default: RELEASE 

- ENABLE_GPU : To turn ON the GPU solver
  - Options: ON/OFF
  - Default: OFF
 
- GPU_DEVELOP : Turns ON GPU development mode which enables automatic data transfers to the CPU. This negatively impacts the performance of the CPU data access operators. 
  - Options: ON/OFF
  - Default: ON
 
- BUILD_XDF_TESTS :  Flag to turn ON the unit-test builds for the CPU Data Framework (CDF) and GPU Data Framework (GDF). Note that currently CDF unit tests are not built when ENABLE_GPU is ON
  - Options: ON/OFF
  - Default: ON

- HW : Specifies the target hardware for AOT compilation
  - Options: Any combination of the following options can be provided as a list enclosed by single qoutes and delimited by semi-colon ('<target_option_1>;<target_option_2>;<target_option_3>') 
    - INTGPU : Compile for Intel GPUs (Currently only supports PVC cards)
    - NVDGPU : Compile for NVIDIA GPUs
    - AMDGPU : Compile for AMD GPUs
    - SYCLCPU: Compile for x86 CPUs
    - ALL: Compile for all of the above
  - Default: ALL
  - NOTE: SYCLCPU is always added to the HW list
 
- SM : Specify the architecture of NVIDIA GPUs to target
  - Options: 61/70/75/80/86 etc.
  - Default: 70
 
- GFX : Specify the architecture of AMD GPUs to target
  - Options: 942a/1100/1101 etc.
  - Default: 1100
 
Here is an example of a CMake command to build for Intel and NVIDIA GPUs (arch 86) without the GPU_DEVELOP option in debug mode:

```
mkdir build_release
cd build_release
cmake -DCMAKE_BUILD_TYPE=DEBUG -DENABLE_GPU=ON -DHW='INTGPU;NVDGPU' ../  
```

## Running KAMTOF

KAMTOF contains 3 different executables: 
- `cpu_framework/cpu_framework_test` : Test certain basic functionalities of the CPU framework (Will not be built is ENABLE_GPU is ON)
- `gpu_framework/gpu_framework_test` : Test certain basic functionalities of the GPU framework
- `solver` : The actual Laplace Solver
  - The input files, [laplace.in](input_files/laplace.in), for the solver are placed in the [input_files](input_files) folder
