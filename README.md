# KAMTOF : Kernel-level Access-based Memory Transfer Optimization Framework

KAMTOF is a SYCL based GPU data framework for rapid incremental GPU porting of a large-scale CFD CPU code

## Building KAMTOF

KAMTOF requires the following tools:
- Intel oneAPI toolkit version 2025.0+
- oneMath
- CUDA toolkit (If compiling for NVIDIA GPUs)
- ROCm toolkit (If compiling for AMD GPUs)
- Intel OCLOC (If compiling for INTEL GPUs) 

### CMake options

These are the CMake options provided by KAMTOF. Please preface all these options with -D flag when using in the CMake command.

- CMAKE_BUILD_TYPE : Build type option
  - Options: RELEASE/DEBUG/MAP/ASAN
  - Default: RELEASE 

- ENABLE_GPU : To turn ON the GPU solver
  - Options: ON/OFF
  - Default: OFF
 
- GPU_DEVELOP : Turns ON GPU development mode in which we incur the cost of branching inside the CPU operators for achieving fully automatic data transfer
  - Options: ON/OFF
  - Default: ON
 
- BUILD_XDF_TESTS :  Flag to turn ON the unit tests builds for the CPU Data Framework (CDF) and GPU Data Framework (GDF). Note that currently CDF unit tests are not built when ENABLE_GPU is ON
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
 
- SM : Specify the architecture of NVIDIA GPUs to target
  - Options: 61/70/75/80/86 etc.
  - Default: 70
 
- GFX : Specify the architecture of AMD GPUs to target
  - Options: 942a/1100/1101 etc.
  - Default: 1100
 
NOTE: SYCLCPU is turned added to HW list by default irrespective of the HW chosen 
 
Here is an example of a CMake command to build for Intel and NVIDIA GPUs (arch 86) without the GPU_DEVELOP option in debug mode:

```
mkdir build_release
cd build_release
cmake -DCMAKE_BUILD_TYPE=DEBUG -DENABLE_GPU=ON -DGPU_DEVELOP=OFF -DHW='INTGPU;NVDGPU' ../  
```

## Running KAMTOF

KAMTOF contains 3 different executables including unit tests and the Laplace solver 
- `cpu_framework/cpu_framework_test` : Test certain basic functionalities of the CPU framework (Will not be built is ENABLE_GPU is ON)
- `gpu_framework/gpu_framework_test` : Test certain basic functionalities of the GPU framework
- `solver` : The actual LaPlace Solver
  - The solver inputs are all command line inputs: `./solver <Nx> <Ny> <num_time_steps> <gpu_solver_bool>`
