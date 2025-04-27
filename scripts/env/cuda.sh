#!/bin/bash

# Define version and paths

# Check if CUDA_ROOT is set
if [[ -z "${CUDA_ROOT}" ]]; then
    echo "Error: CUDA_ROOT is not set. Please set CUDA_ROOT before sourcing this script."
    return 1
fi

echo "Using CUDA_ROOT=${CUDA_ROOT}"

# Export environment variables
export CUDA_INC=${CUDA_ROOT}/include
export CUDA_LIB=${CUDA_ROOT}/lib64
export CUDA_LIB_PATH=${CUDA_ROOT}/lib64/stubs

# Prepend to PATH and LD_LIBRARY_PATH
export PATH=${CUDA_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH}
