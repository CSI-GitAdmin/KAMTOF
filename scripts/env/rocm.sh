#!/bin/bash

# Check if ROCM_PATH is set
if [[ -z "${ROCM_ROOT}" ]]; then
    echo "Error: ROCM_ROOT is not set. Please set ROCM_PATH before sourcing this script."
    return 1
fi

# Check for conflict (optional warning)
echo "Using ROCM_ROOT=${ROCM_ROOT}"

# Export environment variables
export HIP_PATH=${ROCM_ROOT}

# Prepend/append paths
export PATH=${ROCM_ROOT}/llvm/bin:${PATH}
export MANPATH=${ROCM_ROOT}/llvm/share/man1:${MANPATH}
export CMAKE_PREFIX_PATH=${ROCM_ROOT}/lib/cmake:${CMAKE_PREFIX_PATH}
