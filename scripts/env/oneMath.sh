#!/bin/bash

# Check if ONEMATH_ROOT is set
if [[ -z "${ONEMATH_ROOT}" ]]; then
    echo "Error: ONEMATH_ROOT is not set. Please set ONEMATH_ROOT before sourcing this script."
    return 1
fi

# Check for conflict (optional warning)
echo "Using ONEMATH_ROOT=${ONEMATH_ROOT}"

# Prepend to CMAKE_PREFIX_PATH and LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${ONEMATH_ROOT}/lib/cmake:${CMAKE_PREFIX_PATH}
export LD_LIBRARY_PATH=${ONEMATH_ROOT}/lib:${LD_LIBRARY_PATH}
