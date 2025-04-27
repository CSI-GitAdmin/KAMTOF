#!/bin/bash

echo "Setting up environment for KAMTOF..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KAMTOF_ROOT="$(dirname "${SCRIPT_DIR}")"

# Helper function to check required env variables
function require_env_var {
    local var_name="$1"
    if [ -z "${!var_name}" ]; then
        echo "Error: Environment variable '$var_name' is not set."
        return 1
    fi
}

# Check all required environment variables
require_env_var HPCX_ROOT
require_env_var CUDA_ROOT
require_env_var ROCM_ROOT
require_env_var ONEAPI_ROOT
require_env_var ONEMATH_ROOT

source ${KAMTOF_ROOT}/scripts/env/oneAPI.sh
source ${KAMTOF_ROOT}/scripts/env/hpcx.sh
source ${KAMTOF_ROOT}/scripts/env/cuda.sh
source ${KAMTOF_ROOT}/scripts/env/rocm.sh
source ${KAMTOF_ROOT}/scripts/env/oneMath.sh

export CC=icx
export CXX=icpx

echo "Environment setup complete for KAMTOF."