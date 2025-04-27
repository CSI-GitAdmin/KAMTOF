#!/bin/bash

# Check if ONEAPI_ROOT is set
if [[ -z "${ONEAPI_ROOT}" ]]; then
    echo "Error: ONEAPI_ROOT is not set. Please set ONEAPI_ROOT before sourcing this script."
    return 1
fi

# Define the list of scripts you want to source
required_scripts=(
    "${ONEAPI_ROOT}/compiler/latest/env/vars.sh"
    "${ONEAPI_ROOT}/tbb/latest/env/vars.sh"
    "${ONEAPI_ROOT}/umf/latest/env/vars.sh"
    "${ONEAPI_ROOT}/mkl/latest/env/vars.sh"
)

# Check that each script exists before sourcing
for script in "${required_scripts[@]}"; do
    if [[ ! -f "${script}" ]]; then
        echo "Error: Required environment script not found: ${script}"
        return 1
    fi
done

# Now source them
echo "Using ONEAPI_ROOT=${ONEAPI_ROOT}"

source "${ONEAPI_ROOT}/compiler/latest/env/vars.sh" --include-intel-llvm
source "${ONEAPI_ROOT}/tbb/latest/env/vars.sh"
source "${ONEAPI_ROOT}/umf/latest/env/vars.sh"
source "${ONEAPI_ROOT}/mkl/latest/env/vars.sh"