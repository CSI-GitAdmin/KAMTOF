#!/bin/bash

# Check if HPCX_ROOT is set
if [[ -z "${HPCX_ROOT}" ]]; then
    echo "Error: HPCX_ROOT is not set. Please set HPCX_ROOT before sourcing this script."
    return 1
fi

echo "Using HPCX_ROOT=${HPCX_ROOT}"

# Set basic MPI environment variables
export MPI_TYPE=HPCX

export HPCX_HOME=${HPCX_ROOT}

export HPCX_UCX_DIR=${HPCX_ROOT}/ucx
export HPCX_UCC_DIR=${HPCX_ROOT}/ucc
export HPCX_SHARP_DIR=${HPCX_ROOT}/sharp
export HPCX_HCOLL_DIR=${HPCX_ROOT}/hcoll
export HPCX_NCCL_RDMA_SHARP_PLUGIN_DIR=${HPCX_ROOT}/nccl_rdma_sharp_plugin

export HPCX_CLUSTERKIT_DIR=${HPCX_ROOT}/clusterkit
export HPCX_MPI_DIR=${HPCX_ROOT}/ompi
export HPCX_OSHMEM_DIR=${HPCX_MPI_DIR}
export HPCX_MPI_TESTS_DIR=${HPCX_MPI_DIR}/tests
export HPCX_OSU_DIR=${HPCX_MPI_DIR}/tests/osu-micro-benchmarks
export HPCX_OSU_CUDA_DIR=${HPCX_MPI_DIR}/tests/osu-micro-benchmarks-cuda

# MPI Runtime Environment Variables
export OPAL_PREFIX=${HPCX_MPI_DIR}
export PMIX_INSTALL_PREFIX=${HPCX_MPI_DIR}
export OMPI_HOME=${HPCX_MPI_DIR}
export MPI_HOME=${HPCX_MPI_DIR}
export OSHMEM_HOME=${HPCX_MPI_DIR}
export SHMEM_HOME=${HPCX_MPI_DIR}

# Compiler wrappers
export CC=mpicc
export CXX=mpicxx

# Prepend to PATH
export PATH=${HPCX_ROOT}/ucx/bin:${PATH}
export PATH=${HPCX_ROOT}/ucc/bin:${PATH}
export PATH=${HPCX_ROOT}/hcoll/bin:${PATH}
export PATH=${HPCX_ROOT}/sharp/bin:${PATH}
export PATH=${HPCX_MPI_DIR}/tests/imb:${PATH}
export PATH=${HPCX_ROOT}/clusterkit/bin:${PATH}
export PATH=${HPCX_MPI_DIR}/bin:${PATH}

# Prepend to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${HPCX_ROOT}/ucx/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${HPCX_ROOT}/ucx/lib/ucx:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${HPCX_ROOT}/ucc/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${HPCX_ROOT}/ucc/lib/ucc:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${HPCX_ROOT}/hcoll/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${HPCX_ROOT}/sharp/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${HPCX_ROOT}/nccl_rdma_sharp_plugin/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${HPCX_MPI_DIR}/lib:${LD_LIBRARY_PATH}

# Prepend to LIBRARY_PATH
export LIBRARY_PATH=${HPCX_ROOT}/ucx/lib:${LIBRARY_PATH}
export LIBRARY_PATH=${HPCX_ROOT}/ucc/lib:${LIBRARY_PATH}
export LIBRARY_PATH=${HPCX_ROOT}/hcoll/lib:${LIBRARY_PATH}
export LIBRARY_PATH=${HPCX_ROOT}/sharp/lib:${LIBRARY_PATH}
export LIBRARY_PATH=${HPCX_ROOT}/nccl_rdma_sharp_plugin/lib:${LIBRARY_PATH}
export LIBRARY_PATH=${HPCX_MPI_DIR}/lib:${LIBRARY_PATH}

# Prepend to CPATH
export CPATH=${HPCX_ROOT}/hcoll/include:${CPATH}
export CPATH=${HPCX_ROOT}/sharp/include:${CPATH}
export CPATH=${HPCX_ROOT}/ucx/include:${CPATH}
export CPATH=${HPCX_ROOT}/ucc/include:${CPATH}
export CPATH=${HPCX_MPI_DIR}/include:${CPATH}

# Prepend to PKG_CONFIG_PATH
export PKG_CONFIG_PATH=${HPCX_ROOT}/hcoll/lib/pkgconfig:${PKG_CONFIG_PATH}
export PKG_CONFIG_PATH=${HPCX_ROOT}/sharp/lib/pkgconfig:${PKG_CONFIG_PATH}
export PKG_CONFIG_PATH=${HPCX_ROOT}/ucx/lib/pkgconfig:${PKG_CONFIG_PATH}
export PKG_CONFIG_PATH=${HPCX_ROOT}/ompi/lib/pkgconfig:${PKG_CONFIG_PATH}

# Prepend to MANPATH
export MANPATH=${HPCX_MPI_DIR}/share/man:${MANPATH}

