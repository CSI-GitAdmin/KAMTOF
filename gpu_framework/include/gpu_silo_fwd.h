#ifndef GPU_SILO_FWD_H
#define GPU_SILO_FWD_H

#include "cpu_framework_enums.h"

// GPU counterpart of dataSetBase and and its GPURead version
class dataSetBaseGPU;

using dataSetBaseGPURead = const dataSetBaseGPU;

// GPU counterpart of dataSetStorage and and its GPURead version
template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
class dataSetStorageGPU;

template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
using dataSetStorageGPURead = const dataSetStorageGPU<T, TYPE, DIMS>;

// GPU counterpart of Cell, Attached, Face, UniqueFace and their GPURead versions
template <class T, uint8_t DIMS = ZEROD>
using CellGPU = dataSetStorageGPU<T, CDF::StorageType::CELL, DIMS>;

template <class T, uint8_t DIMS = ZEROD>
using CellGPURead = dataSetStorageGPURead<T, CDF::StorageType::CELL, DIMS>;

template <class T, uint8_t DIMS = ZEROD>
using BoundaryGPU = dataSetStorageGPU<T, CDF::StorageType::BOUNDARY, DIMS>;

template <class T, uint8_t DIMS = ZEROD>
using BoundaryGPURead = dataSetStorageGPURead<T, CDF::StorageType::BOUNDARY, DIMS>;

template <class T, uint8_t DIMS = ZEROD>
using FaceGPU = dataSetStorageGPU<T, CDF::StorageType::FACE, DIMS>;

template <class T, uint8_t DIMS = ZEROD>
using FaceGPURead = dataSetStorageGPURead<T, CDF::StorageType::FACE, DIMS>;

template <class T, uint8_t DIMS = ZEROD>
using ParameterGPU = dataSetStorageGPU<T, CDF::StorageType::PARAMETER, DIMS>;

template <class T, uint8_t DIMS = ZEROD>
using ParameterGPURead = dataSetStorageGPURead<T, CDF::StorageType::PARAMETER, DIMS>;

template <class T, uint8_t DIMS = ZEROD>
using VectorGPU = dataSetStorageGPU<T, CDF::StorageType::VECTOR, DIMS>;

template <class T, uint8_t DIMS = ZEROD>
using VectorGPURead = dataSetStorageGPURead<T, CDF::StorageType::VECTOR, DIMS>;

#endif // GPU_SILO_FWD_H