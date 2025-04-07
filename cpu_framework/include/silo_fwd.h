#ifndef SILO_FWD_H
#define SILO_FWD_H

#include <cstdint>
#include "cpu_framework_enums.h"

class dataSetBase;

template<class T, uint8_t DIMS = ZEROD>
class dataSet;

template<class T, uint8_t DIMS = ZEROD>
using dataSetRead =  dataSet<T, DIMS>;

template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
class dataSetStorage;

template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
using dataSetStorageRead = const dataSetStorage<T, TYPE, DIMS>;

template <class T, uint8_t DIMS = ZEROD>
using Cell = dataSetStorage<T, CDF::StorageType::CELL, DIMS>&;

template <class T, uint8_t DIMS = ZEROD>
using CellRead = dataSetStorageRead<T, CDF::StorageType::CELL, DIMS>&;

template <class T, uint8_t DIMS = ZEROD>
using Face = dataSetStorage<T, CDF::StorageType::FACE, DIMS>&;

template <class T, uint8_t DIMS = ZEROD>
using FaceRead = dataSetStorageRead<T, CDF::StorageType::FACE, DIMS>&;

template <class T, uint8_t DIMS = ZEROD>
using Boundary = dataSetStorage<T, CDF::StorageType::BOUNDARY, DIMS>&;

template <class T, uint8_t DIMS = ZEROD>
using BoundaryRead = dataSetStorageRead<T, CDF::StorageType::BOUNDARY, DIMS>&;

template <class T, uint8_t DIMS = ZEROD>
using Vector = dataSetStorage<T, CDF::StorageType::VECTOR, DIMS>&;

template <class T, uint8_t DIMS = ZEROD>
using VectorRead = dataSetStorageRead<T, CDF::StorageType::VECTOR, DIMS>&;

template <class T, uint8_t DIMS = ZEROD>
using Parameter = dataSetStorage<T, CDF::StorageType::PARAMETER, DIMS>&;

template <class T, uint8_t DIMS = ZEROD>
using ParameterRead = dataSetStorageRead<T, CDF::StorageType::PARAMETER, DIMS>&;

#endif // SILO_FWD_H