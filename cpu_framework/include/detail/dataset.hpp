#ifndef DATASET_HPP
#define DATASET_HPP

#include "dataset.h"
#include "extractor.hpp"

template <class T, uint8_t DIMS >
dataSet<T, DIMS>::dataSet(const std::string& name, const uint64_t m_num_entries, const CDF::StorageType storage_type, const uint8_t * const shape,
                          const bool is_unresolved_entry, const bool allocate_mem):
   dataSetBase(name, m_num_entries, storage_type, CDF::extractor<T>::PODType(), DIMS, shape, is_unresolved_entry, allocate_mem)
{}

#endif // DATASET_HPP