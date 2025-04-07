#ifndef DATASETSTORAGE_HPP
#define DATASETSTORAGE_HPP

#include "datasetstorage.h"
#include "extractor.hpp"

template <class T, CDF::StorageType TYPE, uint8_t DIMS>
dataSetStorage<T, TYPE, DIMS>::dataSetStorage(const std::string &name, const uint64_t m_num_entries, const uint8_t* const shape /* = nullptr*/, const bool allocate_mem /* = true */):
   dataSet<T, DIMS>(name, m_num_entries, TYPE, shape, allocate_mem)
{}

#endif // DATASETSTORAGE_HPP