#ifndef SILO_UTILS_H
#define SILO_UTILS_H

#include <cstdint>
#include <string>

namespace CDF
{
enum class StorageType : uint8_t;
enum class PODType : uint8_t;
} // namespace CDF

std::string get_storage_type_name(const CDF::StorageType &TYPE);
uint32_t get_single_element_byte_size(const CDF::PODType& TYPE);

#endif // SILO_UTILS_H