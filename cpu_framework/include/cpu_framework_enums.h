#ifndef CPU_FRAMEWORK_ENUMS_H
#define CPU_FRAMEWORK_ENUMS_H

#include <cstdint>

#define ZEROD 0

namespace CDF // CPU DATA FRAMEWORK
{

enum class StorageType : uint8_t
{
   CELL = 0,
   FACE,
   BOUNDARY,
   VECTOR,
   PARAMETER,
   NUM_STORAGE_TYPES
};

enum class PODType : uint8_t
{
   INT8 = 0,
   INT16,
   INT32,
   INT64,
   UINT8,
   UINT16,
   UINT32,
   UINT64,
   FP32,
   FP64,
   UNSUPPORTED_TYPE,
   POD_TYPE_COUNT
};

enum class LogLevel : uint8_t
{
   ERROR = 0,
   WARNING,
   PROGRESS,
   DEBUG,
   LOG_LEVEL_COUNT
};

} // namespace CDF

#endif // CPU_FRAMEWORK_ENUMS_H
