#ifndef GPU_HELPERS_H
#define GPU_HELPERS_H

#include <cstdint> // For uint8_t

namespace GDF
{

enum class transfer_mode_t : uint8_t;
enum class xpu_data_status_t : uint8_t;

// Function to convert transfer mode enum to C-style string
const char* transfer_mode_to_cstr(transfer_mode_t transfer_mode);
const char* xpu_data_status_to_cstr(xpu_data_status_t data_status);

} // namespace GDF

#endif // GPU_HELPERS_H