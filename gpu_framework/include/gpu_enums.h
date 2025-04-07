#ifndef GPU_ENUMS_H
#define GPU_ENUMS_H

#include <cstdint> // For uint16_t

namespace  GDF
{

enum class sycl_device_vendor_t : uint8_t // Used to identify the SYCL device vendor
{
   AMD,
   INTEL,
   NVIDIA,
   OTHER
};

enum class sycl_device_t : uint8_t // Used to select the type of device to run the SYCL kernels
{
   CPU,
   GPU,
   ACCELERATOR,
   DEFAULT
};

enum class xpu_t : uint8_t // Used to reference the device (xpu) type for returning xpu_data_status_t
{
   CPU,
   GPU
};

#define SINGLE_WG_SIZE 1024

enum class transfer_mode_t : uint8_t
{
   MOVE, // Transfers the data ownership and syncs data values
   COPY, /* Syncs data values and provides the target device with TEMP_WRITE permissions
         *  while the current device still holds ownership */
   NOT_INITIALIZE, // Only transfers ownership and doesn't sync data values
   SYNC_AND_MOVE, // Syncs the cpu and gpu data status based on the most up_to_date value and then moves over the data to the device mentioned while retaining TEMP_WRITE permissions
   READ_ONLY, // Share the data ownership as read only (cannot write to data) and syncs data values if needed
   NOT_SET // The deafult value
};

enum class xpu_data_status_t : uint8_t
{
   UP_TO_DATE_WRITE, // Data is owned and has the most up to date values
   OUT_OF_DATE, // Data is not owned and doesn't have the most up to date values
   TEMP_WRITE, /* Doesn't own the data but can write to it which will be disregarded during transfer calls
                * (CPU writing post (with averaging/filtering calculations) while GPU performs the next iteration) */
   UP_TO_DATE_READ, /* Data is not owned but has the most up to date values.
                    *  The device having this status is not expected to write to the data except for CPU during resize */
   RESIZED_ON_CPU, // The data was resized on the CPU (e.g Mesh Ops)
   NOT_ALLOCATED //  Data is not yet allocated
};

#define GZ 0 // Index used for getting object from GPU device pointers. Mainly useful for scalars

} // GDF

#endif // GPU_ENUMS_H