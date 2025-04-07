#include "gpu_enums.h" // To know possible values of the different enum classes in gpu_enums.h

namespace  GDF
{

// Function to convert transfer mode enum to C-style string
const char* transfer_mode_to_cstr(transfer_mode_t transfer_mode)
{
   switch (transfer_mode)
   {
      case transfer_mode_t::MOVE:
         return "MOVE";
      case transfer_mode_t::READ_ONLY:
         return "READ_ONLY";
      case transfer_mode_t::COPY:
         return "COPY";
      case transfer_mode_t::NOT_INITIALIZE:
         return "NOT_INITIALIZE";
      case transfer_mode_t::SYNC_AND_MOVE:
         return "SYNC_AND_MOVE";
      case transfer_mode_t::NOT_SET:
         return "NOT_SET";
      default:
         return "UNKNOWN";
   }
}

const char* xpu_data_status_to_cstr(xpu_data_status_t data_status)
{
   switch (data_status)
   {
      case xpu_data_status_t::NOT_ALLOCATED :
         return "NOT_ALLOCATED";
      case xpu_data_status_t::OUT_OF_DATE :
         return "OUT_OF_DATE";
      case xpu_data_status_t::RESIZED_ON_CPU :
         return "RESIZED_ON_CPU";
      case xpu_data_status_t::TEMP_WRITE :
         return "TEMP_WRITE";
      case xpu_data_status_t::UP_TO_DATE_READ :
         return "UP_TO_DATE_READ";
      case xpu_data_status_t::UP_TO_DATE_WRITE :
         return "UP_TO_DATE_WRITE";
      default:
         return "UNKNOWN";
   }
}

} // namespace GDF