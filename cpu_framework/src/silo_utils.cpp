#include "silo_utils.h"
#include "cpu_framework_enums.h"
#include "logger.hpp"

std::string get_storage_type_name(const CDF::StorageType &TYPE)
{
   switch (TYPE)
   {
      case CDF::StorageType::CELL:
      {
         return "CELL";
         break;
      }
      case CDF::StorageType::FACE:
      {
         return "FACE";
         break;
      }
      case CDF::StorageType::BOUNDARY:
      {
         return "BOUNDARY";
         break;
      }
      case CDF::StorageType::VECTOR:
      {
         return "VECTOR";
         break;
      }
      case CDF::StorageType::PARAMETER:
      {
         return "PARAMETER";
         break;
      }
      default:
      {
         log_msg<CDF::LogLevel::ERROR>("Invaild Storagetype provided!");
         return "UNKNOWN_TYPE";
      }
   }
}

uint32_t get_single_element_byte_size(const CDF::PODType &TYPE)
{
   switch (TYPE)
   {
      case CDF::PODType::UINT8:
      {
         return 1;
         break;
      }
      case CDF::PODType::UINT16:
      {
         return 2;
         break;
      }
      case CDF::PODType::UINT32:
      {
         return 4;
         break;
      }
      case CDF::PODType::UINT64:
      {
         return 8;
         break;
      }

      case CDF::PODType::INT8:
      {
         return 1;
         break;
      }
      case CDF::PODType::INT16:
      {
         return 2;
         break;
      }
      case CDF::PODType::INT32:
      {
         return 4;
         break;
      }
      case CDF::PODType::INT64:
      {
         return 8;
         break;
      }

      case CDF::PODType::FP32:
      {
         return 4;
         break;
      }
      case CDF::PODType::FP64:
      {
         return 8;
         break;
      }

      default:
      {
         log_msg<CDF::LogLevel::ERROR>("Invaild PODType provided!");
         return 0;
      }
   }
}