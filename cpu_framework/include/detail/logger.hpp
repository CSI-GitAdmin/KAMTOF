#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include "cpu_framework_enums.h"
#include "mpi_utils.h"
#include "cpu_globals.h"

template <CDF::LogLevel LEVEL = CDF::LogLevel::PROGRESS>
void log_msg(std::string msg)
{
   switch (LEVEL)
   {
      case CDF::LogLevel::ERROR:
      {
         std::cout << " ERROR[" << rank << "] : " << msg << std::endl;
         mpi_abort();
         break;
      }

      case CDF::LogLevel::WARNING:
      {
         std::cout << " WARNING[" << rank << "] : " << msg << std::endl;
         break;
      }

      case CDF::LogLevel::PROGRESS:
      {
         std::cout << " [" << rank << "] : " << msg << std::endl;
         break;
      }

      case CDF::LogLevel::DEBUG:
      {
         std::cout << " DEBUG[" << rank << "] : " << msg << std::endl;
         break;
      }

      default:
      {
         std::cout << " ERROR[" << rank << "] : Invaild LogLevel provided! " << std::endl;
         mpi_abort();
      }
   }
}

#endif // LOGGER_HPP