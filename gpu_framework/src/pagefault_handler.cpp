#include <sys/mman.h>
#include <string.h>

#include "logger.hpp"
#include "datasetbase.h"
#include "pagefault_handler.h"
#include "gpu_globals.h"

std::pair<void*,uint64_t> allocate_page_aligned_memory(const uint64_t byte_size)
{
   // For pagefault mechanism to work, the allocation size must be a multiple of system page size
   uint64_t allocation_size = system_page_size * ((byte_size/system_page_size) + ((byte_size % system_page_size) ? 1 : 0));

   /*
    * We allocate the data using the 'mmap' function which gives up page aligned memory of size allocation size.
    * For more details on the arguemnts: https://man7.org/linux/man-pages/man2/mmap.2.html
   */
   void* m_data = mmap(NULL, allocation_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

   memset(m_data, 0, allocation_size); // Default initialize everything to 0 to follow the new behaviour

   if (m_data == MAP_FAILED)
   {
      log_msg<CDF::LogLevel::ERROR>("Failed to allocate page aligned memory!");
   }
   assert(m_data);
   return std::make_pair(m_data, allocation_size);
}

void deallocate_page_aligned_memory(void* m_data, const uint64_t allocation_size)
{
   assert(m_data);
   assert(allocation_size % system_page_size == 0);
   if (munmap(m_data, allocation_size) == -1)
   {
      log_msg<CDF::LogLevel::ERROR>("Failed to deallocate page aligned memory!");
   }
}

/*
 * This is the function that
*/
void pagefault_handler(int sig, siginfo_t *info, void *context)
{
   void *fault_addr = info->si_addr;
   if(fault_addr)
   {
      // Pick the address which is greater than (or) equal to the fault address
      std::set<std::pair<void*, dataSetBase*>>::iterator it = dsb_addr_set.upper_bound(std::make_pair(fault_addr, nullptr));

      bool override_decrement = false;

      // If the fault occured at non zero index of the last element, the above search will return end() but we still need to verify if is in the address range of the last element
      if(it == dsb_addr_set.end())
      {
         std::set<std::pair<void*, dataSetBase*>>::iterator last_element = --dsb_addr_set.end();
         void* bounding_address = static_cast<void*>(static_cast<char*>(last_element->first) + last_element->second->get_allocation_size());
         if(fault_addr <= bounding_address)
         {
            it = last_element;
            override_decrement = true; // In this case fault_addr is not equal to the it->first but we since we have manually pointed it to the correct element, we don't need to decrement
         }
      }

      if(it != dsb_addr_set.end()) // If not found in the list, then it is general SEGFAULT which we do not handle and pass on to mpi_abort()
      {
         if((it->first != fault_addr) && !override_decrement) // If the fault address was exactly found in the set, then we do not need to decrement the iterator
            it--;
         dataSetBase* const dsb_entry = it->second;
         if(in_const_operator)
         {
            if(mprotect(dsb_entry->cpu_data(), dsb_entry->get_allocation_size(), PROT_READ) == -1) // Unlock the data for reading
            {
               log_msg<CDF::LogLevel::ERROR>(std::string("Failure while unlocking data in pagefault handler for variable: ") + dsb_entry->name());
            }
         }
         else
         {
            if(mprotect(dsb_entry->cpu_data(), dsb_entry->get_allocation_size(), PROT_READ | PROT_WRITE) == -1) // Unlock the data for reading and writing
            {
               log_msg<CDF::LogLevel::ERROR>(std::string("Failure while unlocking data in pagefault handler for variable: ") + dsb_entry->name());
            }
         }
         dsb_entry->transfer_to_cpu(in_const_operator); // Transfer the variable to CPU
         return;
      }
   }
   else // Non GPU segfaults
   {  
      char err_msg[200];
      sprintf(err_msg, "Caught SIGSEGV at address: %p", fault_addr);
      log_msg<CDF::LogLevel::ERROR>(err_msg);
   }
}

// Setup the the custom pagefault handler
void setup_pagefault_handler()
{
   system_page_size = sysconf(_SC_PAGESIZE); // Set the system page size

   struct sigaction sa;
   memset(&sa, 0, sizeof(sa)); // Zero out the struct completely

   sa.sa_sigaction = pagefault_handler; // Assign our custom segfault/pagefault handler

   sa.sa_flags = SA_SIGINFO;  // Use siginfo_t

   if (sigaction(SIGSEGV, &sa, NULL) == -1) // Register our sigaction struct for handling SIGSEGV signal
   {
      log_msg<CDF::LogLevel::ERROR>("Failure while registering custom pagefault handler!");
   }
}