#include <sys/mman.h>
#include <string.h>
#include <numeric>

#include "logger.hpp"
#include "datasetbase.h"
#include "pagefault_handler.h"
#include "gpu_globals.h"

struct sigaction old_handler;

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
 * This is the function that is invoked whenever a segfault is encountered by the program.
 *
 * This function identifies if the segfault encountered is triggered by the pagefault mechanism
 * employed by the GPU framework for automatic data transfer to CPU and transfer the data if needed
 *
 * Other segfaults are handeled as regular errors
 *
*/
void pagefault_handler(int sig, siginfo_t *info, void *context)
{
   void *fault_addr = info->si_addr;

   if(fault_addr)
   {
      /*
       * Pick the address which is greater than (or) equal to the fault address
       *
       * By passsing the largest possible pointer address, it is made sure that if fault_addr == one of DSB data pointer's address,
       * we get the element with starting_addr strictly > fault_ddr
      */
      std::set<std::pair<void*, dataSetBase*>>::iterator it = dsb_addr_set.upper_bound(std::make_pair(fault_addr, reinterpret_cast<dataSetBase*>(std::numeric_limits<uintptr_t>::max())));

      bool override_decrement = false; // If the triggering address belongs to the last element of the set

      // If the fault occured at non zero index of the last element, the above search will return end() but we still need to verify if is in the address range of the last element
      if(it == dsb_addr_set.end())
      {
         std::set<std::pair<void*, dataSetBase*>>::iterator last_element = --dsb_addr_set.end();
         void* bounding_address = static_cast<void*>(static_cast<char*>(last_element->first) + last_element->second->byte_size());
         if(fault_addr <= bounding_address)
         {
            it = last_element;
            override_decrement = true; // In this case fault_addr is not equal to the it->first but we since we have manually pointed it to the correct element, we don't need to decrement
         }
      }

      // If not found in the list (or) if it is the first member of the list without overriding, then it is general SEGFAULT which we do not handle and pass on to mpi_abort()
      if(it != dsb_addr_set.end() || (!override_decrement && it == dsb_addr_set.begin()))
      {
         if(!override_decrement)
         {
            assert(it->first > fault_addr);
            it--;
         }
         dataSetBase* const dsb_entry = it->second;
         void* bounding_address = static_cast<void*>(static_cast<char*>(it->first) + it->second->byte_size());
         if(fault_addr <= bounding_address) // If the fault_addr is in the holes/gap between the two entries in an array, it is a non GPU segfault
         {
            dsb_entry->transfer_to_cpu(in_const_operator); // Transfer the variable to CPU
            in_const_operator = false;
            return;
         }
      }
   }

   // The segfault 'might' be GPU realted but not managed by GDF, in that case it is safe to pass it to old handler

   if ( (old_handler.sa_flags & SA_SIGINFO) && old_handler.sa_sigaction) // If the modern handler is set for SIGINFO and present, call it
   {
      old_handler.sa_sigaction(sig, info, context);
   }
   // SIG_DFL and SIG_IGN should be handeled specially as they are not function pointers (https://en.cppreference.com/w/c/program/SIG_strategies)
   else if (old_handler.sa_handler == SIG_DFL)
   {
      // register the default handler and raise the signal
      signal(sig, SIG_DFL);
      raise(sig);
   }
   else if (old_handler.sa_handler == SIG_IGN)
   {
      // Ignore if previous handler was SIG_IGN
      return;
   }
   else if (old_handler.sa_handler) // If the legacy handler is set, call the legacy handler
   {
      old_handler.sa_handler(sig);
   }

   // All SEGFAULTS must be handled by this handler (or) the old_handler. This section is for informing the user about unresolved signals
   char err_msg[200];
   sprintf(err_msg, "Unresolved SIGSEGV at address: %p", fault_addr);
   log_msg<CDF::LogLevel::ERROR>(err_msg);
}

// Setup the the custom pagefault handler
void setup_pagefault_handler()
{
   in_const_operator = 0; // This should only be changed to 0,1 in the data access calls and reset to 0 in pagefault handler

   system_page_size = sysconf(_SC_PAGESIZE); // Get the system page size

   struct sigaction sa;
   memset(&sa, 0, sizeof(sa)); // Zero out the struct completely

   sa.sa_sigaction = pagefault_handler; // Assign our custom segfault/pagefault handler

   sa.sa_flags = SA_SIGINFO;  // Use siginfo_t

   if (sigaction(SIGSEGV, &sa, &old_handler) == -1) // Register our sigaction struct for handling SIGSEGV signal
   {
      log_msg<CDF::LogLevel::ERROR>("Failure while registering custom pagefault handler!");
   }
}
