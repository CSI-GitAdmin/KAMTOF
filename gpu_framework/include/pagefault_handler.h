#ifndef PAGEFAULT_HANDLER_H
#define PAGEFAULT_HANDLER_H

#include <signal.h> // For siginfo_t
#include <cstdint> // For uint64_t
#include <set> // For std::pair

void pagefault_handler(int sig, siginfo_t *info, void *context);
void setup_pagefault_handler();
std::pair<void*,uint64_t> allocate_page_aligned_memory(const uint64_t byte_size);
void deallocate_page_aligned_memory(void* m_data, const uint64_t allocation_size);

#endif // PAGEFAULT_HANDLER_H