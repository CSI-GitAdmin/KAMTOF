#ifndef GPU_GLOBALS_H
#define GPU_GLOBALS_H

#ifdef GPU_MEM_LOG
#include <map>
#include <iostream>
#include <fstream>
#endif // GPU_MEM_LOG

#ifndef CDF_GLOBAL
#define CDF_GLOBAL extern
#endif

namespace GDF
{
class GPUManager_t;
}
CDF_GLOBAL GDF::GPUManager_t* gpu_manager;

#ifdef GPU_MEM_LOG
CDF_GLOBAL std::map<void*, strict_fp_t> gpu_mem_map;
CDF_GLOBAL strict_fp_t tot_gpu_mem_used;
CDF_GLOBAL std::ofstream gpu_mem_usage_log;
#endif // GPU_MEM_LOG

void setup_gpu_globals();
void finalize_gpu_globals();

#endif // GPU_GLOBALS_H
