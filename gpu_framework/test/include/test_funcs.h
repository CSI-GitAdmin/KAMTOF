#ifndef TEST_FUNCS_H
#define TEST_FUNCS_H

void setup_gpu_globals_test();
void finalize_gpu_globals_test();
void test_dss_gpu();
// void test_dss_gpu_resize();
void test_gpu_pointer_api_funcs();
void test_gpu_atomics();
// void test_silo_null();

void test_ncpu_ngpu();

#endif // TEST_FUNCS_H