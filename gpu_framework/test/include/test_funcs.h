#ifndef TEST_FUNCS_H
#define TEST_FUNCS_H

void setup_gpu_globals_test();
void finalize_gpu_globals_test();
void test_global_local_range_setup();
void setup_cdf_vars_for_gpu_framework_tests();
void test_dss_gpu();
void test_dss_gpu_resize();
void test_gpu_pointer_api_funcs();
void test_gpu_atomics();
void test_silo_null();
void test_ncpu_ngpu();
void backend_testing(bool run_backend_tests);

void porting_stage_scenario(bool run);
void demonstrate_temp_write(bool run);

#endif // TEST_FUNCS_H