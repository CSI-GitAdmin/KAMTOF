#ifndef INPUT_PARSER_H
#define INPUT_PARSER_H
#include <vector>
#include <string>
#include "fp_data_types.h"

class InputParser
{
public:

    bool use_gpu_solver  = true;  // Flag to control whether the code will be run on gpu or CPU
    bool implicit_solver = true;  // Use implicit_solver formulation of solver
    int  Nx              = 1000;  // Number of grid points in the x-direction
    int  Ny              = 1000;  // Number of grid points in the y-direction
    int  tol_type        = 0;     // Type of tolerance to be used by the solver: 0 ==> Absolute tol, 1 ==> Relative tol
    int  solver_type     = 0;     // Type of linear solver: 0 ==> Jacobi, 1 ==> BiCGSTAB
    int  num_iter        = 10;    // Number of itrerations that the linear solver will perform
    strict_fp_t tol_val  = 1000;  // Tolerance at which the solver will stop
    uint64_t gpu_global_range  = 262144;  // Global range for GPU kernels
    uint64_t gpu_local_range  = 256;  // Global range for GPU kernels
    
    // list of all valid input strings
    std::vector<std::string> input_strings = {"use_gpu_solver",
                                              "implicit_solver",
                                              "Nx",
                                              "Ny",
                                              "tol_type",
                                              "tol_val",
                                              "num_iter",
                                              "solver_type",
                                              "gpu_global_range",
                                              "gpu_local_range"};

    InputParser(std::string input_file);

    InputParser();

    void read_inputs(const std::string& input_filename);
    
    std::string find_input_option(const std::vector<std::string>& input_strings,
                                  const int ninputs,
                                  const std::string& line,
                                  int& input_count);
    
    void parse_input_value(std::string& input_string,
                           std::string& line);
    
    void print_input_struct();
    
    template <typename T>
    void print_variable_value(std::string& varname, T& value);

    std::string input_filename = "NULL"; // default name of input filename                    
};

#endif /* INPUT_PARSER_H*/