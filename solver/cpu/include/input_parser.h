#ifndef INPUT_PARSER_H
#define INPUT_PARSER_H
#include <vector>
#include <string>
#include "fp_data_types.h"

class InputParser
{
public:

    bool use_gpu_solver  = false; // Flag to control whether the code will be run on gpu or CPU
    bool implicit_solver = false; // Use implicit formulation of solver
    int  Nx              = 11;    // Number of grid points in the x-direction
    int  Ny              = 11;    // Number of grid points in the y-direction
    int  tol_type        = 0;     // Type of tolerance to be used by the solver: 0 ==> Absolute tol, 1 ==> Relative tol
    int  solver_type     = 0;     // Type of linear solver: 0 ==> Jacobi, 1 ==> BiCGSTAB
    int  num_iter        = 10;    // Number of itrerations that the linear solver will perform
    strict_fp_t tol_val  = 0.01;  // Tolerance at which the solver will stop
    
    // list of all valid input strings
    std::vector<std::string> input_strings = {"use_gpu_solver",
                                              "implicit_solver",
                                              "Nx",
                                              "Ny",
                                              "tol_type",
                                              "tol_val",
                                              "num_iter",
                                              "solver_type"};

    InputParser(std::string input_file);

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

    std::string input_filename;                    
};

#endif /* INPUT_PARSER_H*/