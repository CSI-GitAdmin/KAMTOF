#! /bin/bash

# Color markup for messages
BOLD='\033[1m' 
CLEAR='\033[0m'
UNDERLINE='\e[4m'
RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
ERROR_COLOR="${RED}${BOLD}"
ERROR_PREFIX="${ERROR_COLOR}[ERROR]${CLEAR}"
INFO_PREFIX="${BLUE}${BOLD}[I]${CLEAR}"
INFO_SUBLEVEL_PREFIX="-->"
INFO_SUCCESS="${GREEN}${BOLD}[I]${CLEAR}"

# --------------------------------------------------------------------------------------------------
# Timing function
# --------------------------------------------------------------------------------------------------
timer()
{
    PROCESS_COMMAND=$1
    PROCESS_DESC=$2
    local start end elapsed
    start=$(date +%s.%N)
    ${PROCESS_COMMAND}
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc -l)
    
    printf "${TIME_PREFIX} Time taken for ${PROCESS_DESC}: %5.2f seconds.\n" ${elapsed}
    #echo "${elapsed}"

    return 0
}

# --------------------------------------------------------------------------------------------------
# Set help information
# --------------------------------------------------------------------------------------------------
usage() 
{

    read -r -d '' usage_text << END

    ${BOLD}  $(basename $0)${CLEAR} [OPTIONS]
    This script runs both the CPU and GPU based solvers for a user-specified grid size.
    The total runtimes for both runs are then output to the screen.
    Current directory: ${BOLD}${PWD}${CLEAR}

    All options have defaults and are optional and case sensiitive.
    ${CLEAR}
    ${UNDERLINE}OPTIONS:${CLEAR}
    ${BOLD}-x <nx>${CLEAR} default:${BOLD} 500${CLEAR}
                Number of grid cells in the x-direction.
    ${BOLD}-y <ny>${CLEAR} default:${BOLD} 500${CLEAR}
                Number of grid cells in the y-direction.
    ${BOLD}-n <num_procs>${CLEAR}  default:${BOLD} `nproc --all`${CLEAR}
                Number of processes to use for the CPU based code.
                The GPU solver will only be executed on a single GPU card.
    ${BOLD}-e <exec_name>${CLEAR}  default:${BOLD} kamtof${CLEAR}
                Name of executable.
    ${BOLD}-r <run_dir>${CLEAR}  default:${BOLD} ../../laplace_solve${CLEAR}
                Base directory where the CPU & GPU solvers will be run.
    ${BOLD}-o <overwrite_output>${CLEAR}  default:${BOLD} FALSE ${CLEAR}
                Overwrite old base directory if it exists.
    ${BOLD}-h <help>${CLEAR}          Print this help text. 
    ${CLEAR}
END
    echo -e "$usage_text"

    return 0
}

# --------------------------------------------------------------------------------------------------
# Capture command line options
# --------------------------------------------------------------------------------------------------
gather_options() 
{
    while getopts "h-:x:y:n:e:r:o:" OPTION; do
        case ${OPTION} in
        x    ) NX=${OPTARG}        ;;
        y    ) NY=${OPTARG}        ;;
        n    ) NPROCS=${OPTARG}    ;;
        e    ) EXEC_NAME=${OPTARG} ;;
        r    ) BASE_DIR=${OPTARG}  ;;
        o    ) OVERWRITE=${OPTARG} ;;
        h    ) usage
                exit 0;;
        -     )
                echo -e "${ERROR_PREFIX} '--' (double-dash) options are not currently available"
                echo -e "       use -h option for usage information"
                exit 1                      ;;
        :|?|* )
                echo -e "${ERROR_PREFIX} Unknown option ${OPTION}."
                echo -e "       use -h option for usage information"
                exit 1                      ;;
        esac
    done

    # Number of grid cells in the x-direction
    if [ -z ${NX} ]
    then
        NX="500"
    fi

    # Number of grid cells in the y-direction
    if [ -z ${NY} ]
    then
        NY="500"
    fi

    # Number of processors to be used for the CPU run
    if [ -z ${NPROCS} ]
    then
        NPROCS=`nproc --all`
    fi

    # Executable name
    if [ -z ${EXEC_NAME} ]
    then
        EXEC_NAME="kamtof"
    fi

    # Executable name
    if [ -z ${BASE_DIR} ]
    then
        BASE_DIR="../../laplace_solve"
    fi

    # Option to overwrite base directory if it exists
    if [ -z ${OVERWRITE} ]
    then
        OVERWRITE="FALSE"
    fi

  return 0
}

# --------------------------------------------------------------------------------------------------
# Check validity of a single dependency
# --------------------------------------------------------------------------------------------------
check_individual_dependencies()
{
    # Name of dependency to be checked
    local DEP_NAME=$1

    # Check if dependency exists
    if ! command -v "${DEP_NAME}" >/dev/null 2>&1
    then
        echo -e "${ERROR_PREFIX} Required dependency ${BOLD}${DEP_NAME}${CLEAR} not found."
        exit 4
    else
        echo -e "${INFO_SUBLEVEL_PREFIX} ${DEP_NAME} found."
    fi

    return 0
}

# --------------------------------------------------------------------------------------------------
# Loop over all required dependencies and check their existence
# --------------------------------------------------------------------------------------------------
check_all_dependencies()
{
    # Info message
    echo -e "${INFO_PREFIX} Checking required dependencies..."
    
    # List of all dependencies
    KAMTOF_DEPS=("mpirun")
    
    # Loop over all dependencies and check if they exist
    for dep in "${KAMTOF_DEPS[@]}"
    do
        check_individual_dependencies ${dep}
    done

    # Info message
    echo -e "${INFO_PREFIX} Checking required dependencies...${GREEN}successful${CLEAR}"
    echo ""

    return 0
}

# --------------------------------------------------------------------------------------------------
# Check exit code for command and print error if it fails
# --------------------------------------------------------------------------------------------------
check_error_status()
{
    local EXIT_CODE=$1
    local ERROR_MSG=$2

    if [[ ${EXIT_CODE} -ne 0 ]]
    then
        echo -e "${ERROR_MSG}"
        exit 5
    fi

    return 0
}

# --------------------------------------------------------------------------------------------------
# Load dependencies
# --------------------------------------------------------------------------------------------------
load_dependencies()
{
    # Load required modules for KAMTOF
    MODULE_LOG=module.log
    echo -e "${INFO_PREFIX} Loading dependencies using modules for KAMTOF."
    module purge
    
    # Error message to be displayed in case of failure
    ERROR_MSG="${ERROR_PREFIX} Could not load modules. Please check log file: ${MODULE_LOG}."

    #Error checking
    module load kamtof > ${MODULE_LOG} 2>&1
    check_error_status ${PIPESTATUS[0]} "${ERROR_MSG}"

    return 0
}

# --------------------------------------------------------------------------------------------------
# Create base directory if it doesn't exist
# --------------------------------------------------------------------------------------------------
create_base_dir()
{
    # Check existence of directory
    if [ -d ${BASE_DIR} ];
    then
        if [ ${OVERWRITE} == "FALSE" ]
        then
            echo -ne "${ERROR_PREFIX} ${RED}Directory \"${BASE_DIR}\" exists. Please rename/delete before continuing. "
            echo -e "Use option \"-o TRUE\" to overwrite contents of the directory.${CLEAR}"
            exit 2
        else    
            echo -e "${INFO_PREFIX} Removing old output directory (${BASE_DIR})."
            rm -rf ${BASE_DIR}
            mkdir -p ${BASE_DIR}
        fi
    else
        mkdir -p ${BASE_DIR}
    fi

    # Setup different directories for the CPU and GPU runs
    create_cpu_gpu_run_dirs

    return 0 
}

# --------------------------------------------------------------------------------------------------
# Create CPU and GPU run directories
# --------------------------------------------------------------------------------------------------
create_cpu_gpu_run_dirs()
{
    # Save path to current directory
    CURR_DIR=${PWD}

    # Once the base directory has been created, create individual directories for CPU and GPU
    # runs and place the executable in these directories
    
    # Setup CPU case
    setup_case_dir "cpu" "${CURR_DIR}"

     # Setup GPU case
    setup_case_dir "gpu" "${CURR_DIR}"
    

}

# --------------------------------------------------------------------------------------------------
# Setup input files for a given case
# --------------------------------------------------------------------------------------------------
setup_case_dir()
{
    # Obtain case type (CPU or GPU) from input argument
    CASETYPE=${1}

    # Set case directory path
    local CASE_DIR=${BASE_DIR}/${CASETYPE}

    # Create case directory
    mkdir -p ${CASE_DIR}

    # Setup integer value for use_gpu_solver input option
    USE_GPU="1"
    if [ ${CASETYPE} == "cpu" ]
    then
        USE_GPU="0"
    fi

    # Create input file with appropriate options
    create_input_file "${CASE_DIR}"

    # Change directory into case dir to enable softlinking
    cd ${CASE_DIR}

    # Soft link executable in the directories
    ln -sf ${CURR_DIR}/../../build/solver ${EXEC_NAME}

    # Change directory back to previous location
    cd ${CURR_DIR}

    return 0

}


# --------------------------------------------------------------------------------------------------
# Check if build directory contains executable
# --------------------------------------------------------------------------------------------------
check_executable()
{
    # Check that the build directory exists and it contains an executable
    BUILD_DIR="../../build"
    if [[ -d ${BUILD_DIR} ]]
    then
        echo -e "${INFO_PREFIX} ${GREEN}bin${CLEAR} directory found at \"${BUILD_DIR}\"."

        # Check if solver executable exists
        if [[ -x "${BUILD_DIR}/solver" ]]
        then
            echo -e "${INFO_PREFIX} ${GREEN}solver${CLEAR} executable found in \"${BUILD_DIR}\"."
        else        
            echo -e "${ERROR_PREFIX} Executable \"solver\" ${RED}not found${CLEAR} in ${BUILD_DIR}."
            exit 5
        fi
    else      
        echo -e "${ERROR_PREFIX} ${BUILD_DIR} not found."
        exit 5
    fi
}

# --------------------------------------------------------------------------------------------------
# Create input file
# --------------------------------------------------------------------------------------------------
create_input_file()
{
    local CASE_DIR=$1
    INPUTFILE="${CASE_DIR}/laplace_${CASETYPE}.in"

    echo "use_gpu_solver:  ${USE_GPU}" > ${INPUTFILE}
    echo "implicit_solver: 1" >> ${INPUTFILE}
    echo "Nx:              ${NX}" >> ${INPUTFILE}
    echo "Ny:              ${NY}" >> ${INPUTFILE}
    echo "tol_type:        abs" >> ${INPUTFILE}
    echo "tol_val:         0.01" >> ${INPUTFILE}
    echo "num_iter:        10" >> ${INPUTFILE}
    echo "solver_type:     jacobi" >> ${INPUTFILE}
}


# --------------------------------------------------------------------------------------------------
# Obtain run time for the given case (reads the last line)
# --------------------------------------------------------------------------------------------------
get_run_time()
{
    # Name of file to be read for getting runtime
    local LOG_FILE=$1

    # Read last line of file and store to variable 
    local LAST_LINE=$(tail -n 1 ${LOG_FILE})

    # Extract last "word" from the last line
    local RUN_TIME=${LAST_LINE##* }

    # Output the run time using echo to be captured by the function
    echo "${RUN_TIME:0:5}"

    return 0

}

# --------------------------------------------------------------------------------------------------
# Run solver for given device
# --------------------------------------------------------------------------------------------------
run_solver()
{
    local CASETYPE=$1
    local CASE_DIR=${BASE_DIR}/${CASETYPE}

    # Save current location for later
    local CURR_DIR=${PWD}

    # Move to case directory to run solver
    cd ${CASE_DIR}

    echo -ne "${INFO_PREFIX} Running ${CASETYPE} solver..."

    local LOG_FILE="${CASETYPE}_run.log"
    
    # Name of input file
    local INPUTFILE="laplace_${CASETYPE}.in"

    # Set error message to be displayed if command fails
    ERROR_MSG="${CASETYPE} run ${RED}${BOLD}FAILED${CLEAR}. Please check log file: ${LOG_FILE}"

    # For GPUs, only a single GPU card is allowed for now
    if [ ${CASETYPE} == "gpu" ]
    then
        NPROCS="1"
    fi

    # Check error status
    mpirun -np ${NPROCS} ./${EXEC_NAME} ${INPUTFILE} > ${LOG_FILE} 2>&1
    check_error_status ${PIPESTATUS[0]} "${ERROR_MSG}"

    # Get run time from the log file
    RUN_TIME=$(get_run_time ${LOG_FILE})

    echo -e "\r${INFO_PREFIX} Running ${CASETYPE} solver...${GREEN}successful${CLEAR} ${BLUE}(Runtime = ${RUN_TIME}s)${CLEAR}"

    # Go back to previous location
    cd ${CURR_DIR}

    return 0;


}

# --------------------------------------------------------------------------------------------------
# MAIN Function
# --------------------------------------------------------------------------------------------------
main()
{
    echo -e "${INFO_PREFIX} Running CPU and GPU solvers for KAMTOF."

    # Gather command line input arguments
    gather_options $@

    # Load dependencies for KAMTOF
    load_dependencies

    # Check dependencies
    check_all_dependencies

    # Check if executable exists for the Laplace solver
    check_executable 

    # Create directory to run CPU and GPU executables
    create_base_dir

    # Execute solver for CPU
    run_solver "cpu"

    # Execute solver for GPU
    run_solver "gpu"

    echo -e "${INFO_SUCCESS} CPU and GPU runs for the Laplace solver ${GREEN}successful${CLEAR}."

    exit 0
}

# Run script
main $@