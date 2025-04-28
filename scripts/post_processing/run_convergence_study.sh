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
    
    #printf "${TIME_PREFIX} Time taken for ${PROCESS_DESC}: %5.2f seconds.\n" ${elapsed}
    echo "${elapsed}"

    return 0
}

# --------------------------------------------------------------------------------------------------
# Set help information
# --------------------------------------------------------------------------------------------------
usage() 
{
    read -r -d '' usage_text << END

    ${BOLD}  $(basename $0)${CLEAR} [OPTIONS]
    This script performs an order of accuracy study for the Laplace solver.
    Detailed information about the order of accuracy study can be found at:
    Current directory: ${BOLD}${PWD}${CLEAR}

    All options have defaults and are optional and case sensiitive.
    ${CLEAR}
    ${UNDERLINE}OPTIONS:${CLEAR}
    ${BOLD}-r <run_directory>${CLEAR} default:${BOLD} ${KAMTOF_ROOT}/../output${CLEAR}
                Directory where the output from the study will be stored.
                This path is relative to the current directory.
    ${BOLD}-d <device_type>${CLEAR}  default:${BOLD} gpu ${CLEAR}
                Type of device the Laplace solver will be run on.
                Only two valid options: ${BOLD}${RED}cpu${CLEAR} and ${BOLD}${RED}gpu${CLEAR}.
    ${BOLD}-o <overwrite_output>${CLEAR}  default:${BOLD} FALSE ${CLEAR}
                Overwrite old output directory if it exists.
    ${BOLD}-n <num_procs>${CLEAR}  default:${BOLD} 1${CLEAR}
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
    while getopts "h-:r:d:o:n:" OPTION; do
        case ${OPTION} in
        r     ) RUN_DIR=${OPTARG}    ;;
        d     ) DEVICE=${OPTARG}     ;;
        o     ) OVERWRITE=${OPTARG}  ;;
        n     ) NPROCS=${OPTARG}     ;;
        h     ) usage
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

    # Set ./output as default output directory for results
    if [ -z ${RUN_DIR} ]
    then
        RUN_DIR="${KAMTOF_ROOT}/../output"
    fi

    # Set CPU as default device
    if [ -z ${DEVICE} ]
    then
        DEVICE="gpu"
    fi

    # Set overwrite option
    if [ -z ${OVERWRITE} ]
    then
        OVERWRITE="FALSE"
    fi

    # Number of procs
    if [ -z ${NPROCS} ]
    then
        NPROCS="1"
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

    # Minimum acceptable version number
    #  seems a bit complicated; TO DO
    local DEP_REQ_VERSION=${2:-"0"}

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
# Check availability of individual Python modules
# --------------------------------------------------------------------------------------------------
check_python_module()
{
    # Name of dependency to be checked
    local DEP_NAME=$1

    # Minimum acceptable version number
    #  seems a bit complicated; TO DO
    local DEP_REQ_VERSION=${2:-"0"}

    # Check if dependency exists
    if ! python3 -c "import ${DEP_NAME}" >/dev/null 2>&1
    then
        echo -e "${ERROR_PREFIX} Required Python module ${BOLD}${DEP_NAME}${CLEAR} not found."
        
        # Check if user wants to install dependency using PIP3
        echo -ne "${INFO_PREFIX} Do you want to install it using PIP3 (${BLUE}y/n${CLEAR})? "
        read -n 1 -r
        echo ""
        
        if [[ ${REPLY} =~ ^[Yy]$ ]]
        then
            pip3 install ${DEP_NAME}
            if [[ $? -ne 0 ]]
            then
                echo -e "${ERROR_PREFIX} Unable to install ${DEP_NAME} using PIP3. ${RED}Script failed.${CLEAR}"
                echo ""
                exit 6
            fi
        else
            exit 4
        fi
        
    else
        echo -e "${INFO_SUBLEVEL_PREFIX} ${BLUE}(Python)${CLEAR} ${DEP_NAME} found."
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
    KAMTOF_DEPS=("mpirun" "pip3" "python3")
    
    # Loop over all dependencies and check if they exist
    for dep in "${KAMTOF_DEPS[@]}"
    do
        check_individual_dependencies ${dep}
    done
    
    # List of required python modules
    PYTHON_MODULES=("sys" "numpy" "tabulate" "matplotlib" "PyQt5")

    # Loop over all required Python modules and check if it exists
    # If the module doesn't exist, the user can install it using PIP3
    for python_dep in "${PYTHON_MODULES[@]}"
    do 
        check_python_module ${python_dep}
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
    ERROR_MSG="${ERROR_PREFIX} Could not load modules. Please check log file: ${MODULE_LOG}"

    #Error checking
    module load kamtof > ${MODULE_LOG} 2>&1
    check_error_status ${PIPESTATUS[0]} "${ERROR_MSG}"

    return 0
}

# --------------------------------------------------------------------------------------------------
# Setup environment for KAMTOF using script
# --------------------------------------------------------------------------------------------------
setup_environment_using_script()
{
    # Set the KAMTOF root directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export KAMTOF_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"

    source ${KAMTOF_ROOT}/scripts/setup_env.sh
}

# --------------------------------------------------------------------------------------------------
# Load dependencies using scripts
# --------------------------------------------------------------------------------------------------
load_dependencies_using_script()
{
    # Load required modules for KAMTOF
    echo -e "${INFO_PREFIX} Loading dependencies using scripts for KAMTOF."
    
    # Error message to be displayed in case of failure
    ERROR_MSG="${ERROR_PREFIX} Could not load dependencies."

    #Error checking
    setup_environment_using_script
    check_error_status ${PIPESTATUS[0]} "${ERROR_MSG}"

    return 0
}

# --------------------------------------------------------------------------------------------------
# Start MAIN script
# --------------------------------------------------------------------------------------------------

# Load dependencies
load_dependencies_using_script


# Collect input options
gather_options $@

# Check that the build directory exists and it contains an executable
BUILD_DIR="${KAMTOF_ROOT}/../build_release"
EXE="${BUILD_DIR}/solver"

if [[ -d ${BUILD_DIR} ]]
then
    echo -e "${INFO_PREFIX} ${GREEN}bin${CLEAR} directory found at \"${BUILD_DIR}\"."

    # Check if solver executable exists
    if [[ -x "${EXE}" ]]
    then
        echo -e "${INFO_PREFIX} ${GREEN}solver${CLEAR} executable found in \"${BUILD_DIR}\"."
    else        
        echo -e "${ERROR_PREFIX} Executable \"${EXE}\" ${RED}not found${CLEAR} in ${BUILD_DIR}."
        exit 5
    fi
else      
    echo -e "${ERROR_PREFIX} ${BUILD_DIR} not found."
    exit 5
fi

# Get list of dynamically linked dependencies
ldd ${EXE} > ${RUN_DIR}/ldd.log 2>&1

# Check dependencies
check_all_dependencies

TEMP_FILE="${KAMTOF_ROOT}/scripts/post_processing/laplace_template.in"

echo -e "${INFO_PREFIX} Executing script to perform order of accuracy study."

# Check for existence of file
if [ ! -f "${TEMP_FILE}" ]; then
    echo -e "${ERROR_PREFIX} ${RED}File ${TEMP_FILE} not found.${CLEAR}"
    exit 1
fi

# List the grid sizes to be used for the convergence study
grid_sizes=(10 20 40 80)

# Text to be replaced
SEARCH_TEXT="NGRID"

# get number of grids to be run
NRUNS=${#grid_sizes[@]}

# Type of run - CPU or GPU
DEVICE_INT=0
if [ ${DEVICE} == "gpu" ]
then
    DEVICE_INT=1
fi

# Check existence of directory
if [ -d ${RUN_DIR} ];
then
    if [ ${OVERWRITE} == "FALSE" ]
    then
        echo -ne "${ERROR_PREFIX} ${RED}Directory \"${RUN_DIR}\" exists. Please rename/delete before continuing. "
        echo -e "Use option \"-o TRUE\" to overwrite contents of the directory.${CLEAR}"
        exit 2
    else    
        echo -e "${INFO_PREFIX} Removing old output directory (${RUN_DIR})."
        rm -rf ${RUN_DIR}
        mkdir -p ${RUN_DIR}
    fi
else
    mkdir -p ${RUN_DIR}
fi

GRID_LIST=""

echo -e "${INFO_PREFIX} Starting loop..."

# Save current directory for later
CURR_DIR=${PWD}

#loop through each grid size
for ((irun=1; irun<=NRUNS; irun++))
do    
    # Get local value of grid size for this particular case
    nx=${grid_sizes[$((irun-1))]}

    # Create a variable that is a comma-delimited version of the grid size variable
    # This will be passed onto the Python script later for order of accuracy calculations
    GRID_LIST+="${nx},"

    # Print information to screen
    echo -e "${INFO_SUBLEVEL_PREFIX} Running case ${irun}/${NRUNS} with Nx = Ny = ${nx}"

    # Create directory for given case in the result directory
    CASE_DIR="${RUN_DIR}/nx_${nx}"
    mkdir -p ${CASE_DIR}

    # Create copy of template file
    OUTFILENAME=laplace_nx_${nx}.in
    OUT_FILE="${CASE_DIR}/${OUTFILENAME}"
    cp -r ${TEMP_FILE} ${OUT_FILE}

    # Replace search string with the actual value
    sed -i "s/${SEARCH_TEXT}/${nx}/g" "${OUT_FILE}"
    sed -i "s/DEVICE_INT/${DEVICE_INT}/g" ${OUT_FILE} 

    # Switch directory to case directory
    cd ${CASE_DIR}

    # Execute solver
    LOG_FILE="run_nx_${nx}.log"

    # Error message to be output in case of failure
    ERROR_MSG="${ERROR_PREFIX} Run failed. Please check log file: ${CASE_DIR}/${LOG_FILE}"

    # Error checking
    mpirun -np ${NPROCS} ${EXE} ${OUTFILENAME} > ${LOG_FILE} 2>&1
    check_error_status ${PIPESTATUS[0]} "${ERROR_MSG}"

    # Change directories back to where script is being run
    cd ${CURR_DIR}

done

# Execute python script to analyze order of accuracy
echo -e ""
echo -e "${INFO_PREFIX} Running python script to compute order of accuray using cases located in \"${RUN_DIR}\":"

# Set name of Python log file
PYTHON_LOG="${RUN_DIR}/order.log"

# Create comma-delimited list of grid sizes to be passed on to Python
GRID_LIST="${GRID_LIST%?}"

python3 ${KAMTOF_ROOT}/scripts/post_processing/analytical_laplace_solution.py ${GRID_LIST} ${RUN_DIR} ${DEVICE} 2>&1 | tee ${PYTHON_LOG}

# Store exit code from python script
EXIT_CODE=${PIPESTATUS[0]}

# Check for success of script
if [[ ${EXIT_CODE} -eq 0 ]]
then
    # Print success message
    echo -e ""
    echo -e "${INFO_SUCCESS} Order of accuracy script completed ${GREEN}${BOLD}successfully${CLEAR}."
    exit 0
else
    # Print error message
    echo -e "${ERROR_PREFIX} Order of accuracy check ${RED}${BOLD}FAILED.${CLEAR}"
    exit 2
fi
