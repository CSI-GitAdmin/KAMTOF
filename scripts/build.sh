#! /bin/bash

# Color markup for messages
BOLD='\033[1m' 
CLEAR='\033[0m'
UNDERLINE='\e[4m'
RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
ERROR_COLOR="${RED}${BOLD}"
ERROR_PREFIX="\n${ERROR_COLOR}[ERROR]${CLEAR}"
INFO_PREFIX="${BLUE}${BOLD}[I]${CLEAR}"
TIME_PREFIX="${RED}[T]${CLEAR}"
INFO_SUBLEVEL_PREFIX="-->"
INFO_SUCCESS="${GREEN}${BOLD}[I]${CLEAR}"
WARNING_PREFIX="${BOLD}[WARNING]${CLEAR}"

# --------------------------------------------------------------------------------------------------
# Safe exit to make sure exit does not close terminal
# --------------------------------------------------------------------------------------------------
safe_exit() 
{
    local EXIT_CODE=$1
    if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
        # Script is being executed (not sourced), so exit normally
        exit "${EXIT_CODE}"
    else
        # Script is being sourced, so use 'return'
        return "${EXIT_CODE}"
    fi
}

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
    echo -e ""

    return 0
}

# --------------------------------------------------------------------------------------------------
# Check exit code for command and print error if it fails
# --------------------------------------------------------------------------------------------------
check_error_status()
{
    local EXIT_CODE=$1
    local ERROR_MSG=$2

    GLOBAL_ERROR_CODE=0

    if [[ ${EXIT_CODE} -gt 0 ]]
    then
        GLOBAL_ERROR_CODE=5
        echo -e "${ERROR_PREFIX} ${ERROR_MSG}"
        safe_exit ${GLOBAL_ERROR_CODE}
    elif [[ ${EXIT_CODE} -lt 0 ]]
    then
        echo -e "${WARNING_PREFIX} ${ERROR_MSG}"
        GLOBAL_ERROR_CODE=6
        return ${GLOBAL_ERROR_CODE}
    fi

    return 0
}

# --------------------------------------------------------------------------------------------------
# Perform system checks
# --------------------------------------------------------------------------------------------------
system_check()
{
    SYS_TYPE=$(uname -o)
    HW_TYPE=$(uname -m)

    # Get system information
    case ${SYS_TYPE} in
    GNU/Linux|Linux) UNAME_OS="linux"  ;; 
    Cygwin         ) UNAME_OS="cygwin" ;; 
    Darwin         ) UNAME_OS="mac"    ;; 
    *              ) echo -e "${ERROR_PREFIX}: Unknown operating system type."
        safe_exit 1 ;;
    esac

    # Get hardware information
    case `uname -m` in
    x86_64 )          UNAME_HW="x64"   ;;
    aarch64)         UNAME_HW="arm64" ;;
    *      ) echo -e "${ERROR_PREFIX}: Unknown processor (hardware) type."
        safe_exit 1 ;;
    esac

    read -r -d '' system_text << END

    ${BOLD}[I] Current system configuration:${CLEAR}
        Operating System  : ${BOLD}${UNAME_OS}${CLEAR}
        Hardware/Processor: ${BOLD}${UNAME_HW}${CLEAR} 
        Current directory : ${BOLD}${PWD}${CLEAR}
        Kamtof Root       : ${BOLD}${KAMTOF_ROOT}${CLEAR}
    ${CLEAR}
END
    
    echo -e "${system_text}"

    # Check operating system
    if [ ${UNAME_OS} != "linux" ]
    then
        echo -e "${RED}${BOLD}[ERROR]${CLEAR} This script can only be executed on GNU/Linux systems. You have a ${SYS_TYPE} based system."
        safe_exit 1
    fi

    # Check hardware
    if [ ${UNAME_HW} != "x64" ]
    then
        echo -e "${RED}${BOLD}[ERROR]${CLEAR} ${RED}This script can only be executed on x86_64 hardware. You have a ${HW_TYPE} based hardware."
        safe_exit2
    fi

    return 0
}

# --------------------------------------------------------------------------------------------------
# Set help information
# --------------------------------------------------------------------------------------------------
usage() 
{
    read -r -d '' usage_text << END

    ${BOLD}  $(basename $0)${CLEAR} [OPTIONS]
    This script clones and builds the current release version of the KAMTOF solver code.
    Repository: https://git-dev.converge.global/sidarth.narayanan/kamtof
    Current directory: ${BOLD}${PWD}${CLEAR}

    All options have defaults and are optional. The default folders assume the 
    script is run from the scripts folder in the KAMTOF source code (${BLUE}./kamtof/scripts${CLEAR}).
    The default ${BLUE}${BOLD}build${CLEAR} and ${BLUE}${BOLD}bin${CLEAR} directories will be created two levels above the scripts directory.
    
    The following dependencies are required:
        1. ${BOLD} mpirun${CLEAR}
        2. ${BOLD} git${CLEAR}
        3. ${BOLD} wget${CLEAR}
    ${CLEAR}
    ${UNDERLINE}OPTIONS:${CLEAR}
    ${BOLD}-s <source_directory>${CLEAR} default:${BOLD} ${KAMTOF_ROOT}${CLEAR}
                Directory where the KAMTOF code will be cloned.
                This path is relative to the current directory.
    ${BOLD}-b <build_directory>${CLEAR}  default:${BOLD} ${KAMTOF_ROOT}/../build ${CLEAR}
                Directory where the KAMTOF code will be cloned and built (relative to source directory).
                This directory will exist inside the base directory.
                The executable will also be placed in this directory.
    ${BOLD}-t <build_type>${CLEAR}    default:${BOLD} RELEASE${CLEAR}
    ${BOLD}-j <num_proc>${CLEAR}      default:${BOLD} `nproc --all`${CLEAR}
                Number of processors to be used for compilation.
                The default is the maximum number of processors on this machine
                (computed using the 'nproc --all' command)
    ${BOLD}-g <clone_kamtof>${CLEAR}      default:${BOLD} FALSE${CLEAR}
                Clone KAMTOF from the Git repository.
                The default is FALSE.
    ${BOLD}-e <exec_name>${CLEAR}      default:${BOLD} kamtof${CLEAR}
                Name of executable to be placed in the bin directory.
    ${BOLD}-d <del_src>${CLEAR}      default:${BOLD} FALSE${CLEAR}
                Delete source directory if it exists.
    ${BOLD}-h${CLEAR}          Print this help text. 
    ${CLEAR}
END
    echo -e "$usage_text"

    return 0

}

# --------------------------------------------------------------------------------------------------
# Set defaults for all option
# --------------------------------------------------------------------------------------------------
set_defaults()
{
    # Build type
    if [ -z ${BLD_TYPE} ]
    then
        BLD_TYPE="RELEASE"  
    fi

    # Source directory
    if [ -z ${SRC_DIR} ]
    then
        SRC_DIR="${KAMTOF_ROOT}/.."
    fi
    
    # Build directory
    if [ -z ${BLD_DIR} ]
    then
        BLD_DIR="build"
    fi

    # Executable name
    if [ -z ${EXEC_NAME} ]
    then
        EXEC_NAME="kamtof"
    fi

    # Number of threads to be used for building code
    if [ -z ${THREAD_COUNT} ]
    then
        THREAD_COUNT=$(nproc --all)
    fi

    # Clone KAMTOF or use locally available source code
    if [ -z ${CLONE_KAMTOF} ]
    then
        CLONE_KAMTOF="FALSE"
    fi
    
    # Directory to store log files
    LOG_DIR=${SRC_DIR}/logs

    # Delete existing base directory before cloning new directory
    if [ -z ${DEL_SRC} ]
    then
        DEL_SRC="FALSE"
    fi

    return 0
}

# --------------------------------------------------------------------------------------------------
# Capture command line options
# --------------------------------------------------------------------------------------------------
gather_options() 
{
    # Get current directory
    CURR_DIR=${PWD}

    while getopts "h-:b:s:t:j:g:e:d:" OPTION; do
        case ${OPTION} in
        t     ) BLD_TYPE=${OPTARG}           ;;
        s     ) SRC_DIR=${CURR_DIR}/${OPTARG};;
        b     ) BLD_DIR=${OPTARG};;
        j     ) THREAD_COUNT=${OPTARG}       ;;
        g     ) CLONE_KAMTOF=${OPTARG}       ;;
        e     ) EXEC_NAME=${OPTARG}          ;;
        d     ) DEL_SRC=${OPTARG}            ;;
        h     ) usage
                safe_exit 0;;
        -     )
                echo -e "${ERROR_PREFIX} '--' (double-dash) options are not currently available"
                echo -e "       use -h option for usage information"
                safe_exit 1                      ;;
        :|?|* )
                echo -e "${ERROR_PREFIX} Unknown option ${OPTION}."
                echo -e "       use -h option for usage information"
                safe_exit 1                      ;;
        esac
    done

    # Set default values for variables that have not been set
    set_defaults

    # Consistency check if using local source to compile KAMTOF
    if [ ${CLONE_KAMTOF} == "FALSE" ]
    then
        if [ -z "${SRC_DIR}" ]
        then
            echo -e "${ERROR_PREFIX} ${RED}Please use the option \"-s\" to specify a base directory for KAMTOF when using the option \"-g FALSE\".${CLEAR}"
            safe_exit 23
        fi
        
        # Consistency check to make sure that local repository is not deleted if using local repository
        if [ ${DEL_SRC} == "TRUE" ]
        then
            echo -e "${ERROR_PREFIX} ${RED}When using local source for KAMTOF, \"-d\" cannot be TRUE. Please use \"-d FALSE\" or \"-g TRUE\".${CLEAR}"
            safe_exit 24
        fi
    fi  

    #Set full path of build directory
    BLD_DIR="${SRC_DIR}/${BLD_DIR}"
    LOG_DIR="${SRC_DIR}/logs"

    # Create directory to store log files
    mkdir -p ${LOG_DIR}

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
        safe_exit 4
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
    KAMTOF_DEPS=("mpirun" "git" "wget")
    
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
# Clone KAMTOF
# --------------------------------------------------------------------------------------------------
clone_kamtof_repo()
{
    # Set Git Repo path
    # Clone via HTTPS
    #KAMTOF_REPO="https://git-dev.converge.global/sidarth.narayanan/kamtof.git"

    # Clone via SSH (only for testing)
    KAMTOF_REPO="git@git-dev.converge.global:sidarth.narayanan/kamtof.git"
    
    # Set branch name
    BRANCH_NAME="task_021_install_script_serial"

    # Clone KAMTOF from repoository
    LOG_FILE="${LOG_DIR}/git_clone.log"
    echo -e "${INFO_PREFIX} Cloning Git repository for KAMTOF: ${KAMTOF_REPO}"
    git clone -b ${BRANCH_NAME} ${KAMTOF_REPO} > ${LOG_FILE} 2>&1

    return 0
}

# --------------------------------------------------------------------------------------------------
# Download KAMTOF using wget as a ZIP archive
# --------------------------------------------------------------------------------------------------
download_kamtof_zip()
{
    echo -e "${INFO_PREFIX} Downloading KAMTOF as ZIP archive."
    # Name of branch to be used
    BRANCH_NAME="task_021_install_script_serial"
    
    # Set Git Repo path
    KAMTOF_REPO="https://gitlab.com/sidarth.narayanan/kamtof"

    #Set path to KAMTOF archive
    ZIP_FILE_NAME="kamtof-${BRANCH_NAME}.zip"
    KAMTOF_ARCHIVE="${KAMTOF_REPO}/-/archive/${BRANCH_NAME}/${ZIP_FILE_NAME}"


    # Set name of log file
    LOG_FILE="${LOG_DIR}/wget_download.log"
    
    # Download archive using wget
    echo "wget ${KAMTOF_ARCHIVE} > ${LOG_FILE} 2>&1"

    # Unzip archive
    #unzip ${SRC_DIR}/${ZIP_FILE_NAME}


    safe_exit 5

    return 0
}

# --------------------------------------------------------------------------------------------------
# Compile KAMTOF
# --------------------------------------------------------------------------------------------------
compile_kamtof()
{
    # Build KAMTOF using CMAKE
    echo -e "${INFO_PREFIX} Building KAMTOF using CMAKE..."

    # Set names for configure, build and install log files
    CONFIGURE_LOG=${LOG_DIR}/build.log
    BUILD_LOG=${LOG_DIR}/build.log
    INSTALL_LOG=${LOG_DIR}/install.log

    # Configure  KAMTOF
    echo -ne "${INFO_SUBLEVEL_PREFIX} Configuring..."
    
    # Error message to be displayed in case of failure
    ERROR_MSG="\r${INFO_SUBLEVEL_PREFIX} Configure...${RED}${BOLD}FAILED${CLEAR}. Please check log file: ${CONFIGURE_LOG}"
    
    cmake ${SRC_DIR}/kamtof -DCMAKE_BUILD_TYPE=${BLD_TYPE} > ${CONFIGURE_LOG} 2>&1
    check_error_status ${PIPESTATUS[0]} "${ERROR_MSG}"
    
    # Success message
    echo -e "\r${INFO_SUBLEVEL_PREFIX} Configuring...${GREEN}successful${CLEAR}"

    #-----------------------------------------------------------------------------------------------

    # Build KAMTOF
    echo -e "${INFO_SUBLEVEL_PREFIX} Building..."
    
    # Error message to be displayed in case of failure
    ERROR_MSG="\r${INFO_SUBLEVEL_PREFIX} Building...${RED}${BOLD}FAILED${CLEAR}. \n${ERROR_PREFIX} Please check log file: ${BUILD_LOG}"
    
    # Error checking 
    make -j ${THREAD_COUNT} |& tee ${BUILD_LOG}
    check_error_status ${PIPESTATUS[0]} "${ERROR_MSG}"
    
    # Success messsage
    echo -e "${INFO_SUBLEVEL_PREFIX} Building...${GREEN}successful${CLEAR}"

    #-----------------------------------------------------------------------------------------------

    return 0
}

# --------------------------------------------------------------------------------------------------
# Load dependencies using modules
# --------------------------------------------------------------------------------------------------
load_dependencies_using_modules()
{
    # Load required modules for KAMTOF
    MODULE_LOG=${LOG_DIR}/module.log
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
# Check if a required env variable is defined
# --------------------------------------------------------------------------------------------------
require_env_var()
{
    local var_name="$1"
    if [ -z "${!var_name}" ]; then
        return 1
    fi
}

# --------------------------------------------------------------------------------------------------
# Setup environment for KAMTOF
# --------------------------------------------------------------------------------------------------
setup_environment()
{
    # Check all required environment variables
    REQ_VARS_LIST=("HPCX_ROOT" "CUDA_ROOT" "ROCM_ROOT" "ONEAPI_ROOT" "ONEMATH_ROOT")
    
    # Loop over all required env vars and check if defined
    for REQ_ENV_VAR in "${REQ_VARS_LIST[@]}"
    do
        # Error message if check fails
        local ERROR_MSG="Required environment variable \"${RED}${REQ_ENV_VAR}${CLEAR}\" is not set. Script cannot continue."
        
        require_env_var "${REQ_ENV_VAR}"
        check_error_status "-${PIPESTATUS[0]}" "${ERROR_MSG}"
    done

    ERROR_MSG="One or more required environment variables are not set. Please set these variables before continuing.\n"
    ERROR_MSG+="${ERROR_PREFIX} Run \"${BLUE}source export_root.sh${CLEAR}\" after setting required environment variables in ${BLUE}export_root.sh${CLEAR}." 
    check_error_status "${GLOBAL_ERROR_CODE}" "${ERROR_MSG}"


    source ${KAMTOF_ROOT}/scripts/env/oneAPI.sh
    source ${KAMTOF_ROOT}/scripts/env/hpcx.sh
    source ${KAMTOF_ROOT}/scripts/env/cuda.sh
    source ${KAMTOF_ROOT}/scripts/env/rocm.sh
    source ${KAMTOF_ROOT}/scripts/env/oneMath.sh

    export CC=icx
    export CXX=icpx

}

# --------------------------------------------------------------------------------------------------
# Setup environment for KAMTOF using script
# --------------------------------------------------------------------------------------------------
setup_environment_using_script()
{
    # Set the KAMTOF root directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export KAMTOF_ROOT="$(dirname "${SCRIPT_DIR}")"

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
# Set base directory containing the KAMTOF source code
# --------------------------------------------------------------------------------------------------
set_base_directory()
{
    # Check if KAMTOF needs to be cloned
    if [ ${CLONE_KAMTOF} != "FALSE" ]
    then
        # Check for existence of directory containing the source code for KAMTOF
        if [ -d ${SRC_DIR} ]
        then
            if [ ${DEL_SRC} == "FALSE" ]
            then
                echo -ne "${ERROR_PREFIX} ${RED} Source directory \"${SRC_DIR}\" already exists. Please choose a different source directory or "
                echo -e "use option \"-d TRUE\" to delete existing directory.${CLEAR}"
                safe_exit 2
            elif [ ${DEL_SRC} == "TRUE" ]
            then
                echo -e "${INFO_PREFIX} Deleting contents of ${SRC_DIR} for fresh install."
                rm -rf ${SRC_DIR}/
                mkdir -p ${SRC_DIR}
            fi
        else
            mkdir -p ${SRC_DIR}
        fi

        # Change directory to source directory
        echo -e "${INFO_PREFIX} Changing source directory to: \"${SRC_DIR}\"."
        cd ${SRC_DIR}

        echo -e ""
        
        # Clone KAMTOF repo
        clone_kamtof_repo
    else
        if [ -d ${SRC_DIR} ]
        then
            # Create directory to store log files
            mkdir -p ${LOG_DIR}
            
            # Change directory to source directory
            echo -e "${INFO_PREFIX} Building existing source for KAMTOF."
            echo -e "${INFO_PREFIX} Changing source directory to: \"${SRC_DIR}\"."
            cd ${SRC_DIR}
        else
            echo -e "${ERROR_PREFIX} ${RED}Directory for existing KAMTOF source (${SRC_DIR}) does not exist.${CLEAR}"
            echo -ne "${ERROR_PREFIX} ${RED}Please use the option \"-s\" to specify an existing source directory for KAMTOF "
            echo -e  "or use \"-g TRUE\" to clone the latest production branch for KAMTOF."
            safe_exit 3
        fi
    fi

    return 0
}

# --------------------------------------------------------------------------------------------------
# Set build directory and create it if needed
# --------------------------------------------------------------------------------------------------
create_build_directory()
{
    # Creating build directory for KAMTOF
    if [ -d ${BLD_TYPE} ]
    then
        rm -rf ${BLD_DIR}
    else
        mkdir -p ${BLD_DIR}
    fi

    # Create build directory and change directories
    echo -e "${INFO_PREFIX} Changing directory to build directory: \"${BLD_DIR}\"."
    echo -e ""
    cd ${BLD_DIR}

    return 0
}

# --------------------------------------------------------------------------------------------------
# Execute main function if script is not being sourced
# --------------------------------------------------------------------------------------------------
run_main()
{
    if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
        # Script is being executed (not sourced), so exit normally
        main $@

        # Ensure safe exit of function
        exit 0
    else
        # Get name of script
        SCRIPT_NAME=${BASH_SOURCE[0]}
        # Script is being sourced, so use 'return'
        echo -ne "${ERROR_PREFIX} Do not use \"source ${SCRIPT_NAME}\" to run this script."
        echo -e " Please use \"${BLUE}./${SCRIPT_NAME}${CLEAR}\"."
        return 1
    fi
}



# --------------------------------------------------------------------------------------------------
# Main script
# --------------------------------------------------------------------------------------------------
main()
{
    # Load dependencies
    load_dependencies_using_script
    #load_dependencies_using_modules
    
    # gather the command line options and pass them along to function
    gather_options $@

    # Input start-up information
    echo -e "${INFO_PREFIX} Starting script for KAMTOF installation..."

    # Perform check of OS and hardware
    system_check

    # Check for required dependencies
    check_all_dependencies

    # Create KAMTOF directory for compilation by cloning it from Github or use a pre-existing directory
    set_base_directory

    # Create build directory
    create_build_directory

    # Compile/build KAMTOF
    timer compile_kamtof "compilation of KAMTOF"

    # Print success message
    echo -e "${INFO_SUCCESS} ${GREEN}KAMTOF installation successful.${CLEAR}"

    safe_exit 0
}

# Run full script
run_main $@