#! /bin/bash

# Color markup for messages
BOLD='\033[1m' 
CLEAR='\033[0m'
UNDERLINE='\e[4m'
RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
ERROR_COLOR="${RED}${BOLD}"
INFO_PREFIX="${BLUE}${BOLD}[I]${CLEAR}"
INFO_SUCCESS="${GREEN}${BOLD}[I]${CLEAR}"
WARNING_PREFIX="${BOLD}[W]${CLEAR}"

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
    Cygwin)          UNAME_OS="cygwin" ;; 
    Darwin)          UNAME_OS="mac" ;; 
    *) echo -e $RED"[E]$CLEAR: unknown operating system type"
        exit 1 ;;
    esac

    # Get hardware information
    case `uname -m` in
    x86_64)          UNAME_HW="x64"   ;;
    aarch64)         UNAME_HW="arm64" ;;
    *) echo -e $RED"[E]$CLEAR: unknown processor (hardware) type"
        exit 1 ;;
    esac

    read -r -d '' system_text << END

    ${BOLD}[I] Current system configuration:${CLEAR}
        Operating System  : ${BOLD}${UNAME_OS}${CLEAR}
        Hardware/Processor: ${BOLD}${UNAME_HW}${CLEAR} 
    ${CLEAR}
END
    
    echo -e "${system_text}"

    # Check operating system
    if [ ${UNAME_OS} != "linux" ]
    then
        echo -e "${RED}${BOLD}[ERROR]${CLEAR} This script can only be executed on GNU/Linux systems. You have a ${SYS_TYPE} based system."
        exit 1
    fi

    # Check hardware
    if [ ${UNAME_HW} != "x64" ]
    then
        echo -e "${RED}${BOLD}[ERROR]${CLEAR} ${RED}This script can only be executed on x86_64 hardware. You have a ${HW_TYPE} based hardware."
        exit 2
    fi
}

# --------------------------------------------------------------------------------------------------
# Set help information
# --------------------------------------------------------------------------------------------------
usage() 
{
    SUPPORTED_BLD_TYPES="Release,Debug"
    read -r -d '' usage_text << END

${BOLD}  $(basename $0)${CLEAR} [OPTIONS]
    This script clones and builds the current release version of the KAMTOF solver code.
    Repository: 

    All options have defaults and are optional. The default folders assume the 
    script is run from the source folder. The default build folder(s) and prefix 
    folder are at the same level as the source folder. 
${CLEAR}
    ${UNDERLINE}OPTIONS:${CLEAR}
    ${BOLD}-s <source_directory>${CLEAR} default:${BOLD} <current folder>${CLEAR}
                Directory where the KAMTOF code will be cloned.
                This path is relative to the current directory.
    ${BOLD}-b <build_directory>${CLEAR}  default:${BOLD} <source_directory>/build ${CLEAR}
                Directory where the KAMTOF code will be cloned and built (relative to source directory).
                This directory will exist inside the source directory.
                The executable will also be placed in this directory.
    ${BOLD}-t <build_type>${CLEAR}    default:${BOLD} Release${CLEAR}
                This system supports (case insensitive):
                ${BOLD} ${SUPPORTED_BLD_TYPE_LIST[@]}${CLEAR}
    ${BOLD}-j <num_proc>${CLEAR}      default:${BOLD} `nproc --all`${CLEAR}
                Number of processors to be used for compilation.
                The default is the maximum number of processors on this machine
                (computed using the 'nproc --all' command)
    ${BOLD}-g <clone_kamtof>${CLEAR}      default:${BOLD} FALSE${CLEAR}
                Clone KAMTOF from the Git repository.
                The default is FALSE.
    ${BOLD}-c${CLEAR}          Build with a clean build folder
    ${BOLD}-D${CLEAR}          Dryrun only. Only echoes the cmake commands
    ${BOLD}-h${CLEAR}          Print this help text 
${CLEAR}
END
echo -e "$usage_text"
}

# Print help information
#usage

# --------------------------------------------------------------------------------------------------
# Capture command line options
# --------------------------------------------------------------------------------------------------
# Set defaults
CURR_DIR=${PWD}
CLEAN_BUILD_FLAG=0
DRYRUN=0
BLD_TYPE="RELEASE"
SRC_DIR=${CURR_DIR}
BLD_DIR="build"
THREAD_COUNT=$(nproc --all)
CLONE_KAMTOF=FALSE
LOG_DIR=${SRC_DIR}/logs

gather_options() 
{
  while getopts "cDh-:b:s:t:j:g:" OPTION; do
    case ${OPTION} in
      c     ) CLEAN_BUILD_FLAG=${OPTARG}   ;;
      D     ) DRYRUN=${OPTARG}             ;;
      t     ) BLD_TYPE=${OPTARG}           ;;
      s     ) SRC_DIR=${CURR_DIR}/${OPTARG};;
      b     ) BLD_DIR=${OPTARG};;
      j     ) THREAD_COUNT=${OPTARG}       ;;
      g     ) CLONE_KAMTOF=${OPTARG}       ;;
      h     ) usage
              exit 0;;
      -     )
              echo -e ${RED}"[ERROR]${CLEAR} '--' (double-dash) options are not currently available"
              echo -e "       use -h option for usage information"
              exit 1                      ;;
      :|?|* )
              echo -e ${RED}"[ERROR]:${CLEAR} unknown option ${OPTION}"
              echo -e "       use -h option for usage information"
              exit 1                      ;;
    esac
  done

  #Set full path of build directory
  BLD_DIR=${SRC_DIR}/${BLD_DIR}
  LOG_DIR=${SRC_DIR}/logs
}

# --------------------------------------------------------------------------------------------------
# Check dependencies
# --------------------------------------------------------------------------------------------------
check_dependencies()
{
    GIT_VERSION=$(git --version)
}


# --------------------------------------------------------------------------------------------------
# Clone KAMTOF
# --------------------------------------------------------------------------------------------------
clone_kamtof_repo()
{
    # Set Git Repo path
    KAMTOF_REPO="https://git-dev.converge.global/sidarth.narayanan/kamtof.git"

    # Clone KAMTOF from repoository
    LOG_FILE="${LOG_DIR}/git_clone.log"
    echo -e "${INFO_PREFIX} Cloning Git repository for KAMTOF: ${KAMTOF_REPO}"
    git clone ${KAMTOF_REPO} > ${LOG_FILE} 2>&1
}

# --------------------------------------------------------------------------------------------------
# Begin MAIN script
# --------------------------------------------------------------------------------------------------
# Input start-up information
echo -e "${INFO_PREFIX} Starting script for KAMTOF installation..."

# Perform check of OS and hardware
system_check

# Check for required dependencies
check_dependencies

# gather the command line options, but sure to pass the arguments along
gather_options $@

# Check if KAMTOF needs to be cloned
if [ ${CLONE_KAMTOF} != "FALSE" ]
then
    # Check for existence of directory containing the source code for KAMTOF
    if [ -d ${SRC_DIR} ]
    then
        echo -e "${ERROR_COLOR}[ERROR]${CLEAR} ${RED} Directory: \"${SRC_DIR}\" already exists. Please choose a different source directory.${CLEAR}"
        exit 2
    else
        mkdir -p ${SRC_DIR}
    fi

    # Change directory to source directory
    echo -e "${INFO_PREFIX} Changing source directory to: \"${SRC_DIR}\"."
    cd ${SRC_DIR}

    # Create directory to store log files
    mkdir -p ${LOG_DIR}

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
        echo -e "${ERROR_COLOR}[ERROR]${CLEAR} ${RED}Directory for existing KAMTOF source (${SRC_DIR}) does not exist.${CLEAR}"
        exit 3
    fi
fi

# Creating build directory for KAMTOF
if [ -d ${BLD_TYPE} ]
then
    rm -rf ${BLD_DIR}
else
    mkdir -p ${BLD_DIR}
fi

# Create build directory and change directories
echo -e "${INFO_PREFIX} Changing directory to build directory: \"${BLD_DIR}\"."
cd ${BLD_DIR}

# Load required modules for KAMTOF
MODULE_LOG=${LOG_DIR}/module.log
echo -e "${INFO_PREFIX} Loading dependencies using modules for KAMTOF."
module purge
module load kamtof > ${MODULE_LOG} 2>&1

# Build KAMTOF using CMAKE
echo -e "${INFO_PREFIX} Building KAMTOF using CMAKE..."

# Set names for build and install log files
BUILD_LOG=${LOG_DIR}/build.log
INSTALL_LOG=${LOG_DIR}/install.log

# Build  KAMTOF
cmake ${SRC_DIR}/kamtof -DCMAKE_BUILD_TYPE=${BLD_TYPE} > ${BUILD_LOG} 2>&1

# Install KAMTOF
make -j ${THREAD_COUNT} > ${INSTALL_LOG} 2>&1

# Print success message
echo -e "${INFO_SUCCESS} KAMTOF installation successful."