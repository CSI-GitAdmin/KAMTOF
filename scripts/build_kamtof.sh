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
    local EXITc_CODE=$1
    if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
        # Script is being executed (not sourced), so exit normally
        exit "${EXIT_CODE}"
    else
        # Script is being sourced, so use 'return'
        return "${EXIT_CODE}"
    fi
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
# Main function to be run
# --------------------------------------------------------------------------------------------------
main()
{
    # Set the KAMTOF root directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export KAMTOF_ROOT="$(dirname "${SCRIPT_DIR}")"
 
    BUILD_LOG="${KAMTOF_ROOT}/scripts/release.log"

    # Error message to be displayed in case of failure
    ERROR_MSG="Debug mode build failed. Please check logfile ${BUILD_LOG}"

    echo -ne "${INFO_PREFIX} Building in release mode..."

    # Compile in RELEASE mode
    ${KAMTOF_ROOT}/scripts/build.sh -b build_release > ${BUILD_LOG} 2>&1
    check_error_status ${PIPESTATUS[0]} "${ERROR_MSG}"

    echo -e "\r${INFO_PREFIX} Building in release mode...${GREEN}successful${CLEAR}"

    tail -n 3 ${BUILD_LOG} | head -n 1

    echo ""

    # --------------------------------------------------------------------------------------------------
    # DEBUG MODE BUILD

    echo -ne "${INFO_PREFIX} Building in debug mode..."

    BUILD_LOG="${KAMTOF_ROOT}/scripts/debug.log"

    # Error message to be displayed in case of failure
    ERROR_MSG="Debug mode build failed. Please check logfile ${BUILD_LOG}"

    # Compile in DEBUG mode
    ${KAMTOF_ROOT}/scripts/build.sh -b build_debug -t DEBUG -j 8 -g FALSE > ${BUILD_LOG} 2>&1
    check_error_status ${PIPESTATUS[0]} "${ERROR_MSG}"

    echo -e "\r${INFO_PREFIX} Building in debug mode...${GREEN}successful${CLEAR}"

    tail -n 3 ${BUILD_LOG} | head -n 1

    echo ""

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

# Run full script
run_main $@