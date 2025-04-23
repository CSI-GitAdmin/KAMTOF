#! /bin/bash

EXEC_NAME=solver
TEMP_FILE="laplace_template.in"

echo "[I] Executing script to perform order of accuracy study."

# Check for existence of file
if [ ! -f "${TEMP_FILE}" ]; then
    echo "[E]: File ${TEMP_FILE} not found."
    exit 1
fi

# List the grid sizes to be used for the convergence study
grid_sizes=(11 21 41)

# Text to be replaced
SEARCH_TEXT="NGRID"

# get number of grids to be run
NRUNS=${#grid_sizes[@]}

# Create run directory
RUN_DIR=${1}

# Check existence of directory
if [ -d ${RUN_DIR} ];
then
    echo "[E] Directory \"${RUN_DIR}\" exists. Please rename/delete before continuing."
    exit 2
else
    mkdir -p ${RUN_DIR}
fi

GRID_LIST=""

echo "[I] Starting loop..."

#loop through each grid size
for ((irun=1; irun<=NRUNS; irun++))
do    
    # Get local value of grid size for this particular case
    nx=${grid_sizes[$((irun-1))]}

    # Create a variable that is a comma-delimited version of the grid size variable
    # This will be passed onto the Python script later for order of accuracy calculations
    GRID_LIST+="${nx},"

    # Print information to screen
    echo "--> Running case ${irun}/${NRUNS} with Nx = Ny = ${nx}"

    # Create copy of template file
    OUT_FILE="laplace_nx_${nx}.in"
    cp -r ${TEMP_FILE} ${OUT_FILE}

    # Replace search string with the actual value
    sed -i "s/${SEARCH_TEXT}/${nx}/g" "${OUT_FILE}"

    # Execute solver
    LOG_FILE="run_nx_${nx}.log"
    ./solver ${OUT_FILE} > ${LOG_FILE} 2>&1

    # Create directory for given case in the result directory
    CASE_DIR="${RUN_DIR}/nx_${nx}"
    mkdir -p ${CASE_DIR}

    # store file names for solution and residual files
    SOL_FILE="laplace_solution_cpu_nx_${nx}_ny_${nx}.txt"
    RES_FILE="laplace_residual_cpu_nx_${nx}_ny_${nx}.txt"

    #move output and log files to output directory
    mv ${LOG_FILE} ${SOL_FILE} ${RES_FILE} ${OUT_FILE} ${CASE_DIR}

done

# Execute python script to analyze order of accuracy
echo ""
echo "[I] Running python script to compute order of accuray using cases located in \"${RUN_DIR}\":"

# Set name of Python log file
PYTHON_LOG="${RUN_DIR}/order.log"

# Create comma-delimited list of grid sizes to be passed on to Python
GRID_LIST="${GRID_LIST%?}"

python3 analytical_laplace_solution.py ${GRID_LIST} ${RUN_DIR} 2>&1 | tee ${PYTHON_LOG} 

# Print success message
echo ""
echo "[I] Order of accuracy script completed successfully."

# Exit script with success exit code
exit 0
