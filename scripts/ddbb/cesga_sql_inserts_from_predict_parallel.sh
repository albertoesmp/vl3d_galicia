#!/bin/bash

# ---  CONSTANTS  --- #
# ------------------- #
OUTPUT_SCRIPT='/mnt/netapp1/Store_usccisss/ddbb/cesga_vl3dgal_predict_inserts.sql'

EXPERIMENT_NAME=VEGETATION

# Create output dir if it does not exist
mkdir -p $(dirname ${OUTPUT_SCRIPT})

# Model ID (from database) as environment variable
export MODEL_ID='4'

# Paths to directories with the predictions (folder containing MERGE_* directories)
PREDICTIONS_DIR=/mnt/netapp1/Store_usccisss/results/vl3d_galicia/vegetation

PREDICTIONS_DIR=($(find ${PREDICTIONS_DIR} -maxdepth 1 -mindepth 1 -type d))


# create array with basenames of the predictions
for (( i=0 ; i < ${#PREDICTIONS_DIR[@]} ; ++i )); do
    DATASET_NAMES[$i]=RGBIr_$(basename ${PREDICTIONS_DIR[$i]})_minmaxnorm_$EXPERIMENT_NAME
done

# ---   M A I N   --- #
# ------------------- #

# Function to run in parallel
run_sql_insert() {
    local pred_dir="$1"
    local dataset_name="$2"
    python3 sql_insert_from_experiment.py "${pred_dir}" "${dataset_name}" >> $(sed 's/\.sql//g' <<< ${OUTPUT_SCRIPT})_$(basename ${pred_dir})_model${MODEL_ID}_$EXPERIMENT_NAME.sql
}

# Automatically get the number of CPU cores
max_jobs=$(nproc)
echo "Number of CPU cores available: $max_jobs"

# Run processes in parallel
for (( i=0 ; i < ${#DATASET_NAMES[@]} ; ++i )); do
    run_sql_insert "${PREDICTIONS_DIR[$i]}" "${DATASET_NAMES[$i]}" &
    # Limit the number of parallel jobs
    if (( $(jobs -r -p | wc -l) >= max_jobs )); then
        wait -n
    fi
done

# Wait for all background jobs to complete
wait

# # Zip the SQL script
# zip -v9j $(sed 's/\.sql/\.zip/g' <<< ${OUTPUT_SCRIPT}_$(basename ${PREDICTIONS_DIR[$i]})_model${MODEL_ID}.sql) ${OUTPUT_SCRIPT}_$(basename ${PREDICTIONS_DIR[$i]})_model${MODEL_ID}.sql
