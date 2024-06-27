#!/bin/bash

# ---  CONSTANTS  --- #
# ------------------- #
OUTPUT_SCRIPT='/home/usc/ci/myg/store2/Results/galicia_vl3d/ddbb/building/cesga_vl3dgal_predict_inserts.sql'

EXPERIMENT_NAME=BUILDING

# Create output dir if it does not exist
mkdir -p $(dirname ${OUTPUT_SCRIPT})

# Model ID (from database) as environment variable
export MODEL_ID='24'

# Paths to directories with the predictions (folder containing MERGE_* directories)
PREDICTIONS_DIR=/home/usc/ci/myg/store2/Results/galicia_vl3d/building

PREDICTIONS_DIR=($(find ${PREDICTIONS_DIR} -maxdepth 1 -mindepth 1 -type d))


# create array with basenames of the predictions
for (( i=0 ; i < ${#PREDICTIONS_DIR[@]} ; ++i )); do
    DATASET_NAMES[$i]=RGBIr_$(basename ${PREDICTIONS_DIR[$i]})_minmaxnorm_$EXPERIMENT_NAME
done

# ---   M A I N   --- #
# ------------------- #

# Loop over training processes and merge into new script file
for (( i=0 ; i < ${#DATASET_NAMES[@]} ; ++i )); do
    python3 sql_insert_from_experiment.py "${PREDICTIONS_DIR[$i]}" "${DATASET_NAMES[$i]}" >> $(sed 's/\.sql//g' <<< ${OUTPUT_SCRIPT})_$(basename ${PREDICTIONS_DIR[$i]})_model${MODEL_ID}_$EXPERIMENT_NAME.sql
done

# # Zip the SQL script
# zip -v9j $(sed 's/\.sql/\.zip/g' <<< ${OUTPUT_SCRIPT}_$(basename ${PREDICTIONS_DIR[$i]})_model${MODEL_ID}.sql) ${OUTPUT_SCRIPT}_$(basename ${PREDICTIONS_DIR[$i]})_model${MODEL_ID}.sql

