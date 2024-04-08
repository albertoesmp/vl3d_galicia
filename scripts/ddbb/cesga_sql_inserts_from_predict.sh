#!/bin/bash

# ---  CONSTANTS  --- #
# ------------------- #
OUTPUT_SCRIPT='/home/usc/ci/myg/store2/galicia_vl3d/results/ddbb/cesga_vl3dgal_predict_inserts.sql'

# Path where the results (MERGE_1, MERGE_234, etc) are stored
RESULTS_PATH=$STORE2/vl3d/models/kpc_final_X/T2/preds

EXPERIMENT_NAME=VEGETATION

# Create output dir if it does not exist
mkdir -p $(dirname ${OUTPUT_SCRIPT})

# Model ID (from database) as environment variable
export MODEL_ID='3'

# Paths to directories with the predictions
PREDICTIONS_DIR=(
    "$RESULTS_PATH/MERGE_276"
    "$RESULTS_PATH/MERGE_147"
    "$RESULTS_PATH/MERGE_235"
    "$RESULTS_PATH/MERGE_16"
)

# create array with basenames of the predictions
for (( i=0 ; i < ${#PREDICTIONS_DIR[@]} ; ++i )); do
    DATASET_NAMES[$i]=RGBIr_$(basename ${PREDICTIONS_DIR[$i]})_minmaxnorm_$EXPERIMENT_NAME
done

# ---   M A I N   --- #
# ------------------- #
# Remove previous script, if any
rm -f ${OUTPUT_SCRIPT}

# Loop over training processes and merge into new script file
for (( i=0 ; i < ${#DATASET_NAMES[@]} ; ++i )); do
    python3 sql_insert_from_experiment.py "${PREDICTIONS_DIR[$i]}" "${DATASET_NAMES[$i]}" >> $(sed 's/\.sql//g' <<< ${OUTPUT_SCRIPT})_$(basename ${PREDICTIONS_DIR[$i]})_model${MODEL_ID}_$EXPERIMENT_NAME.sql
done

# # Zip the SQL script
# zip -v9j $(sed 's/\.sql/\.zip/g' <<< ${OUTPUT_SCRIPT}_$(basename ${PREDICTIONS_DIR[$i]})_model${MODEL_ID}.sql) ${OUTPUT_SCRIPT}_$(basename ${PREDICTIONS_DIR[$i]})_model${MODEL_ID}.sql

