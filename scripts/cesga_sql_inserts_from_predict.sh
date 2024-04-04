#!/bin/bash

# ---  CONSTANTS  --- #
# ------------------- #
OUTPUT_SCRIPT='/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts.sql'

# Model ID (from database) as environment variable
export MODEL_ID='3'

# Dataset names
TRAINING_JSON=(
    'RGBIr_MERGE_147_minmaxnorm_VEGETATION'
    'RGBIr_MERGE_16_minmaxnorm_VEGETATION'
    'RGBIr_MERGE_235_minmaxnorm_VEGETATION'
    'RGBIr_MERGE_276_minmaxnorm_VEGETATION'
)

# Paths to directories with the predictions
TRAINING_DIR=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X/T1/preds/MERGE_147'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X/T1/preds/MERGE_16'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X/T1/preds/MERGE_235'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X/T1/preds/MERGE_276'
)


# ---   M A I N   --- #
# ------------------- #
# Remove previous script, if any
rm -f ${OUTPUT_SCRIPT}

# Loop over training processes and merge into new script file
for (( i=0 ; i < ${#TRAINING_JSON[@]} ; ++i )); do
    python3 sql_insert_from_experiment.py "${TRAINING_DIR[$i]}" "${TRAINING_JSON[$i]}" >> ${OUTPUT_SCRIPT}
done

# Zip the SQL script
zip -v9j $(sed 's/\.sql/\.zip/g' <<< ${OUTPUT_SCRIPT}) ${OUTPUT_SCRIPT}

