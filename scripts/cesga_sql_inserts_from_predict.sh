#!/bin/bash

# ---  CONSTANTS  --- #
# ------------------- #
OUTPUT_SCRIPT='/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts.sql'

# Model ID (from database) as environment variable
export MODEL_ID='61'

# Dataset names
TRAINING_JSON=(
    'RGBIr_MERGE_260_minmaxnorm_BUILDING'
    'RGBIr_MERGE_68_minmaxnorm_BUILDING'
    'RGBIr_MERGE_216_minmaxnorm_BUILDING'
    'RGBIr_MERGE_21_minmaxnorm_BUILDING'
)

# Paths to directories with the predictions
TRAINING_DIR=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_building/T5/preds/MERGE_260'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_building/T5/preds/MERGE_68'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_building/T5/preds/MERGE_216'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_building/T5/preds/MERGE_21'
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

