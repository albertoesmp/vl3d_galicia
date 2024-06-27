#!/bin/bash

# ---  CONSTANTS  --- #
# ------------------- #
OUTPUT_SCRIPT='/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_predict_inserts.sql'

# Model ID (from database) as environment variable
export MODEL_ID='100'

# Dataset names
TRAINING_JSON=(
    'RGBIr_MERGE_189_minmaxnorm_LMH_VEGETATION'
    'RGBIr_MERGE_234_minmaxnorm_LMH_VEGETATION'
    'RGBIr_MERGE_239_minmaxnorm_LMH_VEGETATION'
    'RGBIr_MERGE_97_minmaxnorm_LMH_VEGETATION'
)

# Paths to directories with the predictions
TRAINING_DIR=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_lmhveg/T4/preds/MERGE_189'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_lmhveg/T4/preds/MERGE_234'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_lmhveg/T4/preds/MERGE_239'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_lmhveg/T4/preds/MERGE_97'
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

