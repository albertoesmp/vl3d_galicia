#!/bin/bash

# ---  CONSTANTS  --- #
# ------------------- #
OUTPUT_SCRIPT='/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_training_rf_fullraw_inserts.sql'

# Paths to training JSONs
TRAINING_JSON=(
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/rf/rf_XIrRGB_fullraw_try5_training.json'
)

# Paths to directories with the output of the training processes
TRAINING_DIR=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/rf_XIrRGB_fullraw_try5/'
)



# ---   M A I N   --- #
# ------------------- #
# Remove previous script, if any
rm -f ${OUTPUT_SCRIPT}

# Loop over training processes and merge into new script file
for (( i=0 ; i < ${#TRAINING_JSON[@]} ; ++i )); do
    python3 sql_insert_from_training_rf.py ${TRAINING_JSON[$i]} ${TRAINING_DIR[$i]} >> ${OUTPUT_SCRIPT}   
done

# Zip the SQL script
zip -v9j $(sed 's/\.sql/\.zip/g' <<< ${OUTPUT_SCRIPT}) ${OUTPUT_SCRIPT}

