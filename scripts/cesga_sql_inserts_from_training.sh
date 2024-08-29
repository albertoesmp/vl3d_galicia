#!/bin/bash

# ---  CONSTANTS  --- #
# ------------------- #
OUTPUT_SCRIPT='/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_training_pnet_buildveg_inserts.sql'

# Paths to training JSONs
TRAINING_JSON=(
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_X_buildveg_training_T1_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_X_buildveg_training_T2_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_X_buildveg_training_T3_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_X_buildveg_training_T4_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_X_buildveg_training_T5_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_XIr_buildveg_training_T1_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_XIr_buildveg_training_T2_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_XIr_buildveg_training_T3_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_XIr_buildveg_training_T4_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_XIr_buildveg_training_T5_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_XRGB_buildveg_training_T1_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_XRGB_buildveg_training_T2_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_XRGB_buildveg_training_T3_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_XRGB_buildveg_training_T4_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_XRGB_buildveg_training_T5_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_XIrRGB_buildveg_training_T1_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_XIrRGB_buildveg_training_T2_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_XIrRGB_buildveg_training_T3_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_XIrRGB_buildveg_training_T4_alt.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/buildveg/pnet_final_XIrRGB_buildveg_training_T5_alt.json'
)

# Paths to directories with the output of the training processes
TRAINING_DIR=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_X_buildveg_alt/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_buildveg_alt/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XRGB_buildveg_alt/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIrRGB_buildveg_alt/T5/'
)



# ---   M A I N   --- #
# ------------------- #
# Remove previous script, if any
rm -f ${OUTPUT_SCRIPT}

# Loop over training processes and merge into new script file
for (( i=0 ; i < ${#TRAINING_JSON[@]} ; ++i )); do
    python3 sql_insert_from_training.py ${TRAINING_JSON[$i]} ${TRAINING_DIR[$i]} >> ${OUTPUT_SCRIPT}   
done

# Zip the SQL script
zip -v9j $(sed 's/\.sql/\.zip/g' <<< ${OUTPUT_SCRIPT}) ${OUTPUT_SCRIPT}

