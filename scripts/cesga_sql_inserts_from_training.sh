#!/bin/bash

# ---  CONSTANTS  --- #
# ------------------- #
OUTPUT_SCRIPT='/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_training_sflnet_fullraw_inserts.sql'

# Paths to training JSONs
TRAINING_JSON=(
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_X_fullraw_training_T1.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_X_fullraw_training_T2.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_X_fullraw_training_T3.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_X_fullraw_training_T4.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_X_fullraw_training_T5.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_XIr_fullraw_training_T1.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_XIr_fullraw_training_T2.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_XIr_fullraw_training_T3.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_XIr_fullraw_training_T4.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_XIr_fullraw_training_T5.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_XRGB_fullraw_training_T1.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_XRGB_fullraw_training_T2.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_XRGB_fullraw_training_T3.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_XRGB_fullraw_training_T4.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_XRGB_fullraw_training_T5.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_XIrRGB_fullraw_training_T1.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_XIrRGB_fullraw_training_T2.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_XIrRGB_fullraw_training_T3.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_XIrRGB_fullraw_training_T4.json'
    '/home/usc/ci/aep/git/vl3dpp/cesga/galicia/fullraw/try/sflnet_try1_XIrRGB_fullraw_training_T5.json'
)

# Paths to directories with the output of the training processes
TRAINING_DIR=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_X_fullraw/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIr_fullraw/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XRGB_fullraw/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_fullraw/T5/'
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

