#!/bin/bash

# ---  CONSTANTS  --- #
# ------------------- #
OUTPUT_SCRIPT='/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_training_inserts.sql'

# Paths to training JSONs
TRAINING_JSON=(
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_X_training_T1.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_X_training_T2.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_X_training_T3.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_X_training_T4.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_X_training_T5.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_XIr_training_T1.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_XIr_training_T2.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_XIr_training_T3.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_XIr_training_T4.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_XIr_training_T5.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_XRGB_training_T1.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_XRGB_training_T2.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_XRGB_training_T3.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_XRGB_training_T4.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_XRGB_training_T5.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_XIrRGB_training_T1.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_XIrRGB_training_T2.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_XIrRGB_training_T3.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_XIrRGB_training_T4.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/vegetation/kpconv_final_XIrRGB_training_T5.json'
)

# Paths to directories with the output of the training processes
TRAINING_DIR=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB/T5/'
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

