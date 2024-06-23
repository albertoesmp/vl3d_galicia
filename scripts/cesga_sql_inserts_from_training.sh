#!/bin/bash

# ---  CONSTANTS  --- #
# ------------------- #
OUTPUT_SCRIPT='/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/ddbb/cesga_vl3dgal_training_building_inserts.sql'

# Paths to training JSONs
TRAINING_JSON=(
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_X_building_training_T1.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_X_building_training_T2.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_X_building_training_T3.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_X_building_training_T4.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_X_building_training_T5.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_XIr_building_training_T1.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_XIr_building_training_T2.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_XIr_building_training_T3.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_XIr_building_training_T4.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_XIr_building_training_T5.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_XIrRGB_building_training_T1.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_XIrRGB_building_training_T2.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_XIrRGB_building_training_T3.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_XIrRGB_building_training_T4.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_XIrRGB_building_training_T5.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_XRGB_building_training_T1.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_XRGB_building_training_T2.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_XRGB_building_training_T3.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_XRGB_building_training_T4.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/kpconv_final_XRGB_building_training_T5.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_X_building_training_T1.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_X_building_training_T2.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_X_building_training_T3.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_X_building_training_T4.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_X_building_training_T5.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_XIr_building_training_T1.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_XIr_building_training_T2.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_XIr_building_training_T3.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_XIr_building_training_T4.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_XIr_building_training_T5.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_XIrRGB_building_training_T1.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_XIrRGB_building_training_T2.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_XIrRGB_building_training_T3.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_XIrRGB_building_training_T4.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_XIrRGB_building_training_T5.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_XRGB_building_training_T1.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_XRGB_building_training_T2.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_XRGB_building_training_T3.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_XRGB_building_training_T4.json'
    '/home/usc/ci/aep/git/virtualearn3d/cesga/galicia/building/pnetpp_final_XRGB_building_training_T5.json'
)

# Paths to directories with the output of the training processes
TRAINING_DIR=(
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_building/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_building/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_building/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_building/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_building/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_building/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_building/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_building/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_building/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIr_building/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_building/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_building/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_building/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_building/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XRGB_building/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_building/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_building/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_building/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_building/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_building/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_building/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_building/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_building/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_building/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_X_building/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_building/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_building/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_building/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_building/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIr_building/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_building/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_building/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_building/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_building/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XIrRGB_building/T5/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_building/T1/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_building/T2/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_building/T3/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_building/T4/'
    '/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnetpp_final_XRGB_building/T5/'
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

