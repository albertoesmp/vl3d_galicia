#!/bin/bash

# -------------------------------------------------------------------------
# AUTHOR : Alberto M. Esmoris Pena
# BRIEF  : Script to transform the VL3D++ models to use exhaustive FPS
#           to compute the support points instead of an approximation.
# -------------------------------------------------------------------------

# Path to auxiliar python script to generate preparation JSONs
PYPREPARE='prepare.py'

# Trainign JSONs
TRAINING_JSON=(
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_X_building_training_T1.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_X_building_training_T2.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_X_building_training_T3.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_X_building_training_T4.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_X_building_training_T5.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_XIr_building_training_T1.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_XIr_building_training_T2.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_XIr_building_training_T3.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_XIr_building_training_T4.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_XIr_building_training_T5.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_XIrRGB_building_training_T1.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_XIrRGB_building_training_T2.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_XIrRGB_building_training_T3.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_XIrRGB_building_training_T4.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_XIrRGB_building_training_T5.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_XRGB_building_training_T1.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_XRGB_building_training_T2.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_XRGB_building_training_T3.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_XRGB_building_training_T4.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/building/kpconv/kpconv_final_XRGB_building_training_T5.json"
)


for (( i=0 ; i < ${#TRAINING_JSON[@]} ; ++i )); do
    jsonf="${TRAINING_JSON[$i]}"
    python "${PYPREPARE}" "${jsonf}"
    prepf=$(sed 's/training/prepare/g' <<< $jsonf)
    echo "RUNNING ${prepf} ..."
    python "${VL3D_SCRIPT}" --pipeline "${prepf}"
    rm -f $prepf
done
