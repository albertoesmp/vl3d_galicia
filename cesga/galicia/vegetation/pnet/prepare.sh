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
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_XIrRGB_vegetation_training_T1_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_XIrRGB_vegetation_training_T2_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_XIrRGB_vegetation_training_T3_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_XIrRGB_vegetation_training_T4_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_XIrRGB_vegetation_training_T5_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_XIr_vegetation_training_T1_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_XIr_vegetation_training_T2_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_XIr_vegetation_training_T3_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_XIr_vegetation_training_T4_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_XIr_vegetation_training_T5_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_XRGB_vegetation_training_T1_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_XRGB_vegetation_training_T2_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_XRGB_vegetation_training_T3_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_XRGB_vegetation_training_T4_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_XRGB_vegetation_training_T5_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_X_vegetation_training_T1_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_X_vegetation_training_T2_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_X_vegetation_training_T3_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_X_vegetation_training_T4_alt.json"
    "/home/usc/ci/aep/git/vl3dgal/cesga/galicia/fast_parallel/vegetation/pnet/pnet_final_X_vegetation_training_T5_alt.json"
)


for (( i=0 ; i < ${#TRAINING_JSON[@]} ; ++i )); do
    jsonf="${TRAINING_JSON[$i]}"
    python "${PYPREPARE}" "${jsonf}"
    prepf=$(sed 's/training/prepare/g' <<< $jsonf)
    echo "RUNNING ${prepf} ..."
    python "${VL3D_SCRIPT}" --pipeline "${prepf}"
    rm -f $prepf
done
