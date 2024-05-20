#!/bin/bash

# Set to your own paths
VL3D_ENV=/home/usc/ci/myg/.conda/envs/vl3d
OUTPUT_PATH=/mnt/netapp1/Store_usccimyg/Results/galicia_vl3d/vegetation

# Set your email
EMAIL=miguel.yermo@usc.es

# Do not touch
VL3D_DIR=$(realpath $(dirname $0)/../)
VL3D_SCRIPTS_DIR=$VL3D_DIR/vl3d.py
MODEL=/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X/T3/pipe
EXPERIMENT_NAME=building
JSON_DIR=$VL3D_DIR/experiments/$EXPERIMENT_NAME/$(whoami)

# Replace e-mail in work.sh
sed -i "s/--mail-user=.*/--mail-user=$EMAIL/" $(realpath $(dirname $0)/slurm/work.sh)

# Replace paths in replace_paths.sh
sed -i 's|sed_model|'$MODEL'|g' $VL3D_DIR/scripts/experiment_generation/replace_paths.sh
sed -i 's|sed_experiment_name|'$EXPERIMENT_NAME'|g' $VL3D_DIR/scripts/experiment_generation/replace_paths.sh
sed -i 's|sed_output_path|'$OUTPUT_PATH'|g' $VL3D_DIR/scripts/experiment_generation/replace_paths.sh
sed -i 's|sed_json_dir|'$JSON_DIR'|g' $VL3D_DIR/scripts/experiment_generation/replace_paths.sh



$VL3D_DIR/scripts/experiment_generation/replace_paths.sh
$VL3D_DIR/scripts/slurm/launch.sh

