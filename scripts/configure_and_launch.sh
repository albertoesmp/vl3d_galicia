#!/bin/bash

# Do not touch
VL3D_DIR=$(realpath $(dirname $0)/../)
VL3D_SCRIPT_DIR=$VL3D_DIR/vl3d.py
EXPERIMENT_NAME=building
MODEL=/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_X_building/T3/pipe
JSON_DIR=$VL3D_DIR/experiments/$EXPERIMENT_NAME/$(whoami)

case $(whoami) in
    usccimyg)
    echo "Miguel"
    EMAIL=miguel.yermo@usc.es
    OUTPUT_PATH=/mnt/netapp1/Store_usccimyg/Results/galicia_vl3d/$EXPERIMENT_NAME
    VL3D_ENV=/home/usc/ci/myg/.conda/envs/vl3d
    ;;

    usccisss)
    echo "Samuel"
    EMAIL=s.soutullo@usc.es
    OUTPUT_PATH=
    VL3D_ENV=
    ;;

    usccisra)
    echo "Silvia"
    EMAIL=silvia.alcaraz@usc.es
    OUTPUT_PATH=
    VL3D_ENV=
    ;;

    *)
    echo "Quen carallo es ti?"
    ;;
esac

echo Conda env: $VL3D_ENV
echo e-mail: $EMAIL
echo output path: $OUTPUT_PATH

# ask user to continue
read -p "Todo ok? (y/n)" -n 1 -r

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi


# Replace paths in cesga/vl3d_cesga_env.sh
sed -i "s|export VL3D_DIR=.*|export VL3D_DIR='$VL3D_DIR'|" $VL3D_DIR/cesga/vl3d_cesga_env.sh
sed -i "s|export VL3D_SCRIPT=.*|export VL3D_SCRIPT='$VL3D_SCRIPT_DIR'|" $VL3D_DIR/cesga/vl3d_cesga_env.sh
sed -i "s|export VL3D_ENV=.*|export VL3D_ENV='$VL3D_ENV'|" $VL3D_DIR/cesga/vl3d_cesga_env.sh

# Replace e-mail in work.sh
sed -i "s/--mail-user=.*/--mail-user=$EMAIL/" $(realpath $(dirname $0)/slurm/work.sh)

# Replace paths in replace_paths.sh
sed -i 's|sed_model|'$MODEL'|g' $VL3D_DIR/scripts/experiment_generation/replace_paths.sh
sed -i 's|sed_experiment_name|'$EXPERIMENT_NAME'|g' $VL3D_DIR/scripts/experiment_generation/replace_paths.sh
sed -i 's|sed_output_path|'$OUTPUT_PATH'|g' $VL3D_DIR/scripts/experiment_generation/replace_paths.sh
sed -i 's|sed_json_dir|'$JSON_DIR'|g' $VL3D_DIR/scripts/experiment_generation/replace_paths.sh


# Replace paths in json files
$VL3D_DIR/scripts/experiment_generation/replace_paths.sh


# Launch work
$VL3D_DIR/scripts/slurm/launch.sh $JSON_DIR

