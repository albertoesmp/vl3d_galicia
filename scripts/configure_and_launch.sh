#!/bin/bash


# Do not touch
VL3D_DIR=$(realpath $(dirname $0)/../)
VL3D_SCRIPT_DIR=$VL3D_DIR/vl3d.py
EXPERIMENT_NAME=vegetation_2
MODEL=/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/pnet_final_XIr_alt/T5/pipe/
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
    OUTPUT_PATH=/mnt/netapp1/Store_usccisss/results/vl3d_galicia/$EXPERIMENT_NAME
    VL3D_ENV=/mnt/netapp1/Store_usccisss/vl3d_env
    ;;

    usccisra)
    echo "Silvia"
    EMAIL=silvia.alcaraz@usc.es
    OUTPUT_PATH=/mnt/netapp2/Store_usccisra/results/vl3d_galicia/$EXPERIMENT_NAME
    VL3D_ENV=/mnt/netapp2/Store_uni/home/usc/ci/sra/.conda/envs/vl3d
    ;;

    *)
    echo "Quen carallo es ti?"
    ;;
esac

echo Conda env: $VL3D_ENV
echo e-mail: $EMAIL
echo output path: $OUTPUT_PATH
echo VL3D directory: $VL3D_DIR

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
sed -i 's|MODEL=.*|MODEL='$MODEL'|' $VL3D_DIR/scripts/experiment_generation/replace_paths.sh
sed -i 's|EXPERIMENT_NAME=.*|EXPERIMENT_NAME='$EXPERIMENT_NAME'|' $VL3D_DIR/scripts/experiment_generation/replace_paths.sh
sed -i 's|OUTPUT_PATH=.*|OUTPUT_PATH='$OUTPUT_PATH'|' $VL3D_DIR/scripts/experiment_generation/replace_paths.sh
sed -i 's|JSON_DIR=.*|JSON_DIR='$JSON_DIR'|' $VL3D_DIR/scripts/experiment_generation/replace_paths.sh

# Replace source paths in work.sh
sed -i "s|source .*|source $VL3D_DIR/cesga/vl3d_cesga_env.sh|" $VL3D_DIR/scripts/slurm/work.sh

# Replace paths in json files
$VL3D_DIR/scripts/experiment_generation/replace_paths.sh


# Launch work
$VL3D_DIR/scripts/slurm/launch.sh $JSON_DIR

