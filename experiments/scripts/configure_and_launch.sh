#!/bin/bash


# Do not touch
VL3D_DIR=$(realpath $(dirname $0)/../../)
VL3D_SCRIPT_DIR=$VL3D_DIR/vl3d.py
EXPERIMENT_NAME=vegetation
MODEL=/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_vegetation/T5/prep_pipe
JSON_DIR=$VL3D_DIR/experiments/spec/$EXPERIMENT_NAME/$(whoami)

case $(whoami) in
    usccimyg)
    echo "Miguel"
    EMAIL=miguel.yermo@usc.es
    OUTPUT_PATH=/mnt/netapp1/Store_usccimyg/Results/galicia_vl3d/2025/$EXPERIMENT_NAME
    ;;

    usccisss)
    echo "Samuel"
    EMAIL=s.soutullo@usc.es
    OUTPUT_PATH=/mnt/netapp1/Store_usccisss/results/vl3d_galicia/2025/$EXPERIMENT_NAME
    ;;

    usccisra)
    echo "Silvia"
    EMAIL=silvia.alcaraz@usc.es
    OUTPUT_PATH=/mnt/netapp2/Store_usccisra/results/vl3d_galicia/2025/$EXPERIMENT_NAME
    ;;

    *)
    echo "Quen carallo es ti?"
    ;;
esac

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
# sed -i "s|export VL3D_SCRIPT=.*|export VL3D_SCRIPT='$VL3D_SCRIPT_DIR'|" $VL3D_DIR/cesga/vl3d_cesga_env.sh
# sed -i "s|export VL3D_ENV=.*|export VL3D_ENV='$VL3D_ENV'|" $VL3D_DIR/cesga/vl3d_cesga_env.sh

# Replace e-mail in work.sh
sed -i "s/--mail-user=.*/--mail-user=$EMAIL/" $(realpath $(dirname $0)/slurm/work.sh)

# Replace source path in work.sh
sed -i "s|source.*|source $VL3D_DIR/cesga/vl3d_cesga_env.sh|" $(realpath $(dirname $0)/slurm/work.sh)

# Replace paths in replace_paths.sh
sed -i 's|MODEL=.*|MODEL='$MODEL'|' $VL3D_DIR/experiments/scripts/experiment_generation/replace_paths.sh
sed -i 's|EXPERIMENT_NAME=.*|EXPERIMENT_NAME='$EXPERIMENT_NAME'|' $VL3D_DIR/experiments/scripts/experiment_generation/replace_paths.sh
sed -i 's|OUTPUT_PATH=.*|OUTPUT_PATH='$OUTPUT_PATH'|' $VL3D_DIR/experiments/scripts/experiment_generation/replace_paths.sh
sed -i 's|JSON_DIR=.*|JSON_DIR='$JSON_DIR'|' $VL3D_DIR/experiments/scripts/experiment_generation/replace_paths.sh

# Replace source paths in work.sh
sed -i "s|source .*|source $VL3D_DIR/cesga/vl3d_cesga_env.sh|" $VL3D_DIR/experiments/scripts/slurm/work.sh

# Replace paths in json files
$VL3D_DIR/experiments/scripts/experiment_generation/replace_paths.sh

# Create logs directory
LOG_DIR=$VL3D_DIR/experiments/scripts/slurm/logs/$(whoami)/$EXPERIMENT_NAME
mkdir -p $LOG_DIR

# Launch work
$VL3D_DIR/experiments/scripts/slurm/launch.sh $JSON_DIR $LOG_DIR