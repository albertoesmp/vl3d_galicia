#!/bin/bash

MODEL=/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_vegetation/T5/prep_pipe
EXPERIMENT_NAME=vegetation
OUTPUT_PATH=/mnt/netapp1/Store_usccimyg/Results/galicia_vl3d/2025/vegetation
JSON_DIR=/mnt/netapp1/Store_usccimyg/Codigos/vl3d_galicia_dev/experiments/spec/vegetation/usccimyg

pipe_filename=$(basename $(find $MODEL -name "*.pipe"))
nn_filename=$(basename $(find $MODEL -name "*.keras"))

for file in $JSON_DIR/*.json
do
  echo $file
  model_copy_path=$STORE2/tmp/2025/models/$EXPERIMENT_NAME/$(basename $file .json)
  mkdir -p $model_copy_path
  cp $MODEL/* $model_copy_path

  # check if cp worked
  # if [ ! -f $model_copy_path/$pipe_filename ] || [ ! -f $model_copy_path/$nn_filename ]; then
  if [ ! -f $model_copy_path/$pipe_filename ]; then
    echo "Error copying model files"
    exit 1
  fi

  pipe_path=$model_copy_path/$pipe_filename
  nn_path=$model_copy_path/$nn_filename

  sed -i 's|path_to_model_pipe|'$pipe_path'|g' $file
  sed -i 's|path_to_model_nn|'$nn_path'|g' $file
  sed -i 's|path_to_output_clouds|'$OUTPUT_PATH'|g' $file
done
