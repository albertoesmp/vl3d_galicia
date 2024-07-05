#!/bin/bash

MODEL=/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/kpc_final_XIrRGB_buildveg/T4/pipe/
EXPERIMENT_NAME=buildveg
OUTPUT_PATH=/mnt/netapp1/Store_usccimyg/Results/galicia_vl3d/buildveg
JSON_DIR=/mnt/netapp2/Store_uni/home/usc/ci/myg/Codigos/vl3d_galicia/experiments/buildveg/usccimyg

pipe_filename=$(basename $(find $MODEL -name "*.pipe"))
nn_filename=$(basename $(find $MODEL -name "*.keras"))

for file in $JSON_DIR/*.json
do
  model_copy_path=$STORE2/tmp/models/$EXPERIMENT_NAME/$(basename $file .json)
  mkdir -p $model_copy_path
  cp $MODEL/* $model_copy_path

  # check if cp worked
  if [ ! -f $model_copy_path/$pipe_filename ] || [ ! -f $model_copy_path/$nn_filename ]; then
    echo "Error copying model files"
    exit 1
  fi

  pipe_path=$model_copy_path/$pipe_filename
  nn_path=$model_copy_path/$nn_filename

  sed -i 's|path_to_model_pipe|'$pipe_path'|g' $file
  sed -i 's|path_to_model_nn|'$nn_path'|g' $file
  sed -i 's|path_to_output_clouds|'$OUTPUT_PATH'|g' $file
done
