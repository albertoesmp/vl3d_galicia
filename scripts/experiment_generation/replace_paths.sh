#!/bin/bash

# # getopts input dir
# while getopts ":i:" opt; do
#   case ${opt} in
#     i )
#       JSON_DIR=$OPTARG
#       ;;
#     \? )
#       echo "Usage: cmd -i <jsons_directory>"
#       ;;
#   esac
# done

MODEL=sed_model
EXPERIMENT_NAME=sed_experiment_name
OUTPUT_PATH=sed_output_path
JSON_DIR=sed_json_dir


pipe_filename=$(basename $(find $MODEL -name "*.pipe"))
nn_filename=$(basename $(find $MODEL -name "*.keras"))

# pipe_path="/home/usc/ci/myg/Codigos/vl3d_galicia/models/vegetation/KPC_T3.pipe"
# nn_path="/home/usc/ci/myg/Codigos/vl3d_galicia/models/vegetation/KPC_T3.keras"

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
