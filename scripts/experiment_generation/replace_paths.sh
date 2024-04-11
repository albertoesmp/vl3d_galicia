#!/bin/bash

# getopts input dir
while getopts ":i:" opt; do
  case ${opt} in
    i )
      JSON_DIR=$OPTARG
      ;;
    \? )
      echo "Usage: cmd -i <jsons_directory>"
      ;;
  esac
done

pipe_path="/home/usc/ci/myg/Codigos/vl3d_galicia/models/vegetation/KPC_T3.pipe"
nn_path="/home/usc/ci/myg/Codigos/vl3d_galicia/models/vegetation/KPC_T3.keras"
output_path="/mnt/netapp1/Store_usccimyg/Results/galicia_vl3d/vegetation"

for file in $JSON_DIR/*.json
do
  sed -i 's|path_to_model_pipe|'$pipe_path'|g' $file
  sed -i 's|path_to_model_nn|'$nn_path'|g' $file
  sed -i 's|path_to_output_clouds|'$output_path'|g' $file
done
