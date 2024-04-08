#!/bin/bash

# Get path of this script
SCRIPT=$(readlink -f "$0")
echo $SCRIPT
SCRIPTPATH=$(dirname "$SCRIPT")
cd $SCRIPTPATH

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

pipe_path="/home/usc/ci/myg/store2/vl3d/models/kpc_final_X/T2/pipe/KPC_T2.pipe"
nn_path="/home/usc/ci/myg/store2/vl3d/models/kpc_final_X/T2/pipe/KPC_T2.keras"
output_path="/home/usc/ci/myg/store2/vl3d/models/kpc_final_X/T2/preds"

for file in $JSON_DIR/*.json
do
  sed -i 's|path_to_model_pipe|'$pipe_path'|g' $file
  sed -i 's|path_to_model_nn|'$nn_path'|g' $file
  sed -i 's|path_to_output_clouds|'$output_path'|g' $file
done
