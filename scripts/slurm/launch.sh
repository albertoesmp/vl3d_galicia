#!/bin/bash


for f in $VL3D_DIR/experiments/vegetation/$(whoami)/*.json
do
  sbatch work.sh $f
done