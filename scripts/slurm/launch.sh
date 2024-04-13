#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for f in $SCRIPT_DIR/../../experiments/vegetation/$(whoami)/*.json
do
	sbatch work.sh $f
done
