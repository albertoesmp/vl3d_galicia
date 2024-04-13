#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for f in $SCRIPT_DIR/../../experiments/vegetation/$(whoami)/*.json
do
	if [ -f $f ]; then
		sbatch work.sh $f
else
    echo "File $f does not exist! Check your paths. Exiting."
		exit 1
fi
done
