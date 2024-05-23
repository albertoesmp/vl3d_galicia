#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Unload all modules (just in case)
module --force purge

for f in $1/*.json
do
	echo $f
	if [ -f $f ]; then
		filename=$(basename -- "$f")
		filename="${filename%.*}"
		sbatch --job-name=$filename $SCRIPT_DIR/work.sh $f
else
    echo "File $f does not exist! Check your paths. Exiting."
		exit 1
fi
done
