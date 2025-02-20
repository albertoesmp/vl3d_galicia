#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Unload all modules (just in case)
module --force purge

for f in $1/*.json
do
	if [ -f $f ]; then
		filename=$(basename -- "$f")
		filename="${filename%.*}"

		# TESTING. Disable GPU for even executions
		num=$(echo "$filename" | grep -oE '[0-9]+$')
		if (( num % 2 == 0)); then
		sed -i '5s|.*|#DISABLED: SBATCH --gres=gpu:a100|g' $SCRIPT_DIR/work.sh
		else
		sed -i '5s|.*|#SBATCH --gres=gpu:a100|g' $SCRIPT_DIR/work.sh
		fi

		sbatch --output=$2/${filename}_%j.log --error=$2/${filename}_error_%j.log --job-name=$filename $SCRIPT_DIR/work.sh $f
else
    echo "File $f does not exist! Check your paths. Exiting."
		exit 1
fi
done
