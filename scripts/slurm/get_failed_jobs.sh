#!/bin/bash

# Script para buscar los trabajos que han fallaron. Se asume que si existen todos los directorios de un .json, y están vacíos, entonces el trabajo falló, siempre y cuando no acabe de entrar uno en cola.

# No lanzar antes de 5 horas después de que haya entrado un trabajo en cola.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Path a donde estan los MERGE_### creados.
RESULTS_DIR=$STORE2/Results/galicia_vl3d/vegetation

for f in $SCRIPT_DIR/../../experiments/vegetation/$(whoami)/*.json
do
  cloud_id=$(cat $f | grep -e "lidar_data" | grep -oP "(?<=MERGE_)\d+")

  all_empty=true

  for i in $cloud_id
  do
    if [ -z "$(ls -A $RESULTS_DIR/MERGE_$i)" ]; then
      # Do nothing
      :
    else
      all_empty=false
    fi
  done

  if [ "$all_empty" = true ]; then
    echo $f
  fi
done