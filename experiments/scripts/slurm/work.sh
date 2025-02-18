#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:a100
#SBATCH -t 24:00:00
#SBATCH --mem 123GB
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=miguel.yermo@usc.es

# ---  PREPARE ENVIRONMENT  --- #
# ----------------------------- #
source /mnt/netapp1/Store_usccimyg/Codigos/vl3d_galicia_dev/cesga/vl3d_cesga_env.sh


# ---  VARIABLES  --- #
# ------------------- #
PREDICTIVE_SPEC=$1

# ---  EXECUTION  --- #
# ------------------- #
# RUN SCRIPTS
cd_vl3d
# srun python ${VL3D_SCRIPT} --test
srun python ${VL3D_SCRIPT} --pipeline ${PREDICTIVE_SPEC}
