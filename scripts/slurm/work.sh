#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:a100
#SBATCH -t 72:00:00
#SBATCH --mem 247GB
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=miguel.yermo@usc.es


# ---  PREPARE ENVIRONMENT  --- #
# ----------------------------- #
source $HOME/Codigos/vl3d_galicia/cesga/vl3d_cesga_env.sh


# ---  VARIABLES  --- #
# ------------------- #
# TRAINING_SPEC="$HOME/Codigos/virtualearn3d/cesga/galicia/kpconv_final_X_training_T2.json"
PREDICTIVE_SPEC=$1

# ---  EXECUTION  --- #
# ------------------- #
# RUN SCRIPTS
cd_vl3d
srun python ${VL3D_SCRIPT} --test
# srun python ${VL3D_SCRIPT} --pipeline ${TRAINING_SPEC}
srun python ${VL3D_SCRIPT} --pipeline ${PREDICTIVE_SPEC}
# python ${VL3D_SCRIPT} --pipeline ${PREDICTIVE_SPEC}
