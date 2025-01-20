#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:a100
#SBATCH -t 48:00:00
#SBATCH --mem 123GB
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=albertoesmp@gmail.com

# Author: Alberto M. Esmoris Pena
# Brief: Script to train models with VL3D framework


# ---  PREPARE ENVIRONMENT  --- #
# ----------------------------- #
source /home/usc/ci/aep/git/vl3dpp/cesga/vl3d_cesga_env.sh


# ---  VARIABLES  --- #
# ------------------- #
PREDICTIVE_SPEC='/home/usc/ci/aep/git/vl3dpp/cesga/galicia/vegetation/pnetpp/pnetpp_final_XIr_vegetation_predict_T3.json'

# ---  EXECUTION  --- #
# ------------------- #
# RUN SCRIPTS
cd_vl3d
#srun python ${VL3D_SCRIPT} --test
srun python ${VL3D_SCRIPT} --pipeline ${PREDICTIVE_SPEC}
