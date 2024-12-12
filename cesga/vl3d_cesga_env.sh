#!/bin/bash

# ------------------------------------------------------------------#
# AUTHOR: Alberto M. Esmoris Pena                                   #
# BRIEF: Configure environment to run VL3D++ at CESGA FT-III        #
#                                                                   #
# ------------------------------------------------------------------#



# ---  GLOBAL VARIABLES  --- #
# -------------------------- #
export VL3D_DIR='/mnt/netapp2/Store_uni/home/usc/ci/aep/vl3dpp'
export VL3D_SCRIPT="${VL3D_DIR}"'/vl3d.py'
export VL3D_CESGA_REQ="${VL3D_DIR}/cesga/requirements_cesga.txt"
export VL3D_CPP="${VL3D_DIR}/cpp"
export VL3D_CPP_LIB="${VL3D_CPP}/lib"
export VL3D_CPP_LIB_INSTALL_SCRIPT="${VL3D_CPP_LIB}/lib_install_cesga.sh"
export VL3D_CPP_BUILD_RELEASE_SCRIPT="${VL3D_CPP}/build_release.sh"
export VL3D_PKG="${STORE}/vl3d_pkg"



# ---  CONFIGURE ENVIRONMENT  --- #
# ------------------------------- #
# Load modules
module load cesga/2022 python/3.10.8 cuda/12.2.0
# Configure PYTHONPATH prioritizing custom packages
export PYTHONPATH=${VL3D_PKG}:${PYTHONPATH}
# Link ptxas (cuda toolkit nvvm libdevice)
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/cesga/2022/software/Core/cuda/12.2.0



# ---  UTIL FUNCTIONS  --- #
# ------------------------ #
# Change working directory to VL3D directory
function cd_vl3d {
    cd "${VL3D_DIR}"
}
