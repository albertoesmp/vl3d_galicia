#!/bin/bash

# ------------------------------------------------------------------#
# AUTHOR: Alberto M. Esmoris Pena                                   #
# BRIEF: Install/deploy VL3D++ at CESGA FT-III                      #
#                                                                   #
# NOTE that the installation must be done from a GPU node           #
# to enable GPU acceleration for deep learning.                     #
# ------------------------------------------------------------------#

# Load environemnt
. vl3d_cesga_env.sh

# Install requirements with pip
pip install --target "${VL3D_PKG}" -r "${VL3D_CESGA_REQ}"

# Install dependencies for C++ extensions
cd ${VL3D_CPP_LIB}
${VL3D_CPP_LIB_INSTALL_SCRIPT}

# Build C++ extensions
cd ${VL3D_CPP}
${VL3D_CPP_BUILD_RELEASE_SCRIPT}

# Launch tests
cd ${VL3D_DIR}
python ${VL3D_SCRIPT} --test
