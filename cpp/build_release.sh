#!/bin/bash

# ----------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena                                   #
# BRIEF: Build the C++ components of VL3D++ in Release mode.        #
# ----------------------------------------------------------------- #

NTHREADS=16

mkdir -p build
cd build
cmake -DBUILD_TYPE=Release -DPython3_FIND_VIRTUALENV=ONLY ..
make -j ${NTHREADS}
cd ..
