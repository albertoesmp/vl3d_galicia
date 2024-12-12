#!/bin/bash

# --------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena                                       #
# BRIEF:    Script to install the dependencies of the VL3D++ libraries  #
#           in Ubuntu systems.                                          #
#                                                                       #
# Better call with sudo.                                                #
# --------------------------------------------------------------------- #

 
apt-get install -y \
    build-essential zip unzip g++ make cmake git pkgconf \
    pcaputils liblz4-dev libpcap-dev \
    libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev \
    libhdf5-dev libboost-dev libboost-all-dev