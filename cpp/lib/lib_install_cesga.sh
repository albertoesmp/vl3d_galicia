#!/bin/bash

# --------------------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena                                       #
# BRIEF: Script to install VL3D++ libraries at CESGA FT-III             #
# DEPS: The script needs:                                               #
#           - zip                                                       #
#           - g++ (supporting C++17)                                    #
#           - make                                                      #
#           - cmake                                                     #
#           - git                                                       #
#           - boost                                                     #
#           - pkgconf                                                   #
#           - pcaputils                                                 #
#           - liblz4                                                    #
#           - BLAS/OpenBLAS                                             #
#           - LAPACK                                                    #
#           - ARPACK2                                                   #
#           - SuperLU                                                   #
# The script must be called from the cpp/lib directory.                 #
# It will automatically download and build the dependencies.            #
# --------------------------------------------------------------------- #


# ---  GLOBAL VARIABLES  --- #
# -------------------------- #
CPP_VER=17
NTHREADS=32


# ---  PYBIND11 : LIBRARY  --- #
# ---------------------------- #
# Download PyBind11
git clone https://github.com/pybind/pybind11

# Install PyBind11
cd pybind11
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=${CPP_VER} -DCMAKE_INSTALL_PREFIX=./install/ .
make -j ${NTHREADS}
make -j ${NTHREADS} install
cd ..


# ---  CARMA : LIBRARY  --- #
# ------------------------- #
# Download carma
git clone https://github.com/RUrlus/carma

# Install carma
cd carma
cmake --build . --config Release
cd ..


# ---  ARMADILLO : LIBRARY  --- #
# ----------------------------- #
# Download armadillo
wget -c https://deac-riga.dl.sourceforge.net/project/arma/armadillo-14.0.1.tar.xz
tar xvf armadillo-14.0.1.tar.xz
mv armadillo-14.0.1 armadillo

# Install armadillo
cd armadillo
./configure
make -j ${NTHREADS}
cd ..


# ---  PCL : DEPENDENCIES  --- #
# ---------------------------- #
# Download Eigen3
wget -c https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip
mv eigen-3.4.0 eigen3

# Install Eigen3
mkdir eigen3/build
cd eigen3/build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=${CPP_VER} -DCMAKE_INSTALL_PREFIX=../install/ ..
make install -j ${NTHREADS}
cd ../..

# Download FLANN
git clone https://github.com/flann-lib/flann

# Install FLANN
mkdir flann/build
cd flann/build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=${CPP_VER} -DCMAKE_INSTALL_PREFIX=../install/ ..
make install -j ${NTHREADS}
cd ../..


# ---  PCL : LIBRARY  --- #
# ----------------------- #
# Download PCL
wget -c https://github.com/PointCloudLibrary/pcl/archive/refs/tags/pcl-1.14.1.zip
unzip pcl-1.14.1.zip
mv pcl-pcl-1.14.1 pcl

# Build PCL
mkdir pcl/build
cd pcl/build
cmake   -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=${CPP_VER} \
        -DCMAKE_CUDA_STANDARD=${CPP_VER} \
        -DBUILD_visualization=OFF \
	-DWITH_OPENGL=OFF \
        -DCMAKE_INSTALL_PREFIX=../install/ \
        ..
make -j ${NTHREADS}
make -j ${NTHREADS} install
cd ..
mv install/include/pcl-1.14/pcl install/include/pcl
rmdir install/include/pcl-1.14
cd ..

