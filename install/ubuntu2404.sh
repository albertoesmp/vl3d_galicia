#!/bin/bash

# AUTHOR: Alberto M. Esmoris Pena
#
# Script to install VL3D on Ubuntu 24.04
# 
# CALL THIS SCRIPT WITH ADMINISTRATOR RIGHTS


# Exit if last command failed
function exit_on_fail {
	if [[ $? != 0 ]]; then
		exit 2
	fi
}


# To vl3d/cpp/lib
cd ../cpp/lib
./ubuntu_deps.sh
exit_on_fail 
# To vl3d/install
cd ../../install
./ubuntu24_python3_11.sh
exit_on_fail
# To vl3d
cd ..
python3.11 -m venv venv
exit_on_fail
. venv/bin/activate
exit_on_fail
pip install --no-cache-dir -r requirements.txt
exit_on_fail
# To vl3d/cpp/lib
cd cpp/lib
./lib_install.sh
exit_on_fail 
# To vl3d/cpp
cd ..
./build_release.sh
exit_on_fail
# To vl3d
cd ..
python vl3d.py --test
# Assign ownership to non-sudo user
if [[ $SUDO_USER != "" ]]; then
	chown -R ${SUDO_USER}: .
fi

