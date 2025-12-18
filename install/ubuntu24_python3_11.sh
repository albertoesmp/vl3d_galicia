#!/bin/bash

# AUTHOR: Alberto M. Esmoris Pena
#
# Script to install Python3.11 on Ubuntu 24.04
# 
# CALL THIS SCRIPT WITH ADMINISTRATOR RIGHTS

# Add dead snakes repository and install PYthon 3.11
add-apt-repository -y ppa:deadsnakes/ppa && \
apt-get update && \
apt-get install -y python3.11*
