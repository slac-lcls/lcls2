#!/bin/bash
source /reg/g/psdm/sw/conda2/manage/bin/psconda.sh

# choose local directory where packages will be installed
export INSTDIR=`pwd`/install

# to build xtcdata with cmake
cd xtcdata
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$INSTDIR ..
make -j 4 install
cd ../..

# to build psdaq and drp (after building xtcdata) with cmake
# eventually psdaq and drp will be merged and put in the INSTDIR
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$INSTDIR ..
make -j 4
cd ..

# to build psana with setuptools
cd psana
python setup.py build_ext --xtcdata=$INSTDIR --inplace
