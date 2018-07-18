#!/bin/bash

set -e
source setup_env_python2.sh

export INSTDIR=`pwd`/install
export PYTHONPATH=$INSTDIR/lib/python2.7/site-packages

function cmake_build() {
    cd $1
    mkdir -p build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$INSTDIR -DCMAKE_BUILD_TYPE=-DCMAKE_BUILD_TYPE=Release ..
    make -j 4 install                                                          
    cd ../..                                                 
}

cmake_build xtcdata
cmake_build psalg

mkdir -p $INSTDIR/lib/python2.7/site-packages/
cd psana
python setup.py build_ext --xtcdata=$INSTDIR -f --inplace
python setup.py develop  --xtcdata=$INSTDIR --prefix=$INSTDIR
cd ..

nosetests psana/psana/tests
