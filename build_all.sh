#!/bin/bash

# USAGE: ./build_all.sh (Optional arguments: Release (default), RelWithDebInfo, Debug)

set -e
source setup_env.sh

# choose local directory where packages will be installed
export INSTDIR=`pwd`/install

if [ "$#" -eq  "0" ]
  then
      cmake_option="-DCMAKE_INSTALL_PREFIX=$INSTDIR"
      echo $cmake_option
  else
      cmake_option="-DCMAKE_INSTALL_PREFIX=$INSTDIR -DCMAKE_BUILD_TYPE=$1"
      echo $cmake_option
fi

# to build xtcdata with cmake
cd xtcdata
mkdir -p build
cd build
cmake $cmake_option ..
make -j 4 install
cd ../..

# to build psdaq and drp (after building xtcdata) with cmake
cd psdaq
mkdir -p build
cd build
cmake $cmake_option ..
make -j 4 install
cd ../..

# to build psalg with cmake
cd psalg
mkdir -p build
cd build
cmake $cmake_option ..
make -j 4 install
cd ../..

pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
# "python setup.py develop" seems to not create this for you
# (although "install" does)
mkdir -p $INSTDIR/lib/python$pyver/site-packages/

# to build psana with setuptools
cd psana
# force build of the dgram extention
python setup.py build_ext --xtcdata=$INSTDIR -f --inplace
python setup.py develop --xtcdata=$INSTDIR --prefix=$INSTDIR
cd ..
# build ami
cd ami
python setup.py develop --prefix=$INSTDIR
