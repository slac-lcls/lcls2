#!/bin/bash

set -e
source setup_env.sh

# choose local directory where packages will be installed
export INSTDIR=`pwd`/install

cmake_option="Debug"
pyInstallStyle="develop"

while getopts ":c:p:" opt; do
  case $opt in
    c) cmake_option="$OPTARG"
    ;;
    p) pyInstallStyle="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo "CMAKE_BUILD_TYPE:" $cmake_option
echo "Python install option:" $pyInstallStyle


function cmake_build() {
    cd $1
    mkdir -p build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$INSTDIR -DCMAKE_BUILD_TYPE=$cmake_option ..
    make -j 4 install
    cd ../..
}

cmake_build xtcdata
cmake_build psdaq
cmake_build psalg

pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
# "python setup.py develop" seems to not create this for you
# (although "install" does)
mkdir -p $INSTDIR/lib/python$pyver/site-packages/

# to build psana with setuptools
cd psana
# force build of the dgram extention
python setup.py build_ext --xtcdata=$INSTDIR -f --inplace
python setup.py $pyInstallStyle --xtcdata=$INSTDIR --prefix=$INSTDIR
cd ..
# build ami
cd ami
python setup.py $pyInstallStyle --prefix=$INSTDIR
