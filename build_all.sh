#!/bin/bash

set -e

# choose local directory where packages will be installed
export INSTDIR=`pwd`/install

cmake_option="Debug"
pyInstallStyle="develop"
psana_setup_args=""
force_clean=0
no_daq=0

while getopts ":c:p:s:f:d" opt; do
  case $opt in
    c) cmake_option="$OPTARG"
    ;;
    d) no_daq=1
    ;;
    p) pyInstallStyle="$OPTARG"
    ;;
    s) psana_setup_args="$OPTARG"
    ;;
    f) force_clean=1                       # Force clean is required building between rhel6&7
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo "CMAKE_BUILD_TYPE:" $cmake_option
echo "Python install option:" $pyInstallStyle

if [ $force_clean == 1 ]; then
    echo "force_clean"
    if [ -d install ]; then
        rm -rf install
    fi
    if [ -d xtcdata/build ]; then
        rm -rf xtcdata/build
    fi
    if [ -d psdaq/build ]; then
        rm -rf psdaq/build
    fi
    if [ -d psalg/build ]; then
        rm -rf psalg/build
    fi
fi

function cmake_build() {
    cd $1
    mkdir -p build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$INSTDIR -DCMAKE_BUILD_TYPE=$cmake_option ..
    make -j 4 install
    cd ../..
}

pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
# "python setup.py develop" seems to not create this for you
# (although "install" does)
mkdir -p $INSTDIR/lib/python$pyver/site-packages/

cmake_build xtcdata
cmake_build psalg
if [ $no_daq == 0 ]; then
    cmake_build psdaq
    cd psdaq
    python setup.py $pyInstallStyle --prefix=$INSTDIR
    cd ..
fi

# to build psana with setuptools
cd psana
# force build of the extensions.  do this because in some cases
# setup.py is unable to detect if an external header file changed
# (e.g. in xtcdata).  but in many cases it is fine without "-f" - cpo
python setup.py build_ext --xtcdata=$INSTDIR -f --inplace
python setup.py $pyInstallStyle $psana_setup_args --xtcdata=$INSTDIR --prefix=$INSTDIR
