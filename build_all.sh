#!/bin/bash

set -e

# choose local directory where packages will be installed
if [ -z "$TESTRELDIR" ]; then
  export INSTDIR=`pwd`/install
else
  export INSTDIR="$TESTRELDIR"
fi

cmake_option="RelWithDebInfo"
pyInstallStyle="develop"
force_clean=1
build_ext_list=""
PSANA_PATH=`pwd`/psana

if [ -d "/cds/sw/" ]; then
    no_daq=0
elif [ -d "/sdf/group/lcls/" ]; then
    no_daq=1
fi

while getopts "c:p:b:fd" opt; do
  case $opt in
    c) cmake_option="$OPTARG"
    ;;
    d) no_daq=1
    ;;
    p) pyInstallStyle="$OPTARG"
    ;;
    b) build_ext_list="$OPTARG"
    ;;
    f) force_clean=1                  # Force clean is required building between rhel6&7
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1
    ;;
  esac
done

pyver=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")

echo "CMAKE_BUILD_TYPE:" $cmake_option
echo "Python install option:" $pyInstallStyle
echo "build_ext_list:" $build_ext_list
export BUILD_LIST=$build_ext_list

if [ $force_clean == 1 ]; then
    echo "force_clean"
    if [ -d "$INSTDIR" ]; then
        rm -rf "$INSTDIR"
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
    if [ -d psana/build ]; then
        rm -rf psana/build
    fi
fi

function cmake_build() {
    # Capture the install flag
    make_install=$1

    # Create and navigate to the build directory
    mkdir -p build
    cd build

    # Run CMake configuration with the remaining arguments
    cmake -DPIP_OPTIONS="$pipOptions" -DCMAKE_INSTALL_PREFIX="$INSTDIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DCMAKE_BUILD_TYPE="$cmake_option" ..

    # Check the make_install flag
    if [ "$make_install" -eq 1 ]; then
        make -j 4 install
    else
        make -j 4
    fi
}

# "python setup.py develop" seems to not create this for you
# (although "install" does)
mkdir -p $INSTDIR/lib/python$pyver/site-packages/
if [ $pyInstallStyle == "develop" ]; then
    pipOptions="--editable"
else
    pipOptions=""
fi

#########
# Build #
#########
cmake_build 1
