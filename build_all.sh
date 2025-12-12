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
force_clean=0
build_ext_list=""

if [ -d "/cds/sw/" ]; then
    no_daq=0
elif [ -d "/sdf/group/lcls/" ]; then
    no_daq=1
fi

while getopts "fd" opt; do
  case $opt in
    d) no_daq=0
    ;;
    f) force_clean=1                  # Force clean is required building between rhel6&7
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1
    ;;
  esac
done

echo "INSTDIR:" $INSTDIR

if [ $force_clean == 1 ]; then
    echo "force_clean"
    if [ -d "$INSTDIR" ]; then
        rm -rf "$INSTDIR"
    fi
    if [ -d build ]; then
        rm -rf build
    fi
    if [ -d builddir ]; then
        rm -rf builddir
    fi
fi


#########
# Build #
#########
if [ ! -d builddir ]; then
    meson setup builddir -Dprefix=$INSTDIR
fi
meson compile -C builddir
pip install --prefix=$INSTDIR .
