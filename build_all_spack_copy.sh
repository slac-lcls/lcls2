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

while getopts "c:p:b:fd" opt; do
  case $opt in
    c) cmake_option="$OPTARG"
    ;;
    d) no_daq=0
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

echo "INSTDIR:" $INSTDIR
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

# "python setup.py develop" seems to not create this for you
# (although "install" does)
mkdir -p $INSTDIR/lib/python$pyver/site-packages/
if [ $pyInstallStyle == "develop" ]; then
    pipOptions="--editable"
else
    pipOptions=""
fi


function cmake_build() {
    # Capture the install flag
    make_install=$1

    # Create and navigate to the build directory
    mkdir -p build
    cd build

    # Run CMake configuration with the remaining arguments
    cmake -DPIP_OPTIONS="$pipOptions" -DCMAKE_INSTALL_PREFIX="$INSTDIR" -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" -DCMAKE_BUILD_TYPE="$cmake_option" .. -DNO_DAQ=$no_daq

    # Check the make_install flag
    if [ "$make_install" -eq 1 ]; then
        make -j 4 install
    else
        make -j 4
    fi
}


#########
# Build #
#########
cmake_build 1

# The removal of site.py in setup 49.0.0 breaks "develop" installations
# which are outside the normal system directories: /usr, /usr/local,
# $HOME/.local. etc. See: https://github.com/pypa/setuptools/issues/2295
# The suggested fix, in the bug report, is the following: "I recommend
# that the project use pip install --prefix or possibly pip install
# --target to install packages and supply a sitecustomize.py to ensure
# that directory ends up as a site dir and gets .pth processing. That
# approach should be future-proof (at least against the sunset of
# easy_install). All python setup.py commands in the code above have
# been replaced with pip commands. The following code implements the
# sitecustomize.py file. Pip bilds the python modules in a sandbox,
# so it requires all the code for the module to be in the same
# folder. The C++ code for the modules built in psana was therefore
# moved from psalg to psana.
if [ $pyInstallStyle == "develop" ]; then
  if [ ! -f $INSTDIR/lib/python$pyver/site-packages/site.py ] && \
     [ ! -f $INSTDIR/lib/python$pyver/site-packages/sitecustomize.py ]; then
cat << EOF > $INSTDIR/lib/python$pyver/site-packages/sitecustomize.py
import site

site.addsitedir('$INSTDIR/lib/python$pyver/site-packages')
EOF
  fi
fi