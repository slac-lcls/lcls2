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
psana_setup_args=""
force_clean=0
no_ana=0
no_shmem=0
build_ext_list=""
install=1
PSANA_PATH=`pwd`/psana

<<<<<<< HEAD
if [ -d "/cds/sw/" ]; then
    no_daq=0
elif [ -d "/sdf/group/lcls/" ]; then
    no_daq=1
fi
=======
export REBUILD=1
>>>>>>> cfd5f6955 (shortcut)

while getopts "c:p:s:b:fdam" opt; do
  case $opt in
    c) cmake_option="$OPTARG"
    ;;
    d) no_daq=1
    ;;
    a) no_ana=1
    ;;
    m) no_shmem=1
    ;;
    p) pyInstallStyle="$OPTARG"
    ;;
    s) psana_setup_args="$OPTARG"
    ;;
    b) build_ext_list="$OPTARG"
    ;;
    f) force_clean=1                       # Force clean is required building between rhel6&7
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
    cd $1
    shift
    mkdir -p build
    cd build
    cmake -DPIP_OPTIONS=$pipOptions -DPSANA_PATH=$PSANA_PATH -DCMAKE_INSTALL_PREFIX=$INSTDIR -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=$cmake_option $@ ..
    if [ $install == 1 ] 
    then
        make -j 4 install
    else
        make -j 4
    fi
    cd ../..
}

# "python setup.py develop" seems to not create this for you
# (although "install" does)
mkdir -p $INSTDIR/lib/python$pyver/site-packages/
if [ $pyInstallStyle == "develop" ]; then
    pipOptions="--editable"
else
    pipOptions=""
fi

cmake_build xtcdata

if [ $no_shmem == 0 ]; then
    cmake_build psalg
else
    cmake_build psalg -DBUILD_SHMEM=OFF
fi
cd psalg
pip install --no-deps --prefix=$INSTDIR $pipOptions .
cd ..

if [ $no_daq == 0 ]; then
    # to build psdaq with setuptools
    cmake_build psdaq
    cd psdaq
    # force build of the extensions.  do this because in some cases
    # setup.py is unable to detect if an external header file changed
    # (e.g. in xtcdata).  but in many cases it is fine without "-f" - cpo
    if [ $pyInstallStyle == "develop" ]; then
        python setup.py build_ext -f --inplace
    fi
    pip install --no-deps --prefix=$INSTDIR $pipOptions .
    cd ..
fi

install=0

if [ $no_ana == 0 ]; then
    cmake_build psana
fi
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
