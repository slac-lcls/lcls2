#!/bin/bash

set -e

BUILDDIR=builddir

# choose local directory where packages will be installed
if [ -z "$TESTRELDIR" ]; then
  export INSTDIR=$(pwd)/install
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

OPTIONS="-Dconda_prefix=$CONDA_PREFIX -Dprefix="$INSTDIR" -Depics_base=$EPICS_BASE -Depics_host_arch=$EPICS_HOST_ARCH"

if [ $no_daq == 0 ]; then
  OPTIONS="$OPTIONS -Dbuild_daq=true"
else
  OPTIONS="$OPTIONS -Dbuild_daq=false"
fi

# Have to clear LDFLAGS set by conda if we are compiling the cuda parts too
if command -v nvcc >/dev/null 2>&1; then
  export LDFLAGS_OLD="$LDFLAGS"
  export LDFLAGS=""
fi

#########
# Build #
#########
if [ ! -d "$BUILDDIR/meson-private" ]; then
    meson setup "$BUILDDIR" $OPTIONS
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
    cmake_build psdaq -DCPM_SOURCE_CACHE=$HOME/.cache/CPM
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

if [ $no_ana == 0 ]; then
    # to build psana with setuptools
    cd psana
    # force build of the extensions.  do this because in some cases
    # setup.py is unable to detect if an external header file changed
    # (e.g. in xtcdata).  but in many cases it is fine without "-f" - cpo
    if [ $pyInstallStyle == "develop" ]; then
        python setup.py build_ext -f --inplace
    fi
    pip install --no-deps --prefix=$INSTDIR $pipOptions .
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
meson compile -C "$BUILDDIR"
meson install -C "$BUILDDIR"

pip install --prefix=$INSTDIR .

# Reset LDFLAGS back:
export LDFLAGS="$LDFLAGS_OLD"
unset LDFLAGS_OLD
