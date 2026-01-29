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
    meson setup "$BUILDDIR" $OPTIONS --reconfigure || \
    meson setup "$BUILDDIR" $OPTIONS --wipe
fi
meson compile -C "$BUILDDIR"
meson install -C "$BUILDDIR"

pip install --prefix=$INSTDIR .

# Reset LDFLAGS back:
export LDFLAGS="$LDFLAGS_OLD"
unset LDFLAGS_OLD
