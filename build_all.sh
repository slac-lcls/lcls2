#!/bin/bash

set -e

BUILDDIR=builddir

# choose local directory where packages will be installed
if [ -z "$TESTRELDIR" ]; then
  export INSTDIR=$(pwd)/install
else
  export INSTDIR="$TESTRELDIR"
fi

force_clean=0
compile_only=0

if [ -d "/cds/sw/" ]; then
  build_daq=1
elif [ -d "/sdf/group/lcls/" ]; then
  build_daq=0
fi

while getopts "fdc" opt; do
  case $opt in
    d) build_daq=1
    ;;
    f) force_clean=1                  # Force clean is required building between rhel6&7
    ;;
    c) compile_only=1
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

OPTIONS="-Dconda_prefix=$CONDA_PREFIX \
         -Dprefix="$INSTDIR" \
         -Depics_base=$EPICS_BASE \
         -Depics_host_arch=$EPICS_HOST_ARCH \
         -Dpython.bytecompile=-1"

# When building for a release (debug is default)
#OPTIONS="$OPTIONS -Dbuildtype=release"

if [ $build_daq == 1 ]; then
  OPTIONS="$OPTIONS -Dbuild_daq=true"
else
  OPTIONS="$OPTIONS -Dbuild_daq=false"
fi

# Have to clear LDFLAGS set by conda if we are compiling the cuda parts too
if command -v nvcc >/dev/null 2>&1; then
  export LDFLAGS_OLD="$LDFLAGS"
  export LDFLAGS=""
  export CXXFLAGS_OLD="$CXXFLAGS"
  export CXXFLAGS=""
  # If CUDA_ROOT is set and exists use that not what's in conda
  if [ -n "$CUDA_ROOT" ] && [ -e "$CUDA_ROOT" ]; then
    OPTIONS="$OPTIONS -Dcustom_cuda_path=$CUDA_ROOT"
  fi
fi

#########
# Build #
#########
if [ ! -d "$BUILDDIR" ]; then
  meson setup "$BUILDDIR" $OPTIONS
fi
meson compile -C "$BUILDDIR" -j8
meson install --only-changed --no-rebuild --quiet -C "$BUILDDIR"

if [ $compile_only == 0 ]; then
  uv pip install . \
    --no-compile \
    --no-deps \
    --no-build-isolation \
    --prefix=$INSTDIR \
    --config-settings setup-args="$OPTIONS" \
    --config-settings compile-args="-j8" \
    --config-settings install-args="--only-changed --no-rebuild"
    #-v
    #--no-index

  if [ $build_daq == 1 ]; then
    cd psdaq
      uv pip install . \
        --no-compile \
        --no-deps \
        --no-build-isolation \
        --prefix=$INSTDIR \
        --config-settings setup-args="$OPTIONS" \
        --config-settings compile-args="-j8" \
        --config-settings install-args="--only-changed --no-rebuild"
      # -v
      #--no-index
    cd ..
  fi
fi

if command -v nvcc >/dev/null 2>&1; then
  # Reset LDFLAGS and CXXFLAGS back:
  export LDFLAGS="$LDFLAGS_OLD"
  unset LDFLAGS_OLD
  export CXXFLAGS="$CXXFLAGS_OLD"
  unset CXXFLAGS_OLD
fi
