#!/bin/bash

OS=${1:-linux}

set -e


export LCLS_TRAVIS=1
if [[ $OS == linux ]]; then
  source activate $CONDA_ENV

  cd "$(dirname "${BASH_SOURCE[0]}")"
  export PATH="$PWD/install/bin:$PATH"

  PYVER=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
  export PYTHONPATH="$PWD/install/lib/python$PYVER/site-packages"

  ./build_all.sh -p install
  pytest psana/psana/tests
  pytest psdaq/psdaq/tests
elif [[ $OS == osx ]]; then
  # add conda to the path
  source "$HOME/miniconda/etc/profile.d/conda.sh"
  # needed to use the conda compilers
  export MACOSX_DEPLOYMENT_TARGET="10.9"
  export CONDA_BUILD_SYSROOT="${HOME}/MacOSX${MACOSX_DEPLOYMENT_TARGET}.sdk"
  # needed for cmake
  export SDKROOT="${CONDA_BUILD_SYSROOT}"
  if [ -d "${CONDA_BUILD_SYSROOT}" ]; then
    echo "Found CONDA_BUILD_SYSROOT: ${CONDA_BUILD_SYSROOT}"
  else
    echo "Missing CONDA_BUILD_SYSROOT: ${CONDA_BUILD_SYSROOT}"
    exit 1
  fi

  conda activate $CONDA_ENV

  cd "$(dirname "${BASH_SOURCE[0]}")"
  export PATH="$PWD/install/bin:$PATH"

  PYVER=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
  export PYTHONPATH="$PWD/install/lib/python$PYVER/site-packages"

  ./build_all.sh -d -p install
  pytest psana/psana/tests
else
  echo "Unknown OS type: ${OS}"
  exit 1
fi
