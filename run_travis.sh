#!/bin/bash

OS=${1:-linux}

set -e


export LCLS_TRAVIS=1
if [[ $OS == linux ]]; then
  #while true
  #do
  #  echo "ci-debug branch: checking calibdb access"
  #  time curl -s "https://pswww.slac.stanford.edu/calib_ws/cdb_ueddaq02/gridfs/6035d64545db0b188f7c78e8" | wc
  #  echo "done checking calibdb access"
  #done
  source activate $CONDA_ENV

  python -c "while 1: import requests; requests.get('https://pswww.slac.stanford.edu/calib_ws/cdb_ueddaq02/gridfs/6035d64545db0b188f7c78e8',None); print('*** done fetch')"

  cd "$(dirname "${BASH_SOURCE[0]}")"
  export PATH="$PWD/install/bin:$PATH"

  PYVER=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
  export PYTHONPATH="$PWD/install/lib/python$PYVER/site-packages"
  export TESTRELDIR="$PWD/install"

  pip install pytest-timeout
  ./build_all.sh -d -p install
  pytest -s --capture=no psana/psana/tests
elif [[ $OS == osx ]]; then
  echo "ignore macos"
else
  echo "Unknown OS type: ${OS}"
  exit 1
fi
