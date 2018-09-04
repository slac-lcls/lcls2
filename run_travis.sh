#!/bin/bash

set -e

source activate $CONDA_ENV

cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH="$PWD/install/bin:$PATH"

PYVER=$(python -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))")
export PYTHONPATH="$PWD/install/lib/python$PYVER/site-packages"

if [[ $PYVER == 2.7 ]]; then
    ./build_python2_psana.sh
    nosetests psana/psana/tests
elif [[ $PYVER == 3.* ]]; then
    ./build_travis.sh -p install
    pytest psana/psana/tests
fi
