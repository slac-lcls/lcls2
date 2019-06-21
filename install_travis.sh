#!/bin/bash

set -e

if [[ $TRAVIS_OS_NAME == osx ]]; then
  wget https://github.com/phracker/MacOSX-SDKs/releases/download/10.13/MacOSX10.9.sdk.tar.xz
  tar xJf MacOSX10.9.sdk.tar.xz -C $HOME/
  rm MacOSX10.9.sdk.tar.xz
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  hash -r
  conda config --set always_yes yes --set changeps1 no
  conda install conda-build anaconda-client
  conda update -q conda conda-build
  conda config --add channels lcls-ii
  conda config --append channels conda-forge
  # Useful for debugging any issues with conda
  conda info -a
  # Create test environment
  conda create -q -n $CONDA_ENV python=$TRAVIS_PYTHON_VERSION --file mac-requirements.txt
fi
