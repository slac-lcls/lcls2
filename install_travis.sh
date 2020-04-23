#!/bin/bash

set -e

if [[ $TRAVIS_OS_NAME == osx ]]; then
  # try to avoid dns (from bldg 50?) because failures may be affecting tests?
  echo "134.79.138.124 pswww.slac.stanford.edu" | sudo tee -a /etc/hosts

  git clone https://github.com/slac-lcls/relmanage.git $HOME/relmanage
  sed -i.old "s|PYTHONVER|${TRAVIS_PYTHON_VERSION}|g" "$HOME/relmanage/env_create_macos.yaml"
  wget https://github.com/phracker/MacOSX-SDKs/releases/download/10.13/MacOSX10.9.sdk.tar.xz
  tar xJf MacOSX10.9.sdk.tar.xz -C $HOME/
  rm MacOSX10.9.sdk.tar.xz
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
  bash miniconda.sh -b -p $HOME/miniconda
  source "$HOME/miniconda/etc/profile.d/conda.sh"
  conda config --set always_yes yes --set changeps1 no
  # Needed if we want to do conda builds and uploads in the future
  #conda install conda-build anaconda-client
  #conda update -q conda conda-build
  conda config --add channels lcls-ii
  conda config --append channels conda-forge
  # Useful for debugging any issues with conda
  conda info -a
  # Create test environment
  conda env create -q -n $CONDA_ENV -f $HOME/relmanage/env_create_macos.yaml
fi
