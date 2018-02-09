# LCLS II development repository
## Build instructions
## Note: build on psbuild-rhel6 where redhat gcc6 compilers are installed
```bash
# setup conda
source /reg/g/psdm/sw/conda2/manage/bin/psconda.sh  (conda requires bash)

# repository consists of seperate packages: xtcdata, psdaq, drp and psana
# all packages depend on xtcdata

# choose local directory where packages will be installed
export INSTDIR=

# to build xtcdata with cmake
cd xtcdata
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$INSTDIR ..
make -j 4 install

# to build psdaq and drp (after building xtcdata) with cmake
# eventually psdaq and drp will be merged and put in the INSTDIR
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$INSTDIR ..
make -j 4

# to build psana with setuptools
export PYTHONPATH=./psana
cd psana
python setup.py build_ext --xtcdata=$INSTDIR


# you can change between optimize/debug builds by running cmake with the following:

cmake -DCMAKE_BUILD_TYPE={Debug, RelWithDebInfo, Release} ..
```
