# LCLS II development repository
## Build instructions
## Note: build on psbuild-rhel6 where redhat gcc6 compilers are installed
```bash
# setup conda
source /reg/g/psdm/sw/conda2/manage/bin/psconda.sh  (conda requires bash)

mkdir build
cd build
cmake ..
make (optionally with "-j N" to compile in parallel on N cores)

make test (runs some quick tests writing/reading xtc and hdf5 files)

you can change between optimize/debug builds by running cmake with the following:

cmake -DCMAKE_BUILD_TYPE={Debug, RelWithDebInfo, Release} ..
```
