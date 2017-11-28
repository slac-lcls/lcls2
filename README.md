# LCLS II development repository
## Build instructions (on psdev)
```bash
# setup conda
source /reg/g/psdm/sw/conda2/manage/bin/psconda.sh  (conda requires bash)

mkdir build
cd build
cmake ..
make (optionally with "-j N" to compile in parallel on N cores)

you can change between optimize/debug builds by running cmake with the following:

cmake -DCMAKE_BUILD_TYPE={Debug, RelWithDebInfo, Release} ..
```

