# LCLS II development repository
## Build instructions (on psdev)
```bash
# setup conda
source /reg/g/psdm/etc/psconda.sh  (conda requires bash)


# activate python 3 environment with cmake and hdf5
source activate ana-1.3.28-py3

 mkdir build
 cd build
 cmake ..
 make
```
