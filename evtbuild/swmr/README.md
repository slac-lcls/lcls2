# Test scripts for measuring the read/write performance in HDF5 SWMR mode

## Running a single test case
These tests have mostly characterized the drp-tst-acc0x and daq-tst-dev0x nodes. You can run a single test case with

```
`which mpirun` -q -map-by node --oversubscribe -n %i -H <<nodes>> python rwc_mpi.py | tee -a <<filename>>
```


## Looping HDF test
The loops are defined in ```rwc_mpi.py```. The user may wish to change the list of nodes (node_list), range of cores iterated over, and number of repeititons of the test. 




``python mpi_call.py```

## Creating a psconda environment in RHEL6 with the latest HDF5/h5py

```
ssh psbuild-rhel6
conda create -n snake_rhel6 --clone ana-1.3.44
source activate snake_rhel6
conda install -c anaconda hdf5
```