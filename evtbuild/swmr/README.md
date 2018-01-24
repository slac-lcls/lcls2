## Test scripts for measuring the read/write performance in HDF5 SWMR mode
These scripts test the performance of simulataneous read/write to HDF5 files in SWMR mode. The tests run in parallel using MPI with three tasks: writing, copying, and filtering.

Each write client creates and writes to a single file. This file is opened by a copy client, which reads in all values, and a filter client, which performs random(ish) reading. 

Configuration values are stores in sconfig, mainly the filter fraction, file size, image size,  and path. 

### Running a single test case
These tests have mostly characterized the drp-tst-acc0x and daq-tst-dev0x nodes. You can run a single test case with

```
`which mpirun` -q -map-by node --oversubscribe -n %i -H <<nodes>> python rwc_mpi.py | tee -a <<filename>>
```

### Looping HDF test
The loops are defined in ```mpi_call.py```. The user may wish to change the list of nodes (node_list), range of cores iterated over, and number of repeititons of the test. This is called simply with
```
python mpi_call.py
```

### Creating a psconda environment in RHEL6 with the latest HDF5/h5py
These scripts require HDF >1.10 and h5py >2.7. These can be installed by
```
ssh psbuild-rhel6
conda create -n snake_rhel6 --clone ana-1.3.44
source activate snake_rhel6
conda install -c anaconda hdf5
```