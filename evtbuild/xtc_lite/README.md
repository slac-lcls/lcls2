## Test scripts for measuring the read/write performance in "XTC" mode
These scripts test the performance of simulataneous read/write to a simplified XTC file format. The tests run in parallel using MPI with three tasks: writing, copying, and filtering.

Each write client creates and writes to a single file. This file is opened by a copy client, which reads in all values, and a filter client, which performs random(ish) reading. 

Configuration values are stores in ```sconfig```, mainly the filter fraction, file size, image size,  and path. 

### Running a single test case
These tests have mostly characterized the drp-tst-acc0x and daq-tst-dev0x nodes. You can run a single test case with

```
`which mpirun` -q -map-by node --oversubscribe -n %i -H <<nodes>> python rwc_xtc_mpi.py | tee -a <<filename>>
```

### Looping the XTC test
The loops are defined in ```xtc_mpi_call.py```. The user may wish to change the list of nodes (node_list), range of cores iterated over, and number of repetitions of the test. This is called simply with
```
python xtc_mpi_call.py
```
The output is saved to a plaintext file for later processing. 
