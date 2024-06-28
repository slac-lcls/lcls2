####################################################################
# A program that takes unsorted hdf5 file and sort the data by its
# timestamp and rewrite all sorted datasets to another hdf5 file.
#
# The output h5file is a virtual datasource that binds all the 
# result_partN.h5 file where N is the MPI rank id. 
#
# Usage: timestamp_sort_h5 
# Parameters:
#   - in-h5:            path to unsorted hdf5 file.
#   - out-h5:           path to the new sorted output hdf5 file.
#   - dask-chunk-size:  (Dask client) no. of rows for each data chunk.
#                       Default is 100000000 for 1-D array for float64.
#                       This is 8*100000000/1e9 = 0.8GB per core. The
#                       guideline from dask is ~ 100MB - 1GB.
#   - dask-n-procs:     (Dask client) no. of processes for Dask clients
#                       Default is 100. Max is no. of cores per node.
#   - dask-n-jobs:      (Dask client) for scaling beyond one node. 
#                       Default is 1.
#   - data-sort-batch_size: (MPI) no. of indices per batch
#                       Default is 10000000.
#   - data-sort-n-ranks:(MPI) no. of MPI ranks. Each receives batch_size data
#                       at a time and forks n_procs for Dask slicing.
#                       Default is 15.
####################################################################

from typing import List
import json
import typer
from psana.app.timestamp_sort_h5.sort_ts import TsSort
import time


def main(in_h5: str,
        out_h5: str,
        dask_chunk_size: int = 10000000, 
        dask_n_procs: int = 100,
        dask_n_jobs: int = 1,
        data_sort_batch_size: int = 10000000,
        data_sort_n_ranks: int = 10,
        ):
    print(f'timestamp_sort_h5:')
    print(f'input:     {in_h5}')
    print(f'output:    {out_h5}')
    print(f'Dask:')
    print(f'chunk_size:{dask_chunk_size} n_procs:{dask_n_procs} n_jobs:{dask_n_jobs}')
    print(f'Data sort:')
    print(f'batch_size:{data_sort_batch_size} n_ranks:{data_sort_n_ranks}')
    print(f'MAIN: start')
    ts_sort = TsSort(in_h5, out_h5,
            dask_chunk_size,
            dask_n_procs,
            dask_n_jobs,
            data_sort_batch_size,
            data_sort_n_ranks)
    t1 = time.time()
    inds_arr = ts_sort.sort()
    print(f'MAIN: sort took {time.time()-t1:.2f}s.')
    t2 = time.time()
    ts_sort.slice_and_write(inds_arr)
    print(f'MAIN: slice and write took {time.time()-t2:.2f}s.')


def start():
    typer.run(main)

if __name__ == "__main__":
    start()
