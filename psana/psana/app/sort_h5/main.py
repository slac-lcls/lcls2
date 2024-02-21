####################################################################
# A program that takes unsorted hdf5 file and sort the data by its
# timestamp and rewrite all sorted datasets to another hdf5 file.
#
# The output h5file is a virtual datasource that binds all the 
# result_partN.h5 file where N is the MPI rank id. 
#
# Usage: h5sort 
# Parameters:
#   - input_file:  path to unsorted hdf5 file.
#   - output_file: path to the new sorted output hdf5 file.
#   - chunk_size:  (Dask client) no. of rows for each data chunk.
#                  Default is 100000000.
#   - n_procs:     (Dask client) no. of processes for Dask clients
#                  Default is 100. Max is no. of cores per node.
#   - n_jobs:      (Dask client) for scaling beyond one node. 
#                  Default is 1.
#   - batch_size:  (MPI) no. of indices per batch
#                  Default is 10000000.
#   - n_ranks:     (MPI) no. of MPI ranks. Each receives batch_size data
#                  at a time and forks n_procs for Dask slicing.
#                  Default is 15.
####################################################################

from typing import List
import json
import typer
from sort_ts import TsSort
import time


def main(in_h5: str,
        out_h5: str,
        chunk_size: int = 10000000, 
        n_procs: int = 100,
        n_jobs: int = 1,
        batch_size: int = 10000000,
        n_ranks: int = 15,
        debug: bool = False,
        ):
    print(f'sort_h5:')
    print(f'input:     {in_h5}')
    print(f'output:    {out_h5}')
    print(f'chunk_size:{chunk_size} n_procs:{n_procs} n_jobs:{n_jobs}')
    print(f'batch_size:{batch_size} n_ranks:{n_ranks}')
    print(f'MAIN: start')
    ts_sort = TsSort(in_h5, out_h5,
            chunk_size,
            n_procs,
            n_jobs,
            batch_size,
            n_ranks)
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
