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
#   - data-sort-batch_size: (MPI) no. of indices per batch
#                       Default is 10000000.
####################################################################

import typer
from psana.app.timestamp_sort_h5.sort_ts import TsSort
import time
import h5py
import numpy as np
import socket

MAX_DATA_SORT_N_RANKS = 10


def main(
    in_h5: str,
    out_h5: str,
    dask_chunk_size: int = 100000000,
    data_sort_batch_size: int = 10000000,
):

    # Set dask_chunk_size to the size of timestamp if timestamp is smaller
    in_f = h5py.File(in_h5, "r")
    ts_len = in_f["timestamp"].shape[0]
    if ts_len < dask_chunk_size:
        dask_chunk_size = ts_len
    # Calculate no. of dask processes needed for n chunks
    dask_n_procs = int(np.ceil(ts_len / dask_chunk_size))
    # Scaling beyond one node if no. of dask processes exceeds no. of cpus per node
    if socket.gethostname().startswith("sdf"):
        n_cpus_per_node = 120
    else:
        n_cpus_per_node = 60
    dask_n_jobs = int(np.ceil(dask_n_procs / n_cpus_per_node))

    # Set data-sort batch_size to the size of timestamp if timestamp is smaller
    if ts_len < data_sort_batch_size:
        data_sort_batch_size = ts_len
    # Calculate no. of mpi ranks needed for n batches (maximum is 10 - this is because
    # each mpi rank will create its own dask cluster so each rank will occupy a full node).
    data_sort_n_ranks = int(np.ceil(ts_len / data_sort_batch_size))
    if data_sort_n_ranks > MAX_DATA_SORT_N_RANKS:
        data_sort_n_ranks = MAX_DATA_SORT_N_RANKS

    print("timestamp_sort_h5:", flush=True)
    print(f"input:     {in_h5}", flush=True)
    print(f"output:    {out_h5}", flush=True)
    print(f"ts_len:    {ts_len}", flush=True)
    print("Dask:", flush=True)
    print(
        f"chunk_size:{dask_chunk_size} n_procs:{dask_n_procs} n_jobs:{dask_n_jobs}",
        flush=True,
    )
    print("Data sort:", flush=True)
    print(f"batch_size:{data_sort_batch_size} n_ranks:{data_sort_n_ranks}", flush=True)
    print("MAIN: start", flush=True)
    ts_sort = TsSort(
        in_h5,
        out_h5,
        dask_chunk_size,
        dask_n_procs,
        dask_n_jobs,
        data_sort_batch_size,
        data_sort_n_ranks,
    )
    t1 = time.time()
    inds_arr = ts_sort.sort()
    print(f"MAIN: sort took {time.time()-t1:.2f}s.", flush=True)
    t2 = time.time()
    ts_sort.slice_and_write(inds_arr)
    print(f"MAIN: slice and write took {time.time()-t2:.2f}s.")


def start():
    typer.run(main)


if __name__ == "__main__":
    start()
