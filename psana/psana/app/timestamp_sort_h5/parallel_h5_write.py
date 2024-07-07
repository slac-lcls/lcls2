from mpi4py import MPI
import numpy as np
import h5py
import time
import os
import dask.array as da
from psana.app.timestamp_sort_h5.utils import get_dask_client 
import tables as tb


# Get and merge parent comm to get the common comm
comm = MPI.Comm.Get_parent()
common_comm=comm.Merge(True)


# Get input parameters from root
input_dict=common_comm.bcast(0, root=0)
n_procs = input_dict['n_procs']
chunk_size = input_dict['chunk_size']
in_h5fname = input_dict['in_h5fname']
out_h5fname = input_dict['out_h5fname']


# Get the output path and basename for _partN.h5 vds
out_dir = os.path.dirname(out_h5fname)
out_basename = os.path.splitext(os.path.basename(out_h5fname))[0]


# Setup cluster for dask
st = time.time()
client, cluster = get_dask_client(n_procs)


t0 = time.time()

# Use pytables to access h5 tree keys and h5py for data
in_pytbl = tb.open_file(in_h5fname, 'r')
in_f = h5py.File(in_h5fname, 'r')
ts_len = in_f['timestamp'].shape[0]
chunk_size = (chunk_size,)
print (f"WRITER RANK:{common_comm.Get_rank()} start slice-and-write ts_len:{ts_len}")

# Start receving data
while True:
    common_comm.Send(np.array([common_comm.Get_rank()], dtype='i'), dest=0)
    info = MPI.Status()
    common_comm.Probe(source=0, status=info)
    count = info.Get_elements(MPI.INT64_T)
    if count > 0:
        data = np.empty(count, dtype=np.int64)
        common_comm.Recv(data, source=0)
    else:
        data = bytearray()
        common_comm.Recv(data, source=0)
        break
    tag = info.Get_tag()
    print (f"WRITER RANK:{common_comm.Get_rank()} received {data.shape=} {tag=}")

    # Get the ordered indices for faster access
    i_data = np.argsort(data)
    access_indices = data[i_data]

    out_fname = os.path.join(out_dir, f'{out_basename}_part{tag}.h5')
    out_f = h5py.File(out_fname, 'w') 
    for group in in_pytbl.walk_groups("/"):
        for array in in_pytbl.list_nodes(group):
            t1 = time.time()
            if isinstance(array, (tb.array.Array, )):
                # We only sort 'aligned' data (data with the same size as timestamp)
                # TODO: once unaligned data are timestamped, we'll sort them too.
                # TODO: find a better way to get dataset_path (e.g. /grp1/subgrp1/ds_name)
                ds_path = str(array).split()[0][1:]
                if len(in_f[ds_path].shape) > 0:
                    if in_f[ds_path].shape[0] == ts_len:
                        # Dask slicing with ordered indices
                        dask_arr = da.from_array(in_f[ds_path], chunks=chunk_size+in_f[ds_path].shape[1:])
                        access_arr = dask_arr[access_indices].compute()
                        # Reorder the slice back to the original order
                        val = access_arr[np.argsort(i_data)]
                        print(f'WRITER RANK:{common_comm.Get_rank()} ds_path:{ds_path} dask slicing:{time.time()-t1:.2f}s.')
                        out_f.create_dataset(ds_path, data=val)
                        t2 = time.time()
                        print(f'WRITER RANK:{common_comm.Get_rank()} ds_path:{ds_path} writing:{time.time()-t2:.2f}s.')
                else:
                    print(f'WRITER RANK:{common_comm.Get_rank()} Warning: {ds_path} has empty shape')
    out_f.close()


in_f.close()
in_pytbl.close()
client.close()
cluster.close()

en = time.time()
print(f"WRITER RANK:{common_comm.Get_rank()} DONE setup_cluster:{t0-st:.2f}s. slice_and_write: {en-t0:.2f}s. total: {en-st:.2f}s.")
common_comm.Barrier()
common_comm.Abort(1)

