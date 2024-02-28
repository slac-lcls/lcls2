from mpi4py import MPI
import numpy as np
import h5py
import time
import os
import dask.array as da
from utils import get_dask_client 


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
client = get_dask_client(n_procs)


t0 = time.time()


# Access the large h5 and slice with the obtained indices
in_f = h5py.File(in_h5fname, 'r')
chunk_size = (chunk_size,)
da_dict = {}
for key in in_f.keys():
    da_dict[key] = da.from_array(in_f[key], chunks=chunk_size+in_f[key].shape[1:])


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
    for key in in_f.keys():
        t1 = time.time()
        # Dask slicing with ordered indices
        access_arr = da_dict[key][access_indices].compute()
        # Reorder the slice back to the original order
        in_arr = access_arr[np.argsort(i_data)]
        print(f'WRITER RANK:{common_comm.Get_rank()} {key=} dask slicing:{time.time()-t1:.2f}s.')
        t2 = time.time()
        out_f.create_dataset(key, data=in_arr)
        print(f'WRITER RANK:{common_comm.Get_rank()} {key=} writing:{time.time()-t2:.2f}s.')
    out_f.close()


in_f.close()

en = time.time()
print(f"WRITER RANK:{common_comm.Get_rank()} DONE setup_cluster:{t0-st:.2f}s. slice_and_write: {en-t0:.2f}s. total: {en-st:.2f}s.")
common_comm.Barrier()
common_comm.Abort(1)

