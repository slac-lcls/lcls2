from mpi4py import MPI
import numpy as np
import h5py
import time
import dask.array as da


# Get and merge parent comm to get the common comm
comm = MPI.Comm.Get_parent()
common_comm=comm.Merge(True)


# Setup cluster for dask
st = time.time()
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, progress
partition = 'milano'  # For LCLS II staff
n_procs = 100

cluster = SLURMCluster(
    queue=partition,
    account="lcls:data",
    local_directory='/sdf/data/lcls/drpsrcf/ffb/users/monarin/tmp/',  # Local disk space for workers to use

    # Resources per SLURM job (per node, the way SLURM is configured on Roma)
    # processes=16 runs 16 Dask workers in a job, so each worker has 1 core & 32 GB RAM.
    cores=n_procs, memory='512GB',
    
)
cluster.scale(jobs=1)
cluster.job_script()
client = Client(cluster)
t0 = time.time()
print(f'WRITER RANK:{common_comm.Get_rank()} {client=} setup cluster done in {t0-st:.2f}s.')


# Access the large h5 and slice with the obtained indices
in_f = h5py.File('/sdf/data/lcls/drpsrcf/ffb/users/monarin/h5/mylargeh5.h5', 'r')
chunk_size = (10000000,)
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
        print (f"WRITER RANK:{common_comm.Get_rank()} received KILL")
        break
    tag = info.Get_tag()
    print (f"WRITER RANK:{common_comm.Get_rank()} received {data.shape=} {tag=}")

    # Get the ordered indices for faster access
    i_data = np.argsort(data)
    access_indices = data[i_data]

    out_f = h5py.File(f'/sdf/data/lcls/drpsrcf/ffb/users/monarin/h5/output/result_part{tag}.h5', 'w') 
    for key in in_f.keys():
        t0 = time.monotonic()
        # Dask slicing with ordered indices
        access_arr = da_dict[key][access_indices].compute()
        # Reorder the slice back to the original order
        in_arr = access_arr[np.argsort(i_data)]
        t1 = time.monotonic()
        print(f'WRITER RANK:{common_comm.Get_rank()} {key=} dask slicing:{t1-t0:.2f}s.')
        out_f.create_dataset(key, data=in_arr)
        t2 = time.monotonic()
        print(f'WRITER RANK:{common_comm.Get_rank()} {key=} writing:{t2-t1:.2f}s.')
    out_f.close()


in_f.close()

print(f"WRITER DONE: RANK:{common_comm.Get_rank()}")
#common_comm.Abort(1)

