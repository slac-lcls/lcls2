import os

def get_dask_client(n_procs, 
        partition='milano', 
        n_jobs=1, 
        memory='512GB', 
        local_directory='/sdf/data/lcls/drpsrcf/ffb/users/monarin/tmp'):
    # TODO: make local_directory default to output directory (dask_scratch)
    # also check how dask cleans up its tmp dir try cluster.close()
    # SBATCH_PARTITION
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, progress
    account = os.environ.get('SLURM_JOB_ACCOUNT', '')
    if not account:
        raise "Account is not available for dask client."

    cluster = SLURMCluster(
        queue=partition,
        account=account,
        local_directory=local_directory,  # Local disk space for workers to use
        cores=n_procs, 
        memory=memory,                    # Memory for all processes
        
    )
    cluster.scale(jobs=n_jobs)
    cluster.job_script()
    client = Client(cluster)
    return client

def create_virtual_dataset(in_h5fname, out_h5fname, n_files):
    import h5py

    in_f = h5py.File(in_h5fname, 'r')

    # Get the output path and basename for _partN.h5 vds
    out_dir = os.path.dirname(out_h5fname)
    out_basename = os.path.splitext(os.path.basename(out_h5fname))[0]
    part_fnames = [os.path.join(out_dir, f'{out_basename}_part{i}.h5') for i in range(n_files)]
    part_files = [h5py.File(filename, 'r') for filename in part_fnames]

    layouts = {}
    for entry_key in in_f.keys():
        layouts[entry_key] = h5py.VirtualLayout(shape=in_f[entry_key].shape, dtype=in_f[entry_key].dtype)

    with h5py.File(out_h5fname, 'w', libver='latest') as out_f:
        for entry_key in in_f.keys():
            st,en = (0,0)
            print(f'Creating virtual source for {entry_key}')
            for i, filename in enumerate(part_fnames):
                vsource = h5py.VirtualSource(filename, entry_key, shape=part_files[i][entry_key].shape)
                en = st + part_files[i][entry_key].shape[0]
                print(f'  part{i} shape:{part_files[i][entry_key].shape} {st=} {en=}')
                layouts[entry_key][st:en] = vsource
                st = en 
            out_f.create_virtual_dataset(entry_key, layouts[entry_key], fillvalue=0)

    # Close input h5 file
    for i in range(n_files):
        part_files[i].close()
    in_f.close()
