import os
from psana import utils
logger = utils.Logger()

def get_dask_client(n_procs, 
        partition='milano', 
        n_jobs=1, 
        memory='512GB', 
        local_directory=None):
    # TODO: make local_directory default to output directory (dask_scratch)
    # also check how dask cleans up its tmp dir try cluster.close()
    # SBATCH_PARTITION
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, progress
    account = os.environ.get('SLURM_JOB_ACCOUNT', '')
    if not account:
        raise "Account is not available for dask client."

    if local_directory is None:
        local_directory = os.environ.get("SCRATCH","./")

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
    return client, cluster

def create_virtual_dataset(in_h5fname, out_h5fname, n_files):
    """ Creates virtual dataset with out_h5fname main and N-part sub h5 files.
    
    Each *part_n.h5 file contains batch_size events sent by rank 0.
    For unaligned data (pytables array with size != ts_len or size = 0) or scalar
    data (pytables type UnImplemented), we add them to the main out_h5fname h5 file
    directly prior to joining the file.
    """
    import h5py
    import tables as tb
    
    # Open the original input h5 to determine the layout shapes and traverse (pytable)
    in_f = h5py.File(in_h5fname, 'r')
    ts_len = in_f['timestamp'].shape[0]
    in_t = tb.open_file(in_h5fname, 'r')

    # Get the output path and basename for _partN.h5 vds
    out_dir = os.path.dirname(out_h5fname)
    out_basename = os.path.splitext(os.path.basename(out_h5fname))[0]
    part_fnames = [os.path.join(out_dir, f'{out_basename}_part{i}.h5') for i in range(n_files)]
    part_files = [h5py.File(filename, 'r') for filename in part_fnames]
    
    # Obtain the layouts for aligned data and creates datasets for unaligned and scalar data
    layouts = {}
    out_f = h5py.File(out_h5fname, 'w', libver='latest')
    for group in in_t.walk_groups("/"):
        for array in in_t.list_nodes(group):
            flag_unaligned = True
            entry_key = str(array).split()[0][1:]
            if isinstance(array, (tb.array.Array, )):
                if len(in_f[entry_key].shape) > 0:
                    if in_f[entry_key].shape[0] == ts_len:
                        layouts[entry_key] = h5py.VirtualLayout(shape=in_f[entry_key].shape, dtype=in_f[entry_key].dtype)
                        flag_unaligned = False
            if flag_unaligned:
                val = None
                if isinstance(array, (tb.array.Array, )):
                    # Check for empty array (scalar type)
                    if len(in_f[entry_key].shape) > 0:
                        val = in_f[entry_key][:]
                    else:
                        val = in_f[entry_key][...]
                elif isinstance(array, (tb.unimplemented.UnImplemented, )):
                    val = in_f[entry_key][...]
                else:
                    logger.debug(f'Warning: found {entry_key} with unsupported data type {type(array)}. This dataset will not be included in the new sorted h5 output file.')
                if val is not None:
                    logger.debug(f'Create {entry_key} dataset for unaligned/scalar data type')
                    out_f.create_dataset(entry_key, data=val)


    # Creates virtual dataset by joining all part files for each entry_key for aligned data
    for group in in_t.walk_groups("/"):
        for array in in_t.list_nodes(group):
            if isinstance(array, (tb.array.Array, )):
                entry_key = str(array).split()[0][1:]
                logger.debug(f'Create {entry_key} virtual dataset')
                if len(in_f[entry_key].shape) > 0:
                    if in_f[entry_key].shape[0] == ts_len:
                        st,en = (0,0)
                        for i, filename in enumerate(part_fnames):
                            vsource = h5py.VirtualSource(filename, entry_key, shape=part_files[i][entry_key].shape)
                            en = st + part_files[i][entry_key].shape[0]
                            logger.debug(f'  part{i} shape:{part_files[i][entry_key].shape} {st=} {en=}')
                            layouts[entry_key][st:en] = vsource
                            st = en 
                        out_f.create_virtual_dataset(entry_key, layouts[entry_key],)

    # Close input h5 file
    for i in range(n_files):
        part_files[i].close()
    in_f.close()
    out_f.close()
