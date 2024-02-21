import os

def get_dask_client(n_procs, 
        partition='milano', 
        n_jobs=1, 
        memory='512GB', 
        local_directory='/sdf/data/lcls/drpsrcf/ffb/users/monarin/tmp'):
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
