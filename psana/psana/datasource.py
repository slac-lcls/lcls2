import os
from psana.psexp.tools import mode
world_size = 1
if mode == 'mpi':
    from mpi4py import MPI
    world_size = MPI.COMM_WORLD.Get_size()
    rank       = MPI.COMM_WORLD.Get_rank()
    
    # set a unique jobid (rank 0 process id) for prometheus client
    if rank == 0:
        prometheus_jobid = os.getpid()
    else:
        prometheus_jobid = None
    prometheus_jobid = MPI.COMM_WORLD.bcast(prometheus_jobid, root=0)
    os.environ['PS_PROMETHEUS_JOBID'] = str(prometheus_jobid)
else:
    os.environ['PS_PROMETHEUS_JOBID'] = str(os.getpid())


class InvalidDataSource(Exception): pass

from psana.psexp.serial_ds     import SerialDataSource
from psana.psexp.singlefile_ds import SingleFileDataSource
from psana.psexp.shmem_ds      import ShmemDataSource
from psana.psexp.legion_ds     import LegionDataSource
from psana.psexp.null_ds       import NullDataSource

def DataSource(*args, **kwargs):
    args = tuple(map(str, args)) # Hack: workaround for unicode and str being different types in Python 2

    # ==== shared memory ====
    if 'shmem' in kwargs:

        if world_size > 1:

            PS_SRV_NODES = int(os.environ.get('PS_SRV_NODES', '0'))
            if PS_SRV_NODES == world_size:
                raise RuntimeError('All allocated cores are smalldata servers '
                                   '(%d of %d)!' % (PS_SRV_NODES, world_size))

            world_group  = MPI.COMM_WORLD.Get_group()
            server_group = world_group.Incl(range(PS_SRV_NODES))
            client_group = world_group.Excl(range(PS_SRV_NODES))
            smalldata_kwargs = {'server_group': server_group,
                                'client_group': client_group}
            kwargs['smalldata_kwargs'] = smalldata_kwargs

            # create NullDataSource for ranks 0...PS_SRV_NODES-1
            # all others are normal ShmemDataSources
            if rank < PS_SRV_NODES:
                return NullDataSource(*args, **kwargs)
            else:
                return ShmemDataSource(*args, **kwargs)

        else: # world_size == 1
            return ShmemDataSource(*args, **kwargs)


    # ==== from experiment directory ====
    elif 'exp' in kwargs: # experiment string - assumed multiple files

        if mode == 'mpi':
            if world_size == 1:
                return SerialDataSource(*args, **kwargs)
            else:
               
                # >> these lines are here to AVOID initializing node.comms
                #    that class instance sets up the MPI environment, which
                #    is global... and can interfere with other uses of MPI
                #    (particularly shmem). Therefore we wish to isolate it
                #    as much as possible.
                from psana.psexp.node   import Communicators
                from psana.psexp.mpi_ds import MPIDataSource
                comms = Communicators()

                smalldata_kwargs = {'server_group' : comms.srv_group(),
                                    'client_group' : comms.bd_group()}
                kwargs['smalldata_kwargs'] = smalldata_kwargs

                if comms._nodetype in ['smd0', 'smd', 'bd']:
                    return MPIDataSource(comms, *args, **kwargs)
                else:
                    return NullDataSource(*args, **kwargs)

        elif mode == 'legion':
            return LegionDataSource(*args, **kwargs)

        elif mode == 'none':
            return SerialDataSource(*args, **kwargs)

        else:
            raise InvalidDataSource("Incorrect mode. DataSource mode only supports either mpi, legion, or none (non parallel mode).")
    
    # ==== from XTC file(s) ====
    elif 'files' in kwargs: # list of files
        return SingleFileDataSource(*args, **kwargs)


    else:
        raise InvalidDataSource("Expected keyword(s) not found. DataSource requires exp, shmem, or files keywords.")


