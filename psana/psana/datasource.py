import os
from psana.psexp.tools import mode
world_size = 1
if mode == 'mpi':
    from mpi4py import MPI
    world_size = MPI.COMM_WORLD.Get_size()

class InvalidDataSource(Exception): pass

from psana.psexp.serial_ds     import SerialDataSource
from psana.psexp.mpi_ds        import MPIDataSource
from psana.psexp.singlefile_ds import SingleFileDataSource
from psana.psexp.shmem_ds      import ShmemDataSource
from psana.psexp.legion_ds     import LegionDataSource
from psana.psexp.null_ds       import NullDataSource


def DataSource(*args, **kwargs):
    args = tuple(map(str, args)) # Hack: workaround for unicode and str being different types in Python 2

    if 'shmem' in kwargs:
        return ShmemDataSource(*args, **kwargs)

    elif 'exp' in kwargs: # experiment string - assumed multiple files

        if mode == 'mpi':
            if world_size == 1:
                return SerialDataSource(*args, **kwargs)
            else:
                from psana.psexp.node import comms
                if comms._nodetype in ['smd0', 'smd', 'bd']:
                    return MPIDataSource(*args, **kwargs)
                else:
                    return NullDataSource(*args, **kwargs)

        elif mode == 'legion':
            return LegionDataSource(*args, **kwargs)

        else:
            raise InvalidDataSource("Incorrect mode. DataSource mode only supports either mpi or legion.")
    
    elif 'files' in kwargs: # list of files
        return SingleFileDataSource(*args, **kwargs)

    else:
        raise InvalidDataSource("Expected keyword(s) not found. DataSource requires exp, shmem, or files keywords.")

