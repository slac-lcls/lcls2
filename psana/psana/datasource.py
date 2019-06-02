import os
from psana.psexp.tools import mode
size = 1
if mode == 'mpi':
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()

from psana.psexp.serial_ds import SerialDataSource
from psana.psexp.mpi_ds import MPIDataSource
from psana.psexp.singlefile_ds import SingleFileDataSource
from psana.psexp.shmem_ds import ShmemDataSource
from psana.psexp.legion_ds import LegionDataSource

class DataSourceFactory:
    factories = {}
    
    @staticmethod
    def createDataSource(id, *args, **kwargs):
        if id not in DataSourceFactory.factories:
            DataSourceFactory.factories[id] = eval(id + '.Factory()')
        return DataSourceFactory.factories[id].create(*args, **kwargs)

def DataSource(*args, **kwargs):
    args = tuple(map(str, args)) # Hack: workaround for unicode and str being different types in Python 2

    if 'shmem' in kwargs:
        return DataSourceFactory.createDataSource('ShmemDataSource', *args, **kwargs)    
    elif 'exp' in kwargs: # experiment string - assumed multiple files
        if mode == 'mpi':
            if size == 1:
                return DataSourceFactory.createDataSource('SerialDataSource', *args, **kwargs)
            else:
                return DataSourceFactory.createDataSource('MPIDataSource', *args, **kwargs)
        elif mode == 'legion':
            return DataSourceFactory.createDataSource('LegionDataSource', *args, **kwargs)
        else:
            raise("Invalid datasource")
    
    elif 'files' in kwargs: # list of files
        return DataSourceFactory.createDataSource('SingleFileDataSource', *args, **kwargs)
    else:
        raise("Invalid datasource")
