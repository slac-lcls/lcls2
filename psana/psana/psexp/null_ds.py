import os
from psana.psexp import Run, mode, DataSourceBase
from psana.psexp.mpi_ds import NullRun
from psana.smalldata import SmallData


class NullDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(NullDataSource, self).__init__(**kwargs)
        self._setup_xtcs()
        # prepare comms for running SmallData
        self.smalldata_obj = SmallData(**self.smalldata_kwargs)

    def runs(self):
        yield NullRun() 

        
