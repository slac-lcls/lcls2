import os
from psana.psexp import Run, mode, DataSourceBase
from psana.smalldata import SmallData

class NullRun(object):
    def __init__(self):
        self.expt = None
        self.runnum = None
        self.epicsinfo = {}
        self.detinfo = {}
    def Detector(self, *args):
        return None
    def events(self):
        return iter([])
    def steps(self):
        return iter([])

class NullDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(NullDataSource, self).__init__(**kwargs)
        # prepare comms for running SmallData
        self.smalldata_obj = SmallData(**self.smalldata_kwargs)

    def runs(self):
        yield NullRun()
    
    def is_mpi(self):
        return False

    def unique_user_rank(self):
        """ NullDataSource is used for srv nodes, therefore not a
        'user'-unique rank."""
        return False
