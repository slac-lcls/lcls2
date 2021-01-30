import os
from psana.psexp import Run, mode, DataSourceBase
from psana.smalldata import SmallData

class NullRun(object):
    def __init__(self):
        self.expt = None
        self.runnum = None
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
