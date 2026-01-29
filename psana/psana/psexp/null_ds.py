from psana.psexp.ds_base import DataSourceBase
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
        # Prepare comms for running SmallData
        self.smalldata_obj = SmallData(**self.smalldata_kwargs)
        if self.smalldata_obj.get_rank() == 0:
            self.setup_psplot_live()
        super()._start_prometheus_client(mpi_rank=self.smalldata_obj.get_world_rank())

    def __del__(self):
        super()._end_prometheus_client()

    def runs(self):
        yield NullRun()

    def is_mpi(self):
        return False

    def unique_user_rank(self):
        """NullDataSource is used for srv nodes, therefore not a
        'user'-unique rank."""
        return False

    def is_srv(self):
        return True

    def is_bd(self):
        return False
