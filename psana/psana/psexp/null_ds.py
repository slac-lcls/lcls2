"""A data source and run that contain no data.

`psana.DataSource` returns `NullDataSource` to MPI ranks that do not read data (for example the
smalldata server ranks).
"""
from psana.psexp.ds_base import DataSourceBase
from psana.smalldata import SmallData


class NullRun(object):
    """Run stand-in with no data.

    `expt` and `runnum` are None, `epicsinfo` and `detinfo` are empty dicts, and the event and step
    iterators are empty.
    """
    def __init__(self):
        self.expt = None
        self.runnum = None
        self.epicsinfo = {}
        self.detinfo = {}

    def Detector(self, *args):
        """Return None for any arguments."""
        return None

    def events(self):
        """Return an empty iterator."""
        return iter([])

    def steps(self):
        """Return an empty iterator."""
        return iter([])

    def close_shared_memory(self):
        """Do nothing; returns None."""
        return


class NullDataSource(DataSourceBase):
    """Data source for ranks that read no data; `runs()` yields one `NullRun`.

    The constructor passes the keyword arguments to `DataSourceBase.__init__`, creates a
    `SmallData` object from `self.smalldata_kwargs`, calls `setup_psplot_live()` on smalldata rank
    0, and starts the Prometheus client with the MPI world rank. `__del__` stops the Prometheus
    client.
    """
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
        """Yield a single `NullRun`."""
        yield NullRun()

    def is_mpi(self):
        """Return False."""
        return False

    def unique_user_rank(self):
        """NullDataSource is used for srv nodes, therefore not a
        'user'-unique rank."""
        return False

    def is_srv(self):
        """Return True."""
        return True

    def is_bd(self):
        """Return False."""
        return False
