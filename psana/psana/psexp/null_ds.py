import os
from psana.psexp import Run, mode, DataSourceBase


class NullDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(NullDataSource, self).__init__(**kwargs)
        self._setup_xtcs()

    def runs(self):
        run = Run(self.dsparms)

        # create a run with no events
        # the following iterator exits immediately
        def events():
            return iter(())

        run.events = events

        yield run
        
