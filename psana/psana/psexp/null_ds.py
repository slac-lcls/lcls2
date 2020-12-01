import os
from psana.psexp import Run, mode, DataSourceBase
from psana.smalldata import SmallData


class NullDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(NullDataSource, self).__init__(**kwargs)
        # prepare comms for running SmallData
        self.smalldata_obj = SmallData(**self.smalldata_kwargs)

    def runs(self):
        run = Run(self.dsparms)
        def events():
            return iter([])
        setattr(run, "events", events)
        yield run
        
