import os
from psana.psexp import Run, mode, DataSourceBase


class NullDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(NullDataSource, self).__init__(**kwargs)
        self.exp, self.run_dict = self._setup_xtcs()

    def runs(self):
        for run_no in self.run_dict:
            run = Run(self.exp, run_no, 
                      max_events=0, 
                      batch_size=self.batch_size,
                      filter_callback=self.filter,
                      prom_man=self.prom_man) 

            # create a run with no events
            # the following iterator exits immediately
            def events():
                return iter(())

            run.events = events

            yield run
        
