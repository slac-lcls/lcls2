from .ds_base import DataSourceBase
from psana.psexp.run import RunLegion

class LegionDataSource(DataSourceBase):
    def __init__(self, *args, **kwargs):
        super(LegionDataSource, self).__init__(**kwargs)
        self.exp, self.run_dict = super(LegionDataSource, self).setup_xtcs()

    class Factory:
        def create(self, *args, **kwargs): return LegionDataSource(*args, **kwargs)

    def runs(self):
        for run_no in self.run_dict:
             yield RunLegion(self.exp, run_no, self.run_dict[run_no], \
                        max_events=self.max_events, batch_size=self.batch_size, \
                        filter_callback=self.filter)
