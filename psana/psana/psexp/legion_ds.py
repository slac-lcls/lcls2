from .ds_base import DataSourceBase
from psana.psexp.run import RunLegion

class LegionDataSource(DataSourceBase):
    def __init__(self, *args, **kwargs):
        expstr = args[0]
        super().__init__(**kwargs)
        self.exp, self.run_dict = super().parse_expstr(expstr)

    class Factory:
        def create(self, *args, **kwargs): return LegionDataSource(*args, **kwargs)

    def runs(self):
        for run_no in self.run_dict:
             yield RunLegion(self.exp, run_no, self.run_dict[run_no][0], \
                        self.run_dict[run_no][1], \
                        self.max_events, self.batch_size, self.filter)
