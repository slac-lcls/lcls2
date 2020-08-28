from psana.psexp import RunLegion, DataSourceBase

class LegionDataSource(DataSourceBase):
    def __init__(self, *args, **kwargs):
        super(LegionDataSource, self).__init__(**kwargs)
        self.exp, self.run_dict = super(LegionDataSource, self)._setup_xtcs()
        super()._start_prometheus_client()

    def runs(self):
        for run_no in self.run_dict:
             yield RunLegion(self.exp, run_no, self.run_dict[run_no], 
                        max_events      = self.max_events, 
                        batch_size      = self.batch_size, 
                        filter_callback = self.filter,
                        prom_man        = self.prom_man)
        super()._end_prometheus_client()
