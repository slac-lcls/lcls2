from psana.psexp import DataSourceBase, RunSingleFile

class SingleFileDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(SingleFileDataSource, self).__init__(**kwargs)
        self.exp, self.run_dict = self._setup_xtcs()

    def runs(self):
        for run_no in self.run_dict:
            run = RunSingleFile(self.exp, run_no, self.run_dict[run_no], \
                        max_events=self.max_events, batch_size=self.batch_size, \
                        filter_callback=self.filter, prom_man=self.prom_man)
            self._configs = run.configs # short cut to config
            yield run

