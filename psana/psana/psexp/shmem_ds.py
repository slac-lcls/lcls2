from .ds_base import DataSourceBase
from .run import RunShmem

class ShmemDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(ShmemDataSource, self).__init__(**kwargs)
        self.exp, self.run_dict = self.setup_xtcs()

    class Factory:
        def create(self, *args, **kwargs): return ShmemDataSource(*args, **kwargs)

    def runs(self):
        print(self.run_dict)
        for run_no in self.run_dict:
            run = RunShmem(self.exp, run_no, self.run_dict[run_no][0],  \
                        self.max_events, self.batch_size, self.filter, self.tag)
            self._configs = run.configs # short cut to config
            yield run

