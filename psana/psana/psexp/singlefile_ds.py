from .ds_base import DataSourceBase
from .run import RunSingleFile

class SingleFileDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        expstr = args[0]
        super(SingleFileDataSource, self).__init__(**kwargs)
        self.run_dict[-1] = ([expstr], None)

    class Factory:
        def create(self, *args, **kwargs): return SingleFileDataSource(*args, **kwargs)

    def runs(self):
        for run_no in self.run_dict:
            run = RunSingleFile(self.exp, run_no, self.run_dict[run_no][0], \
                        self.max_events, self.batch_size, self.filter)
            self._configs = run.configs # short cut to config
            yield run

