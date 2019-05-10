from .ds_base import DataSourceBase
from .run import RunSingleFile

class SingleFileDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        expstr = args[0]
        super(SingleFileDataSource, self).__init__(**kwargs)
        # to support a comma-separated list of files
        files = expstr.split(',')
        for num,file in enumerate(files):
            # cpo isn't sure why we count from -1 here, but
            # keep following Mona's pattern
            self.run_dict[-1*(num+1)] = ([file], None, None)

    class Factory:
        def create(self, *args, **kwargs): return SingleFileDataSource(*args, **kwargs)

    def runs(self):
        for run_no in self.run_dict:
            run = RunSingleFile(self.exp, run_no, self.run_dict[run_no], \
                        max_events=self.max_events, batch_size=self.batch_size, \
                        filter_callback=self.filter)
            self._configs = run.configs # short cut to config
            yield run

