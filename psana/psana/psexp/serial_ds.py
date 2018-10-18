from .ds_base import DataSourceBase
from psana.psexp.run import RunSerial

class SerialDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        expstr = args[0]
        super(SerialDataSource, self).__init__(**kwargs)
        self.exp, self.run_dict = self.parse_expstr(expstr)

    class Factory:
        def create(self, *args, **kwargs): return SerialDataSource(*args, **kwargs)

    def runs(self):
        for run_no in self.run_dict:
            yield RunSerial(self.exp, run_no, self.run_dict[run_no][0], \
                        self.max_events, self.filter)
