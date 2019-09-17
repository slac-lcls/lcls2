from .ds_base import DataSourceBase
from psana.psexp.run import RunSerial

class SerialDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(SerialDataSource, self).__init__(**kwargs)
        self.exp, self.run_dict = self._setup_xtcs()

    def runs(self):
        for run_no in self.run_dict:
            yield RunSerial(self.exp, run_no, self.run_dict[run_no], \
                        max_events=self.max_events, batch_size=self.batch_size, \
                        filter_callback=self.filter)
