import sys, os, glob
from psana.detector.detector import Detector
from psana.psexp.run import Run
from psana.psexp.node import analyze
from psana.psexp.tools import MpiComm, DataSourceHelper

class DataSource(object):
    """ Read XTC files  """ 
    def __init__(self, expstr, filter=0, batch_size=1, max_events=0):
        """Initializes datasource.
        
        Keyword arguments:
        expstr     -- experiment string (eg. exp=xpptut13:run=1) or 
                      a file or list of files (eg. 'data.xtc' or ['data0.xtc','dataN.xtc'])
        batch_size -- length of batched offsets
        max_events -- no. of maximum events
        """
        self.filter = filter
        assert batch_size > 0
        self.batch_size = batch_size
        self.max_events = max_events
        
        self.mpi = MpiComm()
        DataSourceHelper(expstr, self) # setup dgrammanger, configs, & calib.
        if self.nodetype == 'bd':
            self.Detector = Detector(self.configs, calib=self.calib) 

    def runs(self): 
        nruns = 1
        for run_no in range(nruns):
            yield Run(self)

    def events(self): 
        for run in self.runs():
            for evt in run.events(): yield evt

    @property
    def _configs(self):
        assert len(self.configs) > 0
        return self.configs

    def analyze(self, **kwargs):
        analyze(self, **kwargs)
