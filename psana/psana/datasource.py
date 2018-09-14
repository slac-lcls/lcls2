import sys, os, glob
from psana.detector.detector import Detector
from psana.psexp.run import Run
from psana.psexp.node import analyze
from psana.psexp.tools import MpiComm, DataSourceHelper, datasource_from_id

class DataSource(object):
    """ Read XTC files  """ 
    def __init__(self, expstr, **kwargs):
        """Initializes datasource.
        
        Keyword arguments:
        expstr      -- experiment string (eg. exp=xpptut13:run=1) or 
                       a file or list of files (eg. 'data.xtc' or ['data0.xtc','dataN.xtc'])
        filter      -- filtering callback that handles Event object.
        batch_size  -- length of batched offsets
        max_events  -- no. of maximum events
        det_name    -- detector name used to identify dettype and detid in config
        sel_det_ids -- user-selected detector IDs.
        """
        self.filter = 0
        self.batch_size = 1
        self.max_events = 0
        self.sel_det_ids = []
        self.det_name = None
        if kwargs is not None: 
            keywords = ('filter', 'batch_size', 'max_events', 'det_name', 'sel_det_ids')
            for k in keywords:
                if k in kwargs:
                    setattr(self, k, kwargs[k])
        assert self.batch_size > 0
        
        self.mpi = MpiComm()
        DataSourceHelper(expstr, self) # setup exp, run_dict, nodetype

    def runs(self):
        for run_no in self.run_dict:
            yield Run(self, run_no)

    def events(self): 
        for run in self.runs():
            for evt in run.events(): yield evt

    @property
    def _configs(self):
        assert len(self.configs) > 0
        return self.configs

    @property
    def Detector(self):
        """Creates a detector (must be done inside runs())
        Since configs and calib constants depend on each run, 
        the detector object has to wait until ds is set with
        these two parameters."""
        det = Detector(self.configs, calib=self.calib) 
        return det

    def analyze(self, **kwargs):
        analyze(self, **kwargs)

    def __reduce__(self):
        return (datasource_from_id, (self.id,))
