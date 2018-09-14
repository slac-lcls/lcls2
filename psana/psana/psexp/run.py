from psana.psexp.node import run_node
import numpy as np
from psana.dgrammanager import DgramManager
from psana.psexp.tools import RunHelper

def read_event(ds, run_no, event_type=None):
    """Reads an event.

    Keywords:
    ds         -- DataSource with access to run_dict
    run_no     -- Used as key to access run_dict for xtc_files and smd_files
    event_type -- used to determined config updates or other
                  types of event.
    """
    rank = ds.mpi.rank
    size = ds.mpi.size

    if size == 1:
        return ds.dm       # safe for python2
    else:
        return run_node(ds)

class ConfigUpdate(object):
    """ ConfigUpdate generator """
    def __init__(self, run):
        self.run = run

    def events(self):
        for evt in read_event(self.run.ds, self.run.run_no, event_type="config"):
            yield evt

class Run(object):
    """ Run generator """
    def __init__(self, ds, run_no):
        self.ds = ds
        self.run_no = run_no
        RunHelper(self.ds, self.run_no) # setup per-run dgrammanager and calib dict

    def events(self):
        for evt in read_event(self.ds, self.run_no): yield evt

    def configUpdates(self):
        for i in range(1):
            yield ConfigUpdate(self)

    def run(self):
        """ Returns integer representaion of run no.
        default: (when no run is given) is set to -1"""
        return self.run_no

