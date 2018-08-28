from psana.psexp.node import run_node
import numpy as np

def read_event(ds, event_type=None):
    """Reads an event.

    Keywords:
    ds         -- DataSource with access to the smd or xtc files
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
        for evt in read_event(self.run.ds, event_type="config"):
            yield evt

class Run(object):
    """ Run generator """
    def __init__(self, ds):
        self.ds = ds

    def events(self):
        for evt in read_event(self.ds): yield evt

    def configUpdates(self):
        for i in range(1):
            yield ConfigUpdate(self)

    def run(self):
        """ Returns integer representaion of run no.
        default: (when no run is given) is set to -1"""
        return self.ds.run

