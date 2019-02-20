from psana.dgrammanager import DgramManager
from psana.event import Event
import numpy as np

class FuzzyEventStore(object):
    """ Contains slow events 
    First implementation focuses on offline event case.
    This assumes that data in *epc.xtc2 are ready for query.
    """

    def __init__(self, fuzzy_file=None, fuzzy_dgrams=None):

        assert fuzzy_file != fuzzy_dgrams

        if fuzzy_file:
            dm = DgramManager(fuzzy_file)
            # Read all events and keep them in a private list
            # assuming fuzzy events can only come from 1 file (index=0)
            self._fuzzy_dgrams = [evt._dgrams[0] for evt in dm]

            # Automatically retrieve timestamps after iteration
            self.timestamps = dm.get_timestamps()
        else:
            self._fuzzy_dgrams = fuzzy_dgrams
            self.timestamps = [d.seq.timestamp() for d in self._fuzzy_dgrams]
        
        self.n_events = len(self._fuzzy_dgrams)
        assert len(self.timestamps) == self.n_events

    def generate(self, evt):
        # Returns the fuzzy event corresponded to the given bigdata event 
        # (use timestamp for matching)
        event_timestamp = np.asarray([evt._timestamp], dtype=np.uint64)
        found_pos = np.searchsorted(self.timestamps, event_timestamp)
        return Event([self._fuzzy_dgrams[found_pos[0]]])


