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

    def _checkout(self, event_timestamps):
        found_pos = np.searchsorted(self.timestamps, event_timestamps)
        
        # Returns last fuzzy event for all newer events
        found_pos[found_pos == self.n_events] = self.n_events - 1

        fuzzy_events = [ Event( [self._fuzzy_dgrams[pos]] ) for pos in found_pos ] 
        return fuzzy_events

    def checkout_by_events(self, events):
        # Returns fuzzy events corresponded to the given bigdata events 
        # (use timestamp for matching)
        event_timestamps = np.asarray([evt._timestamp for evt in events], dtype=np.uint64)
        return self._checkout(event_timestamps)

    def checkout_by_timestamps(self, event_timestamps):
        return self._checkout(event_timestamps)



