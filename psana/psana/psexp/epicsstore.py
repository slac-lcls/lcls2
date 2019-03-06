from psana.dgram import Dgram
from psana.event import Event
from psana.psexp.packet_footer import PacketFooter
import numpy as np

class EpicsStore(object):
    """ Manages Epics data 
    Takes a view of Epics data and updates the store."""
    _epics_dgrams = []
    timestamps = []
    buf = bytearray() # keeps remaining data of view for each update

    def __init__(self):
        pass

    def force_update(self, epics_events):
        if epics_events:
            self._epics_dgrams = [evt._dgrams[0] for evt in epics_events]
            self.timestamps = [evt._timestamp for evt in epics_events]
            self.buf = bytearray()
            self.n_events = len(epics_events)

    def update(self, view, config, min_ts=None):
        """ Updates the store with new data from view.
        If min_ts is given, remove obsolete timestamps from memory."""
        if min_ts and self.timestamps:
            # Look for the closest timestamp with value < min_ts
            ts_difs = np.asarray(self.timestamps) -  min_ts
            found_pos = np.argmin(np.absolute(ts_difs[ts_difs <= 0])) 
            del self._epics_dgrams[:found_pos]
            del self.timestamps[:found_pos]

        offset = 0
        mmr_view = memoryview(self.buf + view)
        while offset < mmr_view.shape[0]:
            d = Dgram(view=mmr_view, config=config, offset=offset)
            
            # check if this is a broken dgram (not enough data in buffer)
            if offset + d._size > mmr_view.shape[0]:
                break
            
            self._epics_dgrams.append(d)
            self.timestamps.append(d.seq.timestamp())

            offset += d._size
        
        if offset < mmr_view.shape[0]:
            self.buf = mmr_view[offset:].tobytes() # copy remaining data to the beginning of buffer
        self.n_events = len(self._epics_dgrams)

    def _checkout(self, event_timestamps):
        if not self.timestamps:
            return None

        found_pos = np.searchsorted(self.timestamps, event_timestamps)
        
        # Returns last epics event for all newer events
        found_pos[found_pos == self.n_events] = self.n_events - 1

        epics_events = [ Event( [self._epics_dgrams[pos]] ) for pos in found_pos ] 
        return epics_events

    def checkout_by_events(self, events):
        """ Returns epics events corresponded to the given bigdata events 
        (use timestamp for matching). """
        event_timestamps = np.asarray([evt._timestamp for evt in events], dtype=np.uint64)
        return self._checkout(event_timestamps)

    def checkout_by_timestamps(self, event_timestamps):
        """ Returns epics events matched with the given list of timstamps."""
        return self._checkout(event_timestamps)

    def checkout_by_ts_range(self, min_ts, max_ts, to_bytes=False):
        """ Returns epics events within the given range of min - max timestamps."""
        if not self.timestamps:
            if to_bytes:
                return bytearray()
            else:
                return None

        ts = np.asarray(self.timestamps)
        min_difs = ts - min_ts
        max_difs = ts - max_ts
        min_pos = np.argmin(np.absolute(min_difs[min_difs <= 0])) 
        max_pos = np.argmin(np.absolute(max_difs[max_difs <= 0])) 
        epics_events = [ Event( [self._epics_dgrams[pos]] ) for pos in \
                range(min_pos, max_pos+1) ]

        if not to_bytes:
            return epics_events
        else:
            return self._to_bytes(epics_events)

    def _to_bytes(self, epics_events):
        if not epics_events:
            return bytearray()

        pf = PacketFooter(len(epics_events))
        view = bytearray()
        for i, evt in enumerate(epics_events):
            event_bytes = evt._to_bytes()
            view.extend(event_bytes)
            pf.set_size(i, memoryview(event_bytes).shape[0])

        view.extend(pf.footer)
        return view
        
    def _from_bytes(self, epics_config, epics_bytes):
        if not epics_bytes:
            return None

        pf = PacketFooter(view=epics_bytes)
        event_bytes_list = pf.split_packets()
        epics_events = [Event._from_bytes([epics_config], event_bytes) \
                for event_bytes in event_bytes_list]
        return epics_events
        
