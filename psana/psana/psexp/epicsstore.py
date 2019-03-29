from psana.dgram import Dgram
from psana.event import Event
from psana.psexp.packet_footer import PacketFooter
import numpy as np
from collections import defaultdict
import os

class Epics(object):
    """ Store list of Epics dgrams, timestatmps, and variables """
    
    def __init__(self, config):
        self.config = config
        self.dgrams = []
        self.timestamps = []
        self.buf = bytearray() # keeps remaining data of each Epics file
        self.offset = 0
        self.n_items = 0
        self._init_epics_variables()

    def _init_epics_variables(self):
        """ From the given config, build a list of keywords from
        config.software.xppepics.[alg:fast/slow].[] fields."""
        algs = vars(self.config.xppepics[0])
        self.epics_variables = {}
        for alg in algs:
            self.epics_variables[alg] = list(eval("vars(self.config.software.xppepics.%s)"%alg))
            self.epics_variables[alg].remove('version')
            self.epics_variables[alg].remove('software')

    def add(self, d):
        self.dgrams.append(d)
        self.timestamps.append(d.seq.timestamp())
        self.n_items += 1
    
    def alg_from_variable(self, variable_name):
        """ Returns algorithm name from the given epics variable. """
        for key, val in self.epics_variables.items():
            if variable_name in val:
                return key
        return None

class EpicsStore(object):
    """ Manages Epics data 
    Takes list of memoryviews Epics data and updates the store."""

    def __init__(self, configs):
        """ Builds store with the given epics config."""
        self.n_files = 0
        self._epics_list = []
        self.epics_variables = defaultdict(list)
        if configs:
            self.n_files = len(configs)
            self._epics_list = [Epics(config) for config in configs]

            # Collects epics variables from all epics files
            for epics in self._epics_list:
                for key, val in epics.epics_variables.items(): 
                    self.epics_variables[key] += val
            
            self.epics_info = []
            for key, val in self.epics_variables.items():
                val.sort()
                for v in val:
                    self.epics_info.append((v, key))
    
    def alg_from_variable(self, variable_name):
        """ Returns algorithm name from the given epics variable. """
        for key, val in self.epics_variables.items():
            if variable_name in val:
                return key
        return None

    def update(self, views):
        """ Updates the store with new data from list of views. """
        if views:
            for i in range(self.n_files):
                view, epics = views[i], self._epics_list[i]
                mmr_view = memoryview(epics.buf + view)
                while epics.offset < mmr_view.shape[0]:
                    d = Dgram(view=mmr_view, config=epics.config, offset=epics.offset)
                    
                    # check if this is a broken dgram (not enough data in buffer)
                    if epics.offset + d._size > mmr_view.shape[0]:
                        break
                    
                    epics.add(d)
                    epics.offset += d._size
                
                if epics.offset < mmr_view.shape[0]:
                    epics.buf = mmr_view[epics.offset:].tobytes() # copy remaining data to the beginning of buffer

    def values(self, events, epics_variable):
        """ Returns values of the epics_variable for the given events.

        First search for epics file that has this variable (return algorithm e.g.
        fast/slow) then for that epics file, locate position of epics dgram that
        has ts_epics <= ts_evt. If the dgram at found position has the algorithm
        then returns the value, otherwise keeps searching backward until 
        N_EPICS_SEARCH_STEPS is reached."""
        
        N_EPICS_SEARCH_STEPS = int(os.environ.get("N_EPICS_SEARCH_STEPS", "10"))
        epics_values = []
        for i, epics in enumerate(self._epics_list):
            alg = epics.alg_from_variable(epics_variable)
            if alg: 
                event_timestamps = np.asarray([evt._timestamp for evt in events], dtype=np.uint64)
                found_positions = np.searchsorted(epics.timestamps, event_timestamps)
                found_positions[found_positions == epics.n_items] = epics.n_items - 1
                for pos in found_positions:
                    val = None
                    for p in range(pos, pos - N_EPICS_SEARCH_STEPS, -1):
                        if p < 0:
                            break
                        if hasattr(epics.dgrams[p].xppepics[0], alg):
                            val = eval("getattr(epics.dgrams[%d].xppepics[0].%s, '%s')"%(p, alg, epics_variable))
                            break
                    epics_values.append(val)

                break
        
        return epics_values

    def _checkout(self, event_timestamps):
        """ Builds an epics dictionary using data from all epics files
        with matching timestamps."""
        if not self.n_files:
            return None
        
        epics_dicts = [dict() for i in range(self.n_files)] # keeps key-val for each event
        for epics in self._epics_list:
            found_pos = np.searchsorted(epics.timestamps, event_timestamps)
        
            # Returns last epics event for all newer events
            found_pos[found_pos == epics.n_items] = epics.n_items - 1
            for i, pos in enumerate(found_pos):
                algs = vars(epics.config.xppepics[0])
                for alg in algs:
                    if alg in vars(epics.dgrams[pos].xppepics[0]):
                        epics_dicts[i].update(eval("vars(epics.dgrams[%d].xppepics[0].%s)"%(pos, alg)))
        
        return epics_dicts

    def checkout_by_events(self, events):
        """ Returns epics events corresponded to the given bigdata events 
        (use timestamp for matching). """
        event_timestamps = np.asarray([evt._timestamp for evt in events], dtype=np.uint64)
        return self._checkout(event_timestamps)

    def checkout_by_timestamps(self, event_timestamps):
        """ Returns epics events matched with the given list of timstamps."""
        return self._checkout(event_timestamps)

