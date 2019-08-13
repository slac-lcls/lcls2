from psana.dgram import Dgram
from psana.event import Event
from psana.psexp.packet_footer import PacketFooter
import numpy as np
from collections import defaultdict
import os

class Update(object):
    """ Store list of Update dgrams, timestamps, and variables """
    
    def __init__(self, config, update_name):
        self.config = config
        self.update_name = update_name
        self.dgrams = []
        self.timestamps = []
        self.n_items = 0
        self._init_update_variables()

    def _init_update_variables(self):
        """ From the given config, build a list of keywords from
        config.software.update_name.[alg].[] fields.
        
        If the given config does not have attribute update_name in
        the software field, then this is an empty Update object.
        The update_list of UpdateStore still has this empty Update
        as a place holder to maintain the order of input smd files.
        """
        self.update_variables = {}
        if hasattr(self.config.software, self.update_name):
            updates = getattr(self.config.software, self.update_name)[0]  # only supporting segment 0
            self.algs = list(vars(updates).keys())
            self.algs.remove('dettype')
            self.algs.remove('detid')
            for alg in self.algs:
                self.update_variables[alg] = list(vars(getattr(updates, alg)))
                self.update_variables[alg].remove('version')
                self.update_variables[alg].remove('software')

    def add(self, d):
        self.dgrams.append(d)
        self.timestamps.append(d.seq.timestamp())
        self.n_items += 1
    
    def alg_from_variable(self, variable_name):
        """ Returns algorithm name from the given update variable. """
        for key, val in self.update_variables.items():
            if variable_name in val:
                return key
        return None

    def is_empty(self):
        return self.update_variables

class UpdateStore(object):
    """ Manages Update data 
    Takes list of memoryviews Update data and updates the store."""

    def __init__(self, configs, update_name):
        """ Builds store with the given update config."""
        self.n_files = 0
        self._update_list = []
        self.update_variables = defaultdict(list)
        self.update_name = update_name
        if configs:
            self.n_files = len(configs)
            self._update_list = [Update(config, update_name) for config in configs]

            for update in self._update_list:
                for key, val in update.update_variables.items(): 
                    self.update_variables[key] += val
            
            self.update_info = []
            for key, val in self.update_variables.items():
                val.sort()
                for v in val:
                    self.update_info.append((v, key))
    
    def alg_from_variable(self, variable_name):
        """ Returns algorithm name from the given update variable. """
        for key, val in self.update_variables.items():
            if variable_name in val:
                return key
        return None

    def update(self, views):
        """ Updates the store with new data from list of views. """
        if views:
            for i in range(self.n_files):
                view, update = bytearray(views[i]), self._update_list[i]
                offset = 0
                while offset < memoryview(view).shape[0]:
                    d = Dgram(view=view, config=update.config, offset=offset)
                    if hasattr(d, self.update_name):
                        update.add(d)
                    offset += d._size
                
    def dgrams(self, from_pos=0, scan=True):
        """ Generates list of dgrams with 0 as last item. """  
        if scan:
            cn_dgrams = 0
            for update in self._update_list:
                for dgram in update.dgrams:
                    if cn_dgrams < from_pos: 
                        cn_dgrams += 1
                        continue
                    cn_dgrams += 1
                    yield dgram
        yield 0
    
    def values(self, events, update_variable):
        """ Returns values of the update_variable for the given events.

        First search for update file that has this variable (return algorithm e.g.
        fast/slow) then for that update file, locate position of update dgram that
        has ts_update <= ts_evt. If the dgram at found position has the algorithm
        then returns the value, otherwise keeps searching backward until 
        PS_N_UPDATE_SEARCH_STEPS is reached."""
        
        PS_N_UPDATE_SEARCH_STEPS = int(os.environ.get("PS_N_UPDATE_SEARCH_STEPS", "10"))
        update_values = []
        for i, update in enumerate(self._update_list):
            alg = update.alg_from_variable(update_variable)
            if alg: 
                event_timestamps = np.asarray([evt.timestamp for evt in events], dtype=np.uint64)

                found_positions = np.searchsorted(update.timestamps, event_timestamps)
                found_positions -= 1 # return the update event before the found position.
                for pos in found_positions:
                    val = None
                    for p in range(pos, pos - PS_N_UPDATE_SEARCH_STEPS, -1):
                        if p < 0:
                            break
                        updates = getattr(update.dgrams[p], self.update_name)[0]
                        if hasattr(updates, alg):
                            val = getattr(getattr(updates, alg), update_variable)
                            break
                    update_values.append(val)

                break
        
        return update_values
    

