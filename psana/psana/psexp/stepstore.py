from psana.event import Event
from psana.psexp.packet_footer import PacketFooter
import numpy as np
from collections import defaultdict
import os

class StepManager(object):
    """ Store list of Step dgrams, timestamps, and variables """
    
    def __init__(self, config, step_name):
        self.config = config
        self.step_name = step_name
        self.dgrams = []
        self.timestamps = []
        self.n_items = 0
        self._init_step_variables()

    def _init_step_variables(self):
        """ From the given config, build a list of keywords from
        config.software.step_name.[alg].[] fields.
        
        If the given config does not have attribute step_name in
        the software field, then this is an empty StepManager object.
        The step_list of StepStore still has this empty StepManager
        as a place holder to maintain the order of input smd files.
        """
        self.step_variables = {}
        if hasattr(self.config.software, self.step_name):
            steps = getattr(self.config.software, self.step_name)[0]  # only supporting segment 0
            self.algs = list(vars(steps).keys())
            self.algs.remove('dettype')
            self.algs.remove('detid')
            for alg in self.algs:
                self.step_variables[alg] = list(vars(getattr(steps, alg)))
                self.step_variables[alg].remove('version')
                self.step_variables[alg].remove('software')

    def add(self, d):
        self.dgrams.append(d)
        self.timestamps.append(d.seq.timestamp())
        self.n_items += 1
    
    def alg_from_variable(self, variable_name):
        """ Returns algorithm name from the given step variable. """
        for key, val in self.step_variables.items():
            if variable_name in val:
                return key
        return None

    def is_empty(self):
        return self.step_variables

class StepStore(object):
    """ Manages Step data 
    Takes list of memoryviews Step data and update the store."""

    def __init__(self, configs, step_name):
        """ Builds store with the given Step config."""
        self.n_files = 0
        self.step_managers = []
        self.step_variables = defaultdict(list)
        self.step_name = step_name
        if configs:
            self.n_files = len(configs)
            self.step_managers = [StepManager(config, step_name) for config in configs]

            for step in self.step_managers:
                for key, val in step.step_variables.items(): 
                    self.step_variables[key] += val
            
            self.step_info = []
            for key, val in self.step_variables.items():
                val.sort()
                for v in val:
                    self.step_info.append((v, key))
    
    def alg_from_variable(self, variable_name):
        """ Returns algorithm name from the given step variable. """
        for key, val in self.step_variables.items():
            if variable_name in val:
                return key
        return None
    
    def add_to(self, dgram, step_manager_idx):
        self.step_managers[step_manager_idx].add(dgram)

    def dgrams(self, from_pos=0, scan=True):
        """ Generates list of dgrams with 0 as last item. """  
        if scan:
            cn_dgrams = 0
            for sm in self.step_managers:
                for dgram in sm.dgrams:
                    cn_dgrams += 1
                    if cn_dgrams < from_pos: 
                        continue
                    yield dgram
    
    def values(self, events, step_variable):
        """ Returns values of the step_variable for the given events.

        First search for step file that has this variable (return algorithm e.g.
        fast/slow) then for that step file, locate position of step dgram that
        has ts_step <= ts_evt. If the dgram at found position has the algorithm
        then returns the value, otherwise keeps searching backward until 
        PS_N_step_SEARCH_STEPS is reached."""
        
        PS_N_STEP_SEARCH_STEPS = int(os.environ.get("PS_N_STEP_SEARCH_STEPS", "10"))
        step_values = []
        for i, step in enumerate(self.step_managers):
            alg = step.alg_from_variable(step_variable)
            if alg: 
                event_timestamps = np.asarray([evt.timestamp for evt in events], dtype=np.uint64)

                found_positions = np.searchsorted(step.timestamps, event_timestamps)
                found_positions -= 1 # return the step event before the found position.
                for pos in found_positions:
                    val = None
                    for p in range(pos, pos - PS_N_STEP_SEARCH_STEPS, -1):
                        if p < 0:
                            break
                        steps = getattr(step.dgrams[p], self.step_name)[0]
                        if hasattr(steps, alg):
                            val = getattr(getattr(steps, alg), step_variable)
                            break
                    step_values.append(val)

                break
        
        return step_values
    

