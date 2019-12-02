from psana.psexp.packet_footer import PacketFooter
import numpy as np
from collections import defaultdict
import os

class EnvManager(object):
    """ Store list of Env dgrams, timestamps, and variables """
    
    def __init__(self, config, env_name):
        self.config = config
        self.env_name = env_name
        self.dgrams = []
        self.timestamps = []
        self.n_items = 0
        self._init_env_variables()

    def _init_env_variables(self):
        """ From the given config, build a list of keywords from
        config.software.env_name.[alg].[] fields.
        
        If the given config does not have attribute env_name in
        the software field, then this is an empty EnvManager object.
        The env_list of EnvStore still has this empty EnvManager
        as a place holder to maintain the order of input smd files.
        """
        self.env_variables = {}
        if hasattr(self.config.software, self.env_name):
            envs = getattr(self.config.software, self.env_name)[0]  # only supporting segment 0
            self.algs = list(vars(envs).keys())
            self.algs.remove('dettype')
            self.algs.remove('detid')
            for alg in self.algs:
                self.env_variables[alg] = list(vars(getattr(envs, alg)))
                self.env_variables[alg].remove('version')
                self.env_variables[alg].remove('software')

    def add(self, d):
        self.dgrams.append(d)
        self.timestamps.append(d.seq.timestamp())
        self.n_items += 1
    
    def alg_from_variable(self, variable_name):
        """ Returns algorithm name from the given env variable. """
        for key, val in self.env_variables.items():
            if variable_name in val:
                return key
        return None

    def is_empty(self):
        return self.env_variables

class EnvStore(object):
    """ Manages Env data 
    Takes list of memoryviews Env data and update the store."""

    def __init__(self, configs, env_name):
        """ Builds store with the given Env config."""
        self.n_files = 0
        self.env_managers = []
        self.env_variables = defaultdict(list)
        self.env_name = env_name
        if configs:
            self.n_files = len(configs)
            self.env_managers = [EnvManager(config, env_name) for config in configs]

            for env in self.env_managers:
                for key, val in env.env_variables.items(): 
                    self.env_variables[key] += val
            
            self.env_info = []
            for key, val in self.env_variables.items():
                val.sort()
                for v in val:
                    self.env_info.append((v, key))

    def alg_from_variable(self, variable_name):
        """ Returns algorithm name from the given env variable. """
        for key, val in self.env_variables.items():
            if variable_name in val:
                return key
        return None
    
    def add_to(self, dgram, env_manager_idx):
        self.env_managers[env_manager_idx].add(dgram)

    def dgrams(self, from_pos=0, scan=True):
        """ Generates list of dgrams with 0 as last item. """  
        if scan:
            cn_dgrams = 0
            for sm in self.env_managers:
                for dgram in sm.dgrams:
                    cn_dgrams += 1
                    if cn_dgrams < from_pos: 
                        continue
                    yield dgram
    
    def values(self, events, env_variable):
        """ Returns values of the env_variable for the given events.

        First search for env file that has this variable (return algorithm e.g.
        fast/slow) then for that env file, locate position of env dgram that
        has ts_env <= ts_evt. If the dgram at found position has the algorithm
        then returns the value, otherwise keeps searching backward until 
        PS_N_env_SEARCH_STEPS is reached."""
        
        PS_N_STEP_SEARCH_STEPS = int(os.environ.get("PS_N_STEP_SEARCH_STEPS", "10"))
        env_values = []
        for i, env in enumerate(self.env_managers):
            alg = env.alg_from_variable(env_variable)
            if alg: 
                event_timestamps = np.asarray([evt.timestamp for evt in events], dtype=np.uint64)

                found_positions = np.searchsorted(env.timestamps, event_timestamps)
                found_positions -= 1 # return the env event before the found position.
                for pos in found_positions:
                    val = None
                    for p in range(pos, pos - PS_N_STEP_SEARCH_STEPS, -1):
                        if p < 0:
                            break
                        envs = getattr(env.dgrams[p], self.env_name)[0]
                        if hasattr(envs, alg):
                            val = getattr(getattr(envs, alg), env_variable)
                            break
                    env_values.append(val)

                break
        
        return env_values
    

