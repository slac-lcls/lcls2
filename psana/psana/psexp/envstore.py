from psana.psexp import PacketFooter
import numpy as np
from collections import defaultdict
import os
from psana.detector.detector_impl import DetectorImpl

class EnvManager(object):
    """ Store list of Env dgrams, timestamps, and variables 
    for a single smd file. EnvStore own the objects created
    from this class (e.g. if you have two smd files, there will
    be two EnvManagers stored in EnvStore.
    """
    
    def __init__(self, config, env_name):
        self.config = config
        self.env_name = env_name
        self.dgrams = []
        self.timestamps = []
        self.n_items = 0
        self._init_env_variables()

    def _init_env_variables(self):
        """ From the given config, build a list of variables from
        config.software.env_name.[alg].[] fields.
        
        env_variables = {alg: {segment_id: {var_name: var_type, }, }, }
        where Var contains name and type for the variables.
        """
        self.env_variables = {}
        if hasattr(self.config.software, self.env_name):
            envs = getattr(self.config.software, self.env_name)
            for segment_id, env in envs.items(): # check each segment 
                algs = list(vars(env).keys())
                algs.remove('dettype')
                algs.remove('detid')
                for alg in algs:
                    seg_alg = getattr(envs[segment_id], alg)
                    env_vars = {}
                    for var_name in vars(seg_alg):
                        if var_name in ('version', 'software') : continue
                        var_obj = getattr(seg_alg, var_name)
                        var_type = DetectorImpl._return_types(var_obj._type, var_obj._rank)
                        env_vars[var_name] = var_type
                        
                    self.env_variables[alg] = {segment_id: env_vars}

    def add(self, d):
        self.dgrams.append(d)
        self.timestamps.append(d.timestamp())
        self.n_items += 1
    
    def is_empty(self):
        return self.env_variables
    
    def locate_variable(self, var_name):
        """ Returns algorithm name and segment_id from the given env variable
        specifically for this config."""
        for alg, envs in self.env_variables.items():
            for segment_id, var_dict in envs.items():
                if var_name in var_dict:
                    return alg, segment_id
        return None

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

            # EnvStore has env_variables from all the env_managers
            for envm in self.env_managers:
                for alg, env_dict in envm.env_variables.items(): 
                    if alg not in self.env_variables:
                        self.env_variables[alg] = env_dict
                    else:
                        for segment_id, var_dict in env_dict.items():
                            if segment_id not in self.env_variables[alg][segment_id]:
                                self.env_variables[alg] = {segment_id: var_dict}
                            else:
                                self.env_variables[alg][segment_id].update(var_dict)

            self.env_info = []
            for alg, env_dict in self.env_variables.items():
                var_names = []
                for segment_id, var_dict in env_dict.items():
                    for var_name, _ in var_dict.items():
                        var_names.append(var_name)
                var_names.sort()
                for v in var_names:
                    self.env_info.append((v, alg))

    def locate_variable(self, var_name):
        """ Returns algorithm name and segment_id from the given env variable. """
        for alg, envs in self.env_variables.items():
            for segment_id, var_dict in envs.items():
                if var_name in var_dict:
                    return alg, segment_id
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
        
        for evt in events:
            event_timestamp = np.array([evt.timestamp], dtype=np.uint64)
            for i, env_man in enumerate(self.env_managers):
                val = None
                env_var_loc = env_man.locate_variable(env_variable) # check if this xtc has the variable
                if env_var_loc:
                    alg, segment_id = env_var_loc
                    found_pos = np.searchsorted(env_man.timestamps, event_timestamp)[0]
                    
                    if found_pos == env_man.n_items: # this event is the last step or the events after
                        found_pos -= 1
                    
                    for p in range(found_pos, found_pos - PS_N_STEP_SEARCH_STEPS, -1):
                        if p < 0:
                            break
                        envs = getattr(env_man.dgrams[p], self.env_name)[segment_id]
                        if hasattr(envs, alg):
                            val = getattr(getattr(envs, alg), env_variable)
                            break
                    
                    if val is not None: break # found the value from this env manager
            env_values.append(val)
        
        return env_values

    def get_info(self):
        info = {}
        for alg, segment_dict in self.env_variables.items():
            for segment_id, var_dict in segment_dict.items():
                for var_name, _ in var_dict.items():
                    info[(var_name, alg)] = alg
        return info

    def dtype(self, var_name):
        var_loc = self.locate_variable(var_name)
        if var_loc:
            alg, segment_id = var_loc
            var_dict = self.env_variables[alg][segment_id]
            if var_name in var_dict:
                return var_dict[var_name]
        return None

