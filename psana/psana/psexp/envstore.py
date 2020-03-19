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
        
        env_variables = {alg: {segment_id: ['var1','var2']}, }
        """
        self.env_variables = {}
        if hasattr(self.config.software, self.env_name):
            envs = getattr(self.config.software, self.env_name)
            for segment_id, env in envs.items(): # check each segment 
                algs = list(vars(env).keys())
                algs.remove('dettype')
                algs.remove('detid')
                for alg in algs:
                    env_vars = list(vars(getattr(envs[segment_id], alg)))
                    env_vars.remove('version')
                    env_vars.remove('software')
                    self.env_variables[alg] = {segment_id: env_vars}

    def add(self, d):
        self.dgrams.append(d)
        self.timestamps.append(d.timestamp())
        self.n_items += 1
    
    def is_empty(self):
        return self.env_variables
    
    def locate_variable(self, variable_name):
        """ Returns algorithm name and segment_id from the given env variable. """
        for alg, envs in self.env_variables.items():
            for segment_id, env_vars in envs.items():
                if variable_name in env_vars:
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
                        for segment_id, var_list in env_dict.items():
                            if segment_id not in self.env_variables[alg][segment_id]:
                                self.env_variables[alg] = {segment_id: var_list}
                            else:
                                self.env_variables[alg][segment_id] += var_list

            self.env_info = []
            for alg, env_dict in self.env_variables.items():
                all_var_list = []
                for segment_id, var_list in env_dict.items():
                    all_var_list.extend(var_list)
                all_var_list.sort()
                for v in all_var_list:
                    self.env_info.append((v, alg))

    def locate_variable(self, variable_name):
        """ Returns algorithm name and segment_id from the given env variable. """
        for alg, envs in self.env_variables.items():
            for segment_id, env_vars in envs.items():
                if variable_name in env_vars:
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
                    found_pos -= 1 # return the env event before the found position.
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
    

