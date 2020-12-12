import weakref
import os
import logging

# mode can be 'mpi' or 'legion' or 'none' for non parallel 
mode = os.environ.get('PS_PARALLEL', 'mpi')


class RunHelper(object):

    # Every Run is assigned an ID. This permits Run to be
    # pickled and sent across the network, as long as every node has the same
    # Run under the same ID. (This should be true as long as the client
    # code initializes Runs in a deterministic order.)
    next_run_id = 0
    run_by_id = weakref.WeakValueDictionary()

    def __init__(self, run):
        run.id = RunHelper.next_run_id
        RunHelper.next_run_id += 1
        RunHelper.run_by_id[run.id] = run

def run_from_id(run_id):
    return RunHelper.run_by_id[run_id]


class ConfigHelper(object):

    # Given a datasource, this class handles setting up configs
    # related information:
    # - Prune list of configs and files for selected detectors
    # - Setup det_class table
    # - Setup configinfo dict
    
    def __init__(self, ds):
        self.ds = ds

    def _prune_to_sel_det(self):
        if self.ds.sel_det_names:
            s1 = set(self.ds.sel_det_names)
            sel_indices = [i for i in range(len(self.ds.smd_files)) \
                        if s1.intersection(set(self.ds._configs[i].__dict__.keys()))]
            sel_smd_files = [self.ds.smd_files[i] for i in sel_indices]
            sel_xtc_files = [self.ds.xtc_files[i] for i in sel_indices]
            sel_configs   = [self.ds._configs[i] for i in sel_indices]

            self.ds.smd_files = sel_smd_files
            self.ds.xtc_files = sel_xtc_files
            self.ds._configs  = sel_configs

class Logging(object):

    @staticmethod
    def info(msg):
        logging.debug(msg)
