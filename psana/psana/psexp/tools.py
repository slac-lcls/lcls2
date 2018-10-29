import pickle
import weakref
import os
mode = os.environ.get('PS_PARALLEL', 'mpi')

class RunHelper(object):

    # Every Run is assigned an ID. This permits Run to be
    # pickled and sent across the network, as long as every node has the same
    # Run under the same ID. (This should be true as long as the client
    # code initializes Runs in a deterministic order.)
    next_run_id = 0
    run_by_id = weakref.WeakValueDictionary()

    def __init__(self, run):
        run.id = self.next_run_id
        self.next_run_id += 1
        self.run_by_id[run.id] = run

def run_from_id(run_id):
    return RunHelper.run_by_id[run_id]

