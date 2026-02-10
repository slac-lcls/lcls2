import os
import weakref

# mode can be 'mpi' or 'none' for non parallel
mode = os.environ.get("PS_PARALLEL", "mpi")
MODE = "PARALLEL"
if mode == "mpi":
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() == 1:
        MODE = "SERIAL"


def get_smd_n_events():
    """Return PS_SMD_N_EVENTS as int; default is 20000."""
    default_value = "20000"
    raw_value = os.environ.get("PS_SMD_N_EVENTS", default_value)
    try:
        return int(raw_value)
    except ValueError:
        return int(default_value)


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
            sel_indices = [
                i
                for i in range(len(self.ds.smd_files))
                if s1.intersection(set(self.ds._configs[i].__dict__.keys()))
            ]
            sel_smd_files = [self.ds.smd_files[i] for i in sel_indices]
            sel_xtc_files = [self.ds.xtc_files[i] for i in sel_indices]
            sel_configs = [self.ds._configs[i] for i in sel_indices]

            self.ds.smd_files = sel_smd_files
            self.ds.xtc_files = sel_xtc_files
            self.ds._configs = sel_configs


def get_excl_ranks():
    if mode != "mpi":
        return []

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    if size == 1:
        return []

    n_ebs = int(os.environ.get("PS_EB_NODES", "1"))
    n_srvs = int(os.environ.get("PS_SRV_NODES", "0"))
    excl_ranks = [0]  # SMD0
    excl_ranks += list(range(1, n_ebs + 1))
    excl_ranks += list(range(size - n_srvs, size))

    return excl_ranks
