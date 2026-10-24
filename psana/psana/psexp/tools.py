"""Parallel-mode settings and small helpers shared by `psana.psexp`.

`mode` is the environment variable PS_PARALLEL (default "mpi"). `MODE` is "SERIAL" when `mode`
is "mpi" and the MPI world has one rank, and "PARALLEL" in every other case.
"""
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
    """Give a run object a unique integer id and register it.

    The constructor sets `run.id` from the class-level counter `next_run_id`, increments the
    counter, and stores the run in the class-level `WeakValueDictionary` `run_by_id`, so
    `run_from_id` can find it while the run is alive.
    """
    next_run_id = 0
    run_by_id = weakref.WeakValueDictionary()

    def __init__(self, run):
        run.id = RunHelper.next_run_id
        RunHelper.next_run_id += 1
        RunHelper.run_by_id[run.id] = run


def run_from_id(run_id):
    """Return the run registered under `run_id` by `RunHelper`.

    Raises KeyError if no live run has that id.
    """
    return RunHelper.run_by_id[run_id]


class ConfigHelper(object):

    # Given a datasource, this class handles setting up configs
    # related information:
    # - Prune list of configs and files for selected detectors
    # - Setup det_class table
    # - Setup configinfo dict

    """Hold a data source `ds` and prune its file and config lists to the selected detectors.

    The only method, `_prune_to_sel_det`, keeps the entries of `ds.smd_files`, `ds.xtc_files` and
    `ds._configs` whose config has an attribute named in `ds.sel_det_names`; nothing changes if
    that list is empty.
    """
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
    """Return the list of excluded MPI rank numbers.

    The list is rank 0 (commented as SMD0), ranks 1 to PS_EB_NODES (default 1), and the last
    PS_SRV_NODES ranks (default 0). Returns an empty list if `mode` is not "mpi" or there is only
    one rank.
    """
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
