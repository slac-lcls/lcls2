"""
Smalldata (v2)
Parallel data analysis with MPI send/recv

Analysis consists of two different process types:

1. clients
    > these perform per-event analysis
    > are associted with one specific server
    > after processing `batch_size` events, send a
      dict of data over to their server

  2. servers (srv)
    > recv a batch of events from one of many clients
    > add these batches to a `cache`
    > when the cache is full, write to disk
    > each server produces its OWN hdf5 file

>> at the end of execution, rank 0 "joins" all the
   individual hdf5 files together using HDF virtual
   datasets -- this provides a "virtual", unified
   view of all processed data


             CLIENT                SRV
         [ --------- ]       | [ --------- ]
         [ -{event}- ]  send | [ --------- ]
  batch  [ --------- ]  ~~~> | [ --------- ]
         [ --------- ]       | [ --------- ]
         [ --------- ]       | [ --------- ]
                             |
                             | [ --------- ]
                             | [ --------- ]
                             | [ --------- ]
                             | [ --------- ]
                             | [ --------- ]
                             | -- cache


  (I apologize for indulging in some ASCII art)

Some Notes:
  * number of servers to use is set by PS_SRV_NODES
    environment variable
  * if running in psana parallel mode, clients ARE
    BD nodes (they are the same processes)
  * eventual time-stamp sorting would be doable with
    code conceptually similar to this (but would need
    to be optimized for performance):

    import numpy as np
    import h5py
    f = h5py.File('smalldata_test.h5')
    ts = f['timestamp'][:]
    tsneg = f['tsneg']
    for i in np.argsort(ts):
        print(tsneg[i])

"""

import glob
import os
import time
from collections.abc import MutableMapping

import h5py
import numpy as np

from psana.psexp.tools import mode
from psana.psexp.prometheus_manager import get_prom_manager

# -----------------------------------------------------------------------------


if mode == "mpi":
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
else:
    SIZE = 1

if SIZE > 1:
    MODE = "PARALLEL"
else:
    MODE = "SERIAL"

# -----------------------------------------------------------------------------
MISSING_INT = -99999
MISSING_FLOAT = np.nan

INT_TYPES = [
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    int,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uint,
]
FLOAT_TYPES = [float, np.float16, np.float32, np.float64, np.float128, float]

# This is not actually implemented, so let's comment it out for now.
# RAGGED_PREFIX   = 'ragged_'
VAR_PREFIX = "var_"
LEN_SUFFIX = "_len"
ALIGN_GROUP_KW = "align_group"
ALIGN_GROUP_DEFAULT = "default"

# It is assumed that "var_" can appear at any level in the key.
#
# The lengths of all the variables that branch from a given var
# will be the same and will share a "_len" array.

var_dict = {}  # var name to True/False if var
len_dict = {}  # var name to True/False if len
len_map = {}  # var name to len array name (i.e. x/var_mcb/y --> x/var_mcb_len)
len_evt = {}  # len name to {timestamp: length}


def set_keytypes(k):
    """Classify dataset key `k` and record the result in the module dicts.

    A key is variable-length (`var_dict[k] = True`) if any "/"-separated part starts with "var_". If that part is the last one and ends with "_len", the key is itself a length key; otherwise the name of its length key (the path up to and including that part, plus "_len") is stored in `len_map[k]`, marked as a length key in `len_dict` and given an empty entry in `len_evt`.

    Notes
    -----
    Keys with no "var_" part are marked as neither variable nor length keys.
    """
    parts = k.split("/")
    # New regime: any part can have "var_"!
    for i, p in enumerate(parts):
        if p.startswith(VAR_PREFIX):
            var_dict[k] = True
            if p.endswith(LEN_SUFFIX) and i == len(parts) - 1:
                len_dict[k] = True
            else:
                m = "/".join(parts[: i + 1]) + LEN_SUFFIX
                len_map[k] = m
                len_dict[k] = False
                len_dict[m] = True
                if m not in len_evt.keys():
                    len_evt[m] = {}
            return
    var_dict[k] = False
    len_dict[k] = False


def get_len_map(k):
    """Return the name of the length key for variable key `k`.

    Looks `k` up in `len_map`, calling `set_keytypes(k)` first if it is not there yet.

    Raises
    ------
    KeyError
        If `k` has no length key (it has no "var_" part, or it is itself a length key).
    """
    try:
        return len_map[k]
    except Exception:
        set_keytypes(k)
        return len_map[k]


def is_len_key(k):
    """Return True if `k` is a length key, as classified by `set_keytypes`.

    The classification is computed on first lookup and cached in `len_dict`.
    """
    try:
        return len_dict[k]
    except Exception:
        set_keytypes(k)
        return len_dict[k]


def is_var_key(k):
    """Return True if any "/"-separated part of `k` starts with "var_".

    The classification is computed by `set_keytypes` on first lookup and cached in `var_dict`.
    """
    try:
        return var_dict[k]
    except Exception:
        set_keytypes(k)
        return var_dict[k]


def is_group(event_data_dict):
    """Return True if `event_data_dict` contains the reserved key "align_group"."""
    return ALIGN_GROUP_KW in event_data_dict


def get_group_name(event_data_dict):
    """Return the value of the "align_group" key of `event_data_dict`, or "default" if it is absent."""
    datagroup = ALIGN_GROUP_DEFAULT
    if is_group(event_data_dict):
        datagroup = event_data_dict[ALIGN_GROUP_KW]
    return datagroup
# -----------------------------------------------------------------------------


def _flatten_dictionary(d, parent_key="", sep="/"):
    """
    http://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, MutableMapping):
            items.extend(_flatten_dictionary(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _get_missing_value(dtype):
    if type(dtype) is not np.dtype:
        dtype = np.dtype(dtype)

    if dtype in INT_TYPES:
        missing_value = MISSING_INT
    elif dtype in FLOAT_TYPES:
        missing_value = MISSING_FLOAT
    else:
        raise ValueError("%s :: Invalid num type for missing data" % str(dtype))

    return missing_value


def _format_srv_filename(dirname, basename, rank):
    srv_basename = "%s_part%d.h5" % (basename.replace(".h5", ""), rank)
    srv_fn = os.path.join(dirname, srv_basename)
    return srv_fn


# FOR NEXT TIME
# CONSIDER MAKING A FileServer CLASS
# CLASS BASECLASS METHOD THEN HANDLES HDF5


class CacheArray:
    """
    The CacheArray class provides for a *data cache* in
    the server's memory.
    """

    def __init__(self, singleton_shape, dtype, cache_size, is_var):
        self.singleton_shape = singleton_shape
        self.dtype = dtype
        self.cache_size = cache_size
        self.is_var = is_var

        # initialize
        if is_var:
            # We really *can't* cache variable arrays and have to just
            # keep appending to them.
            self.data = None
        else:
            self.data = np.empty(
                (self.cache_size,) + self.singleton_shape, dtype=self.dtype
            )
        self.reset()

        return

    def append(self, data):
        """Add one event's `data` to the cache.

        For a fixed-shape cache, `data` is written into the next row of the preallocated array and `size` grows by 1. For a variable-length cache, the entries of `data` are appended along axis 0 and `size` grows by `len(data)`; None or empty `data` adds nothing.

        Notes
        -----
        `n_events` is incremented in every case.
        """
        if self.is_var:
            if data is None or len(data) == 0:
                self.n_events += 1
                return
            if self.data is None:
                self.data = np.reshape(
                    np.array(data, dtype=self.dtype),
                    (len(data),) + self.singleton_shape,
                )
            else:
                self.data = np.append(self.data, data, axis=0)
            self.size += len(data)
        else:
            self.data[self.n_events, ...] = data
            self.size += 1
        self.n_events += 1
        return

    def reset(self):
        """Mark the cache as empty.

        Sets `size` and `n_events` to 0 and, for a variable-length cache, sets `data` to None. A
        fixed-shape cache keeps its buffer; old rows are overwritten later.
        """
        if self.is_var:
            self.data = None
        self.size = 0
        self.n_events = 0
        return


class Server:  # (hdf5 handling)
    """Receive per-event data from smalldata clients and write it to one HDF5 file.

    For each alignment group it records the dtype and shape of every dataset and keeps a
    `CacheArray` per dataset, which is flushed to the file when it holds `cache_size` entries;
    datasets missing from an event are backfilled. If `filename` is None nothing is written, but
    the callbacks are still called for every event.

    Parameters
    ----------
    filename : str, optional
        HDF5 file to create (opened with mode "w" and `libver="latest"` in the constructor).
    smdcomm : MPI communicator, optional
        Communicator used by `recv_loop`; every rank in it except one is counted as a client.
    cache_size : int
        Number of entries cached per dataset before writing.
    callbacks : list of callable
        Each is called with one flattened dict per event in `handle`.
    swmr_mode : bool
        If True, turn on HDF5 SWMR mode after the first dataset is created and flush the file after
        every batch.
    """
    def __init__(self, filename=None, smdcomm=None, cache_size=10000, callbacks=[], swmr_mode=False):
        self.filename = filename
        self.smdcomm = smdcomm
        self.cache_size = cache_size
        self.callbacks = callbacks
        self.swmr_mode = swmr_mode

        # maps datagroup --> {dataset_name --> (dtype, shape), }
        self._dsets = {}

        # maps datagroup --> {dataset_name --> CacheArray(), }
        self._cache = {}

        # maps datagroup --> #evt seen
        self.num_events_seen = {}

        if self.filename is not None:
            self.file_handle = h5py.File(self.filename, "w", libver="latest")
            # SWMR mode will be enabled after first dataset creation
            self._swmr_enabled = False

        pm = get_prom_manager()
        self.wait_gauge = pm.get_metric("psana_srv_wait")
        self.rate_gauge = pm.get_metric("psana_srv_rate")

        return

    def recv_loop(self):
        """Receive batches from clients on `smdcomm` until every client has sent "done".

        The number of clients is `smdcomm.Get_size() - 1`. A received list is passed to `handle`; the string "done" counts one finished client.

        Notes
        -----
        The receive wait time and `number of events / processing seconds * 1e-3` are set on the "psana_srv_wait" and "psana_srv_rate" Prometheus gauges.
        """
        num_clients_done = 0
        num_clients = self.smdcomm.Get_size() - 1
        while num_clients_done < num_clients:
            wait_start = time.time()
            msg = self.smdcomm.recv(source=MPI.ANY_SOURCE)
            wait_end = time.time()
            self.wait_gauge.set(wait_end - wait_start)
            if type(msg) is list:
                proc_start = time.time()
                self.handle(msg)
                proc_end = time.time()

                num_evts = len(msg)  # <-- count number of events per batch
                self.rate_gauge.set((num_evts / (proc_end - proc_start)) * 1e-3)
            elif msg == "done":
                num_clients_done += 1

        return

    def handle(self, batch):
        """Process a batch of events: store their datasets and call the callbacks.

        When a file is open, new datasets are created with `new_dset`, values go through
        `append_to_cache`, datasets seen before but absent from this event are backfilled, and the length of "var_"
        data is appended to the shared "_len" dataset (once per event when the keys sharing it have
        equal lengths). Each callback gets one flat dict
        per event whose keys are "<group>/<name>" (just "<name>" for the default group).

        Parameters
        ----------
        batch : list of dict
            One dict per event, mapping alignment group name to that event's
            {dataset name: value} dict, which must contain "timestamp" when a file is written.

        Raises
        ------
        KeyError
            If an event key is itself a "var_..._len" length key.
        """
        for grp_event_data in batch:
            # flatten datagroup to align_group/dataset_name: val for the callbacks
            flatten_event_data_dict = {}
            for datagroup, event_data_dict in grp_event_data.items():
                if datagroup not in self._dsets:
                    self._dsets[datagroup] = {}
                    self._cache[datagroup] = {}
                    self.num_events_seen[datagroup] = 0
                dsets = self._dsets[datagroup]

                to_backfill = []
                if self.filename is not None:
                    # to_backfill: list of keys we have seen previously
                    #              we want to be sure to backfill if we
                    #              dont see them
                    to_backfill = list(dsets.keys())

                for dataset_name, data in event_data_dict.items():
                    # save with proper flatten key for the callbacks
                    if datagroup != ALIGN_GROUP_DEFAULT:
                        flatten_key = datagroup + "/" + dataset_name
                    else:
                        flatten_key = dataset_name
                    flatten_event_data_dict[flatten_key] = data

                    if self.filename is not None:
                        is_var = is_var_key(dataset_name)
                        is_len = is_len_key(dataset_name)
                        if is_var:
                            if is_len:
                                raise KeyError(
                                    'Key: event keys cannot have the form "var_*_len! (%s)'
                                    % (dataset_name)
                                )
                            else:
                                len_name = len_map[dataset_name]
                        if dataset_name not in dsets.keys():
                            if is_var:
                                # A new var array.
                                if len(data) > 0:
                                    self.new_dset(dataset_name, data, datagroup)
                                    # Several dataset_names can share a length dataset!
                                    if len_name not in dsets.keys():
                                        self.new_dset(len_name, len(data), datagroup)
                                else:
                                    # If we don't have any actual data, we can't even
                                    # figure out types, so we'll just have to backfill
                                    # later!
                                    continue
                            else:
                                self.new_dset(dataset_name, data, datagroup)
                        else:
                            to_backfill.remove(dataset_name)
                        self.append_to_cache(dataset_name, data, datagroup)
                        ts = event_data_dict["timestamp"]
                        if is_var:
                            try:
                                # All of the datasets that share a length dataset should
                                # be the same size.  Otherwise, flag an error.
                                exp_len = len_evt[len_name][ts]
                                if len(data) != exp_len:
                                    raise TypeError(
                                        "Data for %s is length %d, not %d!"
                                        % (dataset_name, len(data), exp_len)
                                    )
                            except Exception:
                                # This is the first dataset for this length dataset,
                                # so remember the length and forget all of the older ones!
                                len_evt[len_name] = {ts: len(data)}
                                self.append_to_cache(len_name, len(data), datagroup)

                # end for dataset_name
                for dataset_name in to_backfill:
                    if is_var_key(dataset_name):
                        # So, we never have to backfill a variable key.  Or for
                        # that matter, the length key should never be in the event,
                        # so we can ignore it in the to_backfill list.
                        #
                        # However, we might have to fill its length *if* the key
                        # itself is in the list!
                        if is_len_key(dataset_name):
                            continue
                        len_name = len_map[dataset_name]
                        try:
                            exp_len = len_evt[len_name][event_data_dict["timestamp"]]
                        except Exception:
                            # Only backfill the first time we see this timestamp.
                            self.backfill(len_name, 1, datagroup, missing_value=0)
                            len_evt[len_name][event_data_dict["timestamp"]] = 0
                    else:
                        self.backfill(dataset_name, 1, datagroup)

                # end if self.filename...
                self.num_events_seen[datagroup] += 1

            # end for datagroup...
            for cb in self.callbacks:
                cb(flatten_event_data_dict)

        # end for grp_event_data...
        if self.swmr_mode: self.file_handle.flush()
        return

    def _get_data_info(self, data, dataset_name):
        if type(data) is int:
            shape = ()
            maxshape = (None,)
            dtype = "i8"
        elif type(data) is float:
            shape = ()
            maxshape = (None,)
            dtype = "f8"
        elif hasattr(data, "dtype"):
            shape = data.shape
            maxshape = (None,) + data.shape
            dtype = data.dtype
        else:
            raise TypeError(
                "Type: Dataset %s type %s not compatible" % (dataset_name, type(data))
            )
        return (shape, maxshape, dtype)

    def new_dset(self, dataset_name, data, datagroup):
        """Create an empty, resizable HDF5 dataset for `dataset_name` in `datagroup`.

        The per-event shape and dtype come from `data` (from its first element for a "var_" key) and
        are recorded in `_dsets`; the dataset is created at the file root for the default group and in
        the group `datagroup` otherwise, chunked by `cache_size`. Then, except for non-length "var_"
        keys, `num_events_seen[datagroup]` zero entries are backfilled.

        Raises
        ------
        TypeError
            If `data` is not an int, float or object with a `dtype`, or a "var_" key's data is not a
            list or array.
        ValueError
            If the per-event shape is (0,).
        """
        is_var = is_var_key(dataset_name)
        is_len = is_len_key(dataset_name)
        if is_var and not is_len:
            if type(data) is list:
                data = np.array(data)
            if hasattr(data, "dtype"):
                (shape, maxshape, dtype) = self._get_data_info(data[0], dataset_name)
            else:
                raise TypeError(
                    "Type: Dataset %s is variable and should be a list!" % dataset_name
                )
        else:
            (shape, maxshape, dtype) = self._get_data_info(data, dataset_name)
        if shape == (0,):
            raise ValueError("Dataset %s has illegal shape (0,)" % dataset_name)

        self._dsets[datagroup].update({dataset_name: (dtype, shape)})

        # determine handle is root or user-specified group
        if datagroup == ALIGN_GROUP_DEFAULT:
            safe_handle = self.file_handle
        else:
            if datagroup not in self.file_handle:
                self.file_handle.create_group(datagroup)
            safe_handle = self.file_handle[datagroup]

        safe_handle.create_dataset(
            dataset_name,
            (0,) + shape,  # (0,) -> expand dim
            maxshape=maxshape,
            dtype=dtype,
            chunks=(self.cache_size,) + shape,
        )

        # Enable SWMR mode after first dataset is created
        if self.swmr_mode and not self._swmr_enabled:
            self.file_handle.swmr_mode = True
            self._swmr_enabled = True

        if is_var:
            if is_len:
                self.backfill(
                    dataset_name,
                    self.num_events_seen[datagroup],
                    datagroup,
                    missing_value=0,
                )
        else:
            self.backfill(
                dataset_name,
                self.num_events_seen[datagroup],
                datagroup,
                missing_value=0,
            )
        return

    def append_to_cache(self, dataset_name, data, datagroup):
        """Append `data` to the cache of `dataset_name` in `datagroup`.

        Creates a `CacheArray` from the recorded dtype and shape on first use. When the cache holds
        `cache_size` or more entries, it is written out with `write_to_file`.
        """
        if dataset_name not in self._cache[datagroup].keys():
            dtype, shape = self._dsets[datagroup][dataset_name]
            cache = CacheArray(
                shape,
                dtype,
                self.cache_size,
                is_var_key(dataset_name) and not is_len_key(dataset_name),
            )
            self._cache[datagroup][dataset_name] = cache
        else:
            cache = self._cache[datagroup][dataset_name]

        cache.append(data)

        if cache.size >= self.cache_size:
            self.write_to_file(dataset_name, datagroup, cache)

        return

    def write_to_file(self, dataset_name, datagroup, cache):
        # --- HOT PATCH (2025-06-22): skip writing cache with no data ---
        # Avoids TypeError when cache.data is None (e.g. detector never produced data).
        # This happens for variable-length datasets that were registered but never filled.
        # See traceback from tmo-preproc/preproc_tab_v2.py using SmallData done().
        """Write the entries held in `cache` to the end of the HDF5 dataset and reset the cache.

        The dataset (`dataset_name` at the root for the default group, else
        `datagroup/dataset_name`) is resized by `cache.size` and the first `cache.size` cached rows are
        written into the new space. If the cache has no data, a warning is printed, the cache is reset
        and nothing is written.
        """
        if cache.data is None or cache.size == 0:
            print(f"[WARN] Skipping flush for {datagroup}/{dataset_name}: no data to write ({cache.data=}, {cache.size=})")
            cache.reset()
            return
        # --- END HOT PATCH ---
        if datagroup == ALIGN_GROUP_DEFAULT:
            dset = self.file_handle.get(dataset_name)
        else:
            dset = self.file_handle.get(datagroup + "/" + dataset_name)
        new_size = (dset.shape[0] + cache.size,) + dset.shape[1:]
        dset.resize(new_size)
        # remember: data beyond size in the cache may be OLD
        dset[-cache.size :, ...] = cache.data[: cache.size, ...]
        cache.reset()
        return

    def backfill(self, dataset_name, num_to_backfill, datagroup, missing_value=None):
        """Append `num_to_backfill` fill entries to the cache of `dataset_name`.

        The fill entry has the dataset's recorded shape and dtype and is filled with `missing_value`,
        or, when that is None, with -99999 for integer dtypes and NaN for float dtypes.

        Raises
        ------
        ValueError
            If `missing_value` is None and the dtype is neither integer nor float.
        """
        dtype, shape = self._dsets[datagroup][dataset_name]

        if missing_value is None:
            missing_value = _get_missing_value(dtype)
        fill_data = np.empty(shape, dtype=dtype)
        fill_data.fill(missing_value)

        for i in range(num_to_backfill):
            self.append_to_cache(dataset_name, fill_data, datagroup)

        return

    def done(self):
        """Flush all non-empty caches, close the HDF5 file and fsync it.

        Does nothing if `filename` is None. Errors from the final HDF5 flush and from the fsync are
        ignored.
        """
        if self.filename is not None:
            # flush the data caches (in case did not hit cache_size yet)
            for dgroup, dset_cache in self._cache.items():
                for dset, cache in dset_cache.items():
                    if cache.n_events > 0:
                        self.write_to_file(dset, dgroup, cache)

            # flush h5 cache
            try:
                self.file_handle.flush()
            except Exception:
                pass

            self.file_handle.close()

            # flush node cache
            try:
                with open(self.filename, "r") as fo:
                    os.fsync(fo.fileno())
            except Exception:
                pass
        return


class SmallData:  # (client)
    """Client-side object for saving per-event and summary data to HDF5 with smalldata.

    In parallel mode (more than one MPI rank) the constructor builds communicators from
    `server_group` and `client_group` and gives this rank the role "server", "client" or "other";
    in serial mode the constructor does nothing. Clients collect events in batches and send them to
    their server, servers write part files, and `done` joins the part files into one file with HDF5
    virtual datasets (on client rank 0).

    Raises
    ------
    Exception
        In parallel mode, if `server_group` is empty.
    """
    def __init__(self, server_group=None, client_group=None):
        """
        Parameters
        ----------
        server_group : MPI.Group
            The MPI group to allocate to server processes

        client_group : MPI.Group
            The MPI group to allocate to client processes
        """
        if MODE == "PARALLEL":
            self._server_group = server_group
            self._client_group = client_group

            self._comm_partition()

    def setup_parms(
        self, filename=None, batch_size=1000, cache_size=None, callbacks=[], swmr_mode=False
    ):
        """
        Parameters
        ----------
        filename : str
            The file path of the (new) HDF5 file to write data to,
            will be overwritten if it exits -- if "None", data
            will not be written to disk.

        batch_size : int
            Number of events before send/recv

        cache_size : int
            Number of events before write

        callbacks : list of functions
            Functions that get called on each server's data before
            being written to disk. The functions should take as
            arguments a dictionary, where the keys are the data field
            names and the values are the data themselves. Each event
            processed will have it's own dictionary of this form
            containing the data saved for that event.

        swmr_mode : bool
            Enable SWMR (Single Writer Multiple Reader) mode for HDF5 files.
            Allows concurrent read access while writing.
        """

        self.batch_size = batch_size
        self._batch = []
        self._previous_timestamp = -1
        self.swmr_mode = swmr_mode

        if cache_size is None:
            cache_size = batch_size
        if cache_size < batch_size:
            print("Warning: `cache_size` smaller than `batch_size`")
            print("setting cache_size -->", batch_size)
            cache_size = batch_size

        if filename is not None:
            self._basename = os.path.basename(filename)
            self._dirname = os.path.dirname(filename)
            self._full_filename = str(filename)
        else:
            # this can happen for shmem analysis where there is no file
            self._full_filename = filename

        self._first_open = True  # filename has not been opened yet

        if MODE == "PARALLEL":
            # hide intermediate files -- join later via VDS
            if filename is not None:
                # clean up previous part files associated with the filename.
                # This needs to be done on a single rank to avoid trying to delete i
                # the same file multiple times.
                if self._type == "client" and self._full_filename is not None:
                    if self._client_comm.Get_rank() == 0:
                        for f in glob.glob(
                            self._full_filename.replace(".h5", "_part*.h5")
                        ):
                            os.remove(f)
                # Need to make sure all smalldata ranks wait for the clean up to be done
                # before they go about creating the new files.
                if self._type != "other":  # other = not smalldata (Mona)
                    self._smalldata_comm.barrier()

                # Now make file
                self._srv_filename = _format_srv_filename(
                    self._dirname, self._basename, self._server_group.Get_rank()
                )
            else:
                self._srv_filename = None

            if self._type == "server":
                self._server = Server(
                    filename=self._srv_filename,
                    smdcomm=self._srvcomm,
                    cache_size=cache_size,
                    callbacks=callbacks,
                    swmr_mode=self.swmr_mode,
                )
                self._server.recv_loop()

        elif MODE == "SERIAL":
            if filename is not None:
                # clean up previous part files associated with the filename.
                for f in glob.glob(self._full_filename.replace(".h5", "_part*.h5")):
                    os.remove(f)
            self._srv_filename = self._full_filename  # dont hide file
            self._type = "serial"
            self._server = Server(
                filename=self._srv_filename, cache_size=cache_size, callbacks=callbacks,
                swmr_mode=self.swmr_mode
            )

        return

    def get_rank(self):
        """Return this rank's number in the smalldata communicator (servers plus clients).

        The communicator is created by the constructor only in parallel mode; in serial mode the
        attribute does not exist and AttributeError is raised.
        """
        return self._smalldata_comm.Get_rank()

    def get_world_rank(self):
        """Return the rank in MPI.COMM_WORLD, or 0 in serial mode."""
        if MODE == "SERIAL":
            return 0
        return RANK

    def _comm_partition(self):
        self._smalldata_group = MPI.Group.Union(self._server_group, self._client_group)
        self._smalldata_comm = COMM.Create(self._smalldata_group)
        self._client_comm = COMM.Create(self._client_group)

        # partition into comms
        n_srv = self._server_group.size
        if n_srv < 1:
            raise Exception(
                "Attempting to run smalldata with no servers"
                " set env var PS_SRV_NODES to be 1 or more"
            )

        if self._server_group.rank != MPI.UNDEFINED:  # if in server group
            self._type = "server"
            self._srv_color = self._server_group.rank
            self._srvcomm = self._smalldata_comm.Split(self._srv_color, 0)  # rank=0
            if self._srvcomm.Get_size() == 1:
                print("WARNING: server has no associated clients!")
                print("This core is therefore idle... set PS_SRV_NODES")
                print("to be smaller, or increase the number of mpi cores")
        elif self._client_group.rank != MPI.UNDEFINED:  # if in client group
            self._type = "client"
            self._srv_color = self._client_group.rank % n_srv
            self._srvcomm = self._smalldata_comm.Split(
                self._srv_color, RANK + 1
            )  # keep rank order
        else:
            # we are some other node type
            self._type = "other"

        return

    def _get_full_file_handle(self):
        """
        makes sure we overwrite on first open, but not after that
        """

        if MODE == "PARALLEL":
            if self._first_open and self._full_filename is not None:
                fh = h5py.File(self._full_filename, "w", libver="latest")
                if self.swmr_mode:
                    fh.swmr_mode = True
                self._first_open = False
            else:
                fh = h5py.File(self._full_filename, "r+", libver="latest")
                if self.swmr_mode:
                    fh.swmr_mode = True

        elif MODE == "SERIAL":
            fh = self._server.file_handle

        return fh

    def event(self, event, *args, **kwargs):
        """
        event: int, psana.event.Event
        """
        if type(event) is int:
            timestamp = event
        elif hasattr(event, "timestamp"):
            timestamp = int(event.timestamp)
        else:
            raise ValueError("`event` must have a timestamp attribute")

        # collect all new data to add
        event_data_dict = {}
        event_data_dict.update(kwargs)
        for d in args:
            event_data_dict.update(_flatten_dictionary(d))

        # group data to ALIGN_GROUP_DEFAULT and user-specifed groups
        # kwarg: align_group is reserved!!! and if found, all args and kwargs
        # are grouped datasets.
        datagroup = get_group_name(event_data_dict)
        if is_group(event_data_dict):
            event_data_dict.pop(ALIGN_GROUP_KW)

        # check to see if the timestamp indicates a new event...

        #   >> multiple calls to self.event(...), same event as before
        if timestamp == self._previous_timestamp:
            grp_event_data = self._batch[-1]
            if datagroup not in grp_event_data:
                grp_event_data[datagroup] = {"timestamp": timestamp}
            grp_event_data[datagroup].update(event_data_dict)

        #   >> we have a new event
        elif timestamp > self._previous_timestamp:
            # if we have a "batch_size", ship events
            # (this avoids splitting events if we have multiple
            #  calls to self.event)
            if len(self._batch) >= self.batch_size:
                if MODE == "SERIAL":
                    self._server.handle(self._batch)
                elif MODE == "PARALLEL":
                    self._srvcomm.send(self._batch, dest=0)
                self._batch = []

            event_data_dict["timestamp"] = timestamp
            self._previous_timestamp = timestamp
            self._batch.append({datagroup: event_data_dict})

        else:
            # FIXME: cpo
            print(
                'event data is "old", event timestamps'
                " must increase monotonically"
                " previous timestamp: %d, current: %d"
                "" % (self._previous_timestamp, timestamp)
            )
            """
            raise IndexError('event data is "old", event timestamps'
                             ' must increase monotonically'
                             ' previous timestamp: %d, current: %d'
                             '' % (self._previous_timestamp, timestamp))
            """

        return

    @property
    def summary(self):
        """
        This "flag" is required when you save summary data OR
        do a "reduction" operation (e.g. sum) across MPI procs

        >> if SmallData.summary:
        >>     whole = SmallData.sum(part)
        >>     SmallData.save_summary(mysum=whole)
        """
        r = False
        if MODE == "PARALLEL":
            if self._type == "client":
                r = True
        elif MODE == "SERIAL":
            r = True
        else:
            raise RuntimeError()
        return r

    def sum(self, value, inplace=False):
        """Sum `value` element-wise across client ranks (MPI.SUM) onto client rank 0.

        In serial mode `value` is returned unchanged. In parallel mode ints and floats become
        one-element arrays, client ranks with no array contribute zeros, and the result is returned on
        client rank 0; other client ranks get None (or their own `value` if `inplace`), and non-client
        ranks get None.

        Parameters
        ----------
        value : int, float or numpy.ndarray
            Local value.
        inplace : bool
            If True, reduce into `value` itself instead of a new array.

        Raises
        ------
        Exception
            If no client rank has an array, or the arrays differ in shape or dtype.
        """
        result = self._reduction(value, MPI.SUM, inplace)
        return result

    def max(self, value, inplace=False):
        """Take the element-wise maximum of `value` across client ranks (MPI.MAX) onto client rank 0.

        In serial mode `value` is returned unchanged. In parallel mode ints and floats become
        one-element arrays, client ranks with no array contribute an array filled with the global
        minimum, and the result is returned on client rank 0; other client ranks get None (or their
        own `value` if `inplace`), and non-client ranks get None.
        """
        return self._reduction(value, MPI.MAX, inplace)

    def min(self, value, inplace=False):
        """Take the element-wise minimum of `value` across client ranks (MPI.MIN) onto client rank 0.

        In serial mode `value` is returned unchanged. In parallel mode ints and floats become
        one-element arrays, client ranks with no array contribute an array filled with the global
        maximum, and the result is returned on client rank 0; other client ranks get None (or their
        own `value` if `inplace`), and non-client ranks get None.
        """
        return self._reduction(value, MPI.MIN, inplace)

    def prod(self, value, inplace=False):
        """Multiply `value` element-wise across client ranks (MPI.PROD) onto client rank 0.

        In serial mode `value` is returned unchanged. In parallel mode the result is returned on client rank 0, other client ranks get None (or their own `value` if `inplace`), and non-client ranks get None.

        Notes
        -----
        No stand-in array is made for a client rank without data, so every client rank must pass an array or number.
        """
        return self._reduction(value, MPI.PROD, inplace)

    def land(self, value, inplace=False):
        """Combine `value` across client ranks with logical AND (MPI.LAND) onto client rank 0.

        In serial mode `value` is returned unchanged. In parallel mode the result is returned on client rank 0, other client ranks get None (or their own `value` if `inplace`), and non-client ranks get None.

        Notes
        -----
        No stand-in array is made for a client rank without data.
        """
        return self._reduction(value, MPI.LAND, inplace)

    def band(self, value, inplace=False):
        """Combine `value` across client ranks with bitwise AND (MPI.BAND) onto client rank 0.

        In serial mode `value` is returned unchanged. In parallel mode the result is returned on client rank 0, other client ranks get None (or their own `value` if `inplace`), and non-client ranks get None.

        Notes
        -----
        No stand-in array is made for a client rank without data.
        """
        return self._reduction(value, MPI.BAND, inplace)

    def lor(self, value, inplace=False):
        """Combine `value` across client ranks with logical OR (MPI.LOR) onto client rank 0.

        In serial mode `value` is returned unchanged. In parallel mode the result is returned on client rank 0, other client ranks get None (or their own `value` if `inplace`), and non-client ranks get None.

        Notes
        -----
        No stand-in array is made for a client rank without data.
        """
        return self._reduction(value, MPI.LOR, inplace)

    def bor(self, value, inplace=False):
        """Combine `value` across client ranks with bitwise OR (MPI.BOR) onto client rank 0.

        In serial mode `value` is returned unchanged. In parallel mode the result is returned on client rank 0, other client ranks get None (or their own `value` if `inplace`), and non-client ranks get None.

        Notes
        -----
        No stand-in array is made for a client rank without data.
        """
        return self._reduction(value, MPI.BOR, inplace)

    def lxor(self, value, inplace=False):
        """Combine `value` across client ranks with logical XOR (MPI.LXOR) onto client rank 0.

        In serial mode `value` is returned unchanged. In parallel mode the result is returned on client rank 0, other client ranks get None (or their own `value` if `inplace`), and non-client ranks get None.

        Notes
        -----
        No stand-in array is made for a client rank without data.
        """
        return self._reduction(value, MPI.LXOR, inplace)

    def bxor(self, value, inplace=False):
        """Combine `value` across client ranks with bitwise XOR (MPI.BXOR) onto client rank 0.

        In serial mode `value` is returned unchanged. In parallel mode the result is returned on client rank 0, other client ranks get None (or their own `value` if `inplace`), and non-client ranks get None.

        Notes
        -----
        No stand-in array is made for a client rank without data.
        """
        return self._reduction(value, MPI.BXOR, inplace)

    def _safe_reduction(self, value, op, inplace):
        """
        method that is robust to ranks that have received no data
        """

        # convert float/int to array
        if isinstance(value, int) or isinstance(value, float):
            value = np.array([value])

        comm = self._client_comm
        if not isinstance(value, np.ndarray):
            arrInfo = None
        else:
            # in principle could avoid amin/amax calls here since only
            # really needed for min/max reductions
            arrInfo = [value.shape, value.dtype, np.amin(value), np.amax(value)]
        arrInfoAll = comm.allgather(arrInfo)
        summaryArrInfo = None
        for arrInfo in arrInfoAll:
            if arrInfo is None:
                continue
            if summaryArrInfo is None:
                summaryArrInfo = arrInfo
            if arrInfo[:2] != summaryArrInfo[:2]:  # check shape/dtype compatible
                raise Exception(
                    "Unable to reduce incompatible array shapes/types:",
                    arrInfo,
                    summaryArrInfo,
                )
            # find global min/max (only necessary for min/max reduction)
            if arrInfo[2] < summaryArrInfo[2]:
                summaryArrInfo[2] = arrInfo[2]
            if arrInfo[3] > summaryArrInfo[3]:
                summaryArrInfo[3] = arrInfo[3]
        if summaryArrInfo is None:
            raise Exception("No arrays found for MPI reduce")
        (shape, dtype, amin, amax) = summaryArrInfo
        # if our rank has no data manufacture some that won't affect the result
        if not isinstance(value, np.ndarray):
            if op is MPI.SUM:
                value = np.zeros(shape, dtype=dtype)
            elif op is MPI.MIN:
                value = np.empty(shape, dtype=dtype)
                value.fill(amax)  # use global max so we don't affect min calculation
            elif op is MPI.MAX:
                value = np.empty(shape, dtype=dtype)
                value.fill(amin)  # use global min so we don't affect max calculation
        if inplace:
            if comm.Get_rank() == 0:
                comm.Reduce(MPI.IN_PLACE, value, op=op)
            else:
                comm.Reduce(value, value, op=op)
            result = value
        else:
            if comm.Get_rank() == 0:
                result = np.empty(shape, dtype=dtype)
            else:
                result = None
            comm.Reduce(value, result, op=op)

        return result

    def _reduction(self, value, op, inplace):
        """
        perform a reduction across the worker MPI procs
        """

        # because only client nodes may have certain summary
        # variables, we collect the summary data on client
        # rank 0 -- later, we need to remember this client
        # is the one who needs to WRITE the summary data to disk!

        if MODE == "PARALLEL":
            red_val = None

            if self._type == "client":
                red_val = self._safe_reduction(value, op, inplace)

        elif MODE == "SERIAL":
            red_val = value  # just pass it through...

        return red_val

    def save_summary(self, *args, **kwargs):
        """
        Save 'summary data', ie any data that is not per-event (typically
        slower, e.g. at the end of the job).

        Interface is identical to SmallData.event()

        Note: this function should be called in a SmallData.summary: block
        """
        if self._full_filename is None:
            print(
                "Warning: smalldata not saving summary since no h5 filename specified"
            )
            return

        # in parallel mode, only client rank 0 writes to file
        if MODE == "PARALLEL":
            if self._client_comm.Get_rank() != 0:
                return

        # >> collect summary data
        data_dict = {}
        data_dict.update(kwargs)
        for d in args:
            data_dict.update(_flatten_dictionary(d))

        # >> write to file
        fh = self._get_full_file_handle()
        for dataset_name, data in data_dict.items():
            if data is None:
                print(
                    'Warning: dataset "%s" was passed value: None'
                    "... ignoring that dataset" % dataset_name
                )
            else:
                fh[dataset_name] = data

        # we don't want to close the file in serial mode
        # this file is the server's main (only) file
        if MODE == "PARALLEL":
            fh.close()

        return

    def done(self):
        """
        Finish any final communication and join partial files
        (in parallel mode).
        """
        # >> finish communication
        if self._type == "client":
            # we want to send the finish signal to the server
            if len(self._batch) > 0:
                self._srvcomm.send(self._batch, dest=0)
            self._srvcomm.send("done", dest=0)

        elif self._type == "server":
            self._server.done()

        elif self._type == "serial":
            if len(self._batch) > 0:
                self._server.handle(self._batch)
            self._server.done()

        # stuff only one process should do in parallel mode
        if MODE == "PARALLEL":
            if self._type != "other":  # other = not smalldata (Mona)
                self._smalldata_comm.barrier()

                # CLIENT rank 0 does all final file writing
                # this is because this client may write to the file
                # during "save_summary(...)" calls, and we want
                # ONE file owner
                if self._type == "client" and self._full_filename is not None:
                    if self._client_comm.Get_rank() == 0:
                        self.join_files()

        return

    def join_files(self):
        """Join the per-server part files into the main HDF5 file with virtual datasets.

        Looks for `<basename>_part<i>.h5` for every server rank i, printing a warning for each one that is missing. For every dataset found in any part file, it creates a virtual dataset whose rows are the part files' rows in server order; a part file without that dataset leaves a gap as long as its "/timestamp" dataset, filled with -99999 (integers) or NaN (floats).

        Notes
        -----
        The main file is opened with `_get_full_file_handle` and closed at the end.
        """

        joined_file = self._get_full_file_handle()

        # locate the srv (partial) files we expect
        files = []
        for i in range(self._server_group.Get_size()):
            srv_fn = _format_srv_filename(self._dirname, self._basename, i)
            if os.path.exists(srv_fn):
                files.append(srv_fn)
            else:
                print("!!! WARNING: expected partial (srv) file:")
                print(srv_fn)
                print("NOT FOUND. Trying to proceed with remaining data...")
                print("This almost certainly means something went wrong.")
        print("Joining: %d files --> %s" % (len(files), self._basename))

        # discover all the dataset names
        file_dsets = {}

        def assign_dset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                tmp_dsets[obj.name] = (obj.dtype, obj.shape)

        all_dsets = []
        for fn in files:
            tmp_dsets = {}
            f = h5py.File(fn, "r")
            f.visititems(assign_dset_info)
            file_dsets[fn] = tmp_dsets
            all_dsets += list(tmp_dsets.keys())
            f.close()

        all_dsets = set(all_dsets)

        # h5py requires you declare the size of the VDS at creation
        # (we have been told by Quincey Koziol that this is not
        # necessary for the C++ version).
        # so: we must first loop over each file to find # events
        #     that come from each file
        # then: do a second loop to join the data together

        for dset_name in all_dsets:
            # part (1) : loop over all files and get the total number
            # of events for this dataset
            total_events = 0
            for fn in files:
                dsets = file_dsets[fn]
                if dset_name in dsets.keys():
                    dtype, shape = dsets[dset_name]
                    total_events += shape[0]
                else:
                    if "/timestamp" in dsets:
                        total_events += dsets["/timestamp"][1][0]

            combined_shape = (total_events,) + shape[1:]

            layout = h5py.VirtualLayout(shape=combined_shape, dtype=dtype)

            # part (2): now that the number of events is known for this
            # dataset, fill in the "soft link" that points from the
            # master file to all the smaller files.
            index_of_last_fill = 0
            for fn in files:
                dsets = file_dsets[fn]

                if dset_name in dsets.keys():
                    _, shape = dsets[dset_name]
                    vsource = h5py.VirtualSource(fn, dset_name, shape=shape)
                    layout[
                        index_of_last_fill : index_of_last_fill + shape[0], ...
                    ] = vsource
                    index_of_last_fill += shape[0]

                else:
                    if "/timestamp" in dsets:
                        n_timestamps = dsets["/timestamp"][1][0]
                        index_of_last_fill += n_timestamps

            joined_file.create_virtual_dataset(
                dset_name, layout, fillvalue=_get_missing_value(dtype)
            )

        joined_file.close()

        return
