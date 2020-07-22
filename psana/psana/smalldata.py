
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

import os
import numpy as np
import h5py
from collections.abc import MutableMapping

# -----------------------------------------------------------------------------

from psana.psexp.tools import mode

if mode == 'mpi':
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
else:
    SIZE = 1

if SIZE > 1:
    MODE = 'PARALLEL'
else:
    MODE = 'SERIAL'

# -----------------------------------------------------------------------------

MISSING_INT   = -99999
MISSING_FLOAT = np.nan

INT_TYPES   = [int, np.int8, np.int16, np.int32, np.int64,
               np.int, np.uint8, np.uint16, np.uint32, np.uint64, np.uint]
FLOAT_TYPES = [float, np.float16, np.float32, np.float64, np.float128, np.float]

RAGGED_PREFIX   = 'ragged_'
UNALIGED_PREFIX = 'unaligned_'

def is_unaligned(dset_name):
    return dset_name.split('/')[-1].startswith(UNALIGED_PREFIX)

# -----------------------------------------------------------------------------


def _flatten_dictionary(d, parent_key='', sep='/'):
    """
    http://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
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
        raise ValueError('%s :: Invalid num type for missing data' % str(dtype))

    return missing_value


def _format_srv_filename(dirname, basename, rank):
    srv_basename = '%s_part%d.h5' % (basename.strip('.h5'), rank)
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

    def __init__(self, singleton_shape, dtype, cache_size):
        
        self.singleton_shape = singleton_shape
        self.dtype = dtype
        self.cache_size = cache_size

        # initialize
        self.data = np.empty((self.cache_size,) + self.singleton_shape,
                             dtype=self.dtype)
        self.reset()

        return

    def append(self, data):
        self.data[self.n_events,...] = data
        self.n_events += 1
        return

    def reset(self):
        self.n_events = 0
        return


class Server: # (hdf5 handling)

    def __init__(self, filename=None, smdcomm=None, cache_size=10000,
                 callbacks=[]):

        self.filename   = filename
        self.smdcomm    = smdcomm
        self.cache_size = cache_size
        self.callbacks  = callbacks

        # maps dataset_name --> (dtype, shape)
        self._dsets = {}

        # maps dataset_name --> CacheArray()
        self._cache = {}

        self.num_events_seen = 0

        if (self.filename is not None):
            self.file_handle = h5py.File(self.filename, 'w')

        return

    def recv_loop(self):

        num_clients_done = 0
        num_clients = self.smdcomm.Get_size() - 1
        while num_clients_done < num_clients:
            msg = self.smdcomm.recv(source=MPI.ANY_SOURCE)
            if type(msg) is list:
                self.handle(msg)
            elif msg == 'done':
                num_clients_done += 1

        return


    def handle(self, batch):

        for event_data_dict in batch:

            for cb in self.callbacks:
                cb(event_data_dict)

            if self.filename is not None:

                # to_backfill: list of keys we have seen previously
                #              we want to be sure to backfill if we
                #              dont see them
                to_backfill = list(self._dsets.keys())

                for dataset_name, data in event_data_dict.items():

                    if dataset_name not in self._dsets.keys():
                        self.new_dset(dataset_name, data)
                    else:
                        to_backfill.remove(dataset_name)
                    self.append_to_cache(dataset_name, data)

                for dataset_name in to_backfill:
                    if not is_unaligned(dataset_name):
                        self.backfill(dataset_name, 1)

            self.num_events_seen += 1

        return


    def new_dset(self, dataset_name, data):

        if type(data) == int:
            shape = ()
            maxshape = (None,)
            dtype = 'i8'
        elif type(data) == float:
            shape = ()
            maxshape = (None,)
            dtype = 'f8'
        elif hasattr(data, 'dtype'):
            shape = data.shape
            maxshape = (None,) + data.shape
            dtype = data.dtype
        else:
            raise TypeError('Type: %s not compatible' % type(data))

        self._dsets[dataset_name] = (dtype, shape)

        dset = self.file_handle.create_dataset(dataset_name,
                                               (0,) + shape, # (0,) -> expand dim
                                               maxshape=maxshape,
                                               dtype=dtype,
                                               chunks=(self.cache_size,) + shape)

        if not is_unaligned(dataset_name):
            self.backfill(dataset_name, self.num_events_seen)

        return


    def append_to_cache(self, dataset_name, data):

        if dataset_name not in self._cache.keys():
            dtype, shape = self._dsets[dataset_name]
            cache = CacheArray(shape, dtype, self.cache_size)
            self._cache[dataset_name] = cache
        else:
            cache = self._cache[dataset_name]

        cache.append(data)

        if cache.n_events == self.cache_size:
            self.write_to_file(dataset_name, cache)

        return


    def write_to_file(self, dataset_name, cache):
        dset = self.file_handle.get(dataset_name)
        new_size = (dset.shape[0] + cache.n_events,) + dset.shape[1:]
        dset.resize(new_size)
        # remember: data beyond n_events in the cache may be OLD
        dset[-cache.n_events:,...] = cache.data[:cache.n_events,...] 
        cache.reset()
        return


    def backfill(self, dataset_name, num_to_backfill):
        
        dtype, shape = self._dsets[dataset_name]

        missing_value = _get_missing_value(dtype) 
        fill_data = np.empty(shape, dtype=dtype)
        fill_data.fill(missing_value)
    
        for i in range(num_to_backfill):
            self.append_to_cache(dataset_name, fill_data)
        
        return


    def done(self):
        if (self.filename is not None):
            # flush the data caches (in case did not hit cache_size yet)
            for dset, cache in self._cache.items():
                if cache.n_events > 0:
                    self.write_to_file(dset, cache)
            self.file_handle.close()
        return



class SmallData: # (client)

    def __init__(self, server_group=None, client_group=None, 
                 filename=None, batch_size=10000, cache_size=None,
                 callbacks=[]):
        """
        Parameters
        ----------
        server_group : MPI.Group
            The MPI group to allocate to server processes

        client_group : MPI.Group
            The MPI group to allocate to client processes

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
        """

        self.batch_size = batch_size
        self._batch = []
        self._previous_timestamp = -1

        if cache_size is None:
            cache_size = batch_size
        if cache_size < batch_size:
            print('Warning: `cache_size` smaller than `batch_size`')
            print('setting cache_size -->', batch_size)
            cache_size = batch_size

        self._full_filename = filename
        if (filename is not None):
            self._basename = os.path.basename(filename)
            self._dirname  = os.path.dirname(filename)
        self._first_open = True # filename has not been opened yet

        if MODE == 'PARALLEL':

            self._server_group = server_group
            self._client_group = client_group

            # hide intermediate files -- join later via VDS
            if filename is not None:
                self._srv_filename = _format_srv_filename(self._dirname,
                                                          self._basename,
                                                          self._server_group.Get_rank())
            else:
                self._srv_filename = None

            self._comm_partition()
            if self._type == 'server':
                self._server = Server(filename=self._srv_filename, 
                                      smdcomm=self._srvcomm, 
                                      cache_size=cache_size,
                                      callbacks=callbacks)
                self._server.recv_loop()

        elif MODE == 'SERIAL':
            self._srv_filename = self._full_filename # dont hide file
            self._type = 'serial'
            self._server = Server(filename=self._srv_filename,
                                  cache_size=cache_size,
                                  callbacks=callbacks)

        return


    def _comm_partition(self):

        self._smalldata_group = MPI.Group.Union(self._server_group, self._client_group)
        self._smalldata_comm  = COMM.Create(self._smalldata_group)
        self._client_comm     = COMM.Create(self._client_group)

        # partition into comms
        n_srv = self._server_group.size
        if n_srv < 1:
            print(f'xxx n_srv={n_srv}')
            raise Exception('Attempting to run smalldata with no servers'
                            ' set env var PS_SRV_NODES to be 1 or more')

        if self._server_group.rank != MPI.UNDEFINED: # if in server group
            self._type = 'server'
            self._srv_color = self._server_group.rank
            self._srvcomm = self._smalldata_comm.Split(self._srv_color, 0) # rank=0
            if self._srvcomm.Get_size() == 1:
                print('WARNING: server has no associated clients!')
                print('This core is therefore idle... set PS_SRV_NODES')
                print('to be smaller, or increase the number of mpi cores')
        elif self._client_group.rank != MPI.UNDEFINED: # if in client group
            self._type = 'client'
            self._srv_color = self._client_group.rank % n_srv
            self._srvcomm = self._smalldata_comm.Split(self._srv_color, 
                                                       RANK+1) # keep rank order
        else:
            # we are some other node type
            self._type = 'other'

        return


    def _get_full_file_handle(self):
        """
        makes sure we overwrite on first open, but not after that
        """

        if MODE == 'PARALLEL':
            if self._first_open == True and self._full_filename is not None:
                fh = h5py.File(self._full_filename, 'w', libver='latest')
                self._first_open = False
            else:
                fh = h5py.File(self._full_filename, 'r+', libver='latest')

        elif MODE == 'SERIAL':
            fh = self._server.file_handle

        return fh


    def event(self, event, *args, **kwargs):
        """
        event: int, psana.event.Event
        """

        if type(event) is int:
            timestamp = event
        elif hasattr(event, 'timestamp'):
            timestamp = int(event.timestamp)
        else:
            raise ValueError('`event` must have a timestamp attribute')

        # collect all new data to add
        event_data_dict = {}
        event_data_dict.update(kwargs)
        for d in args:
            event_data_dict.update( _flatten_dictionary(d) )


        # check to see if the timestamp indicates a new event...

        #   >> multiple calls to self.event(...), same event as before
        if timestamp == self._previous_timestamp:
            self._batch[-1].update(event_data_dict)

        #   >> we have a new event
        elif timestamp > self._previous_timestamp:

            # if we have a "batch_size", ship events
            # (this avoids splitting events if we have multiple
            #  calls to self.event)
            if len(self._batch) >= self.batch_size:
                if MODE == 'SERIAL':
                    self._server.handle(self._batch)
                elif MODE == 'PARALLEL':
                    self._srvcomm.send(self._batch, dest=0)
                self._batch = []           

            event_data_dict['timestamp'] = timestamp
            self._previous_timestamp = timestamp
            self._batch.append(event_data_dict)

        else:
            # FIXME: cpo
            print('event data is "old", event timestamps'
                             ' must increase monotonically'
                             ' previous timestamp: %d, current: %d'
                             '' % (self._previous_timestamp, timestamp))
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
        if MODE == 'PARALLEL':
            if self._type == 'client':
                r = True
        elif MODE == 'SERIAL':
            r = True
        else:
            raise RuntimeError()
        return r


    def sum(self, value):
        return self._reduction(value, MPI.SUM)


    def _reduction(self, value, op):
        """
        perform a reduction across the worker MPI procs
        """

        # because only client nodes may have certain summary
        # variables, we collect the summary data on client
        # rank 0 -- later, we need to remember this client
        # is the one who needs to WRITE the summary data to disk!

        if MODE == 'PARALLEL':
            red_val = None

            if self._type == 'client':
                red_val = self._client_comm.reduce(value, op)

        elif MODE == 'SERIAL':
            red_val = value # just pass it through...

        return red_val


    def save_summary(self, *args, **kwargs):
        """
        Save 'summary data', ie any data that is not per-event (typically 
        slower, e.g. at the end of the job).

        Interface is identical to SmallData.event()

        Note: this function should be called in a SmallData.summary: block
        """
        if self._full_filename is None:
            print('Warning: smalldata not saving summary since no h5 filename specified')
            return

        # in parallel mode, only client rank 0 writes to file
        if MODE == 'PARALLEL':
            if self._client_comm.Get_rank() != 0:
                return

        # >> collect summary data
        data_dict = {}
        data_dict.update(kwargs)
        for d in args:
            data_dict.update( _flatten_dictionary(d) )

        # >> write to file
        fh = self._get_full_file_handle()
        for dataset_name, data in data_dict.items():
            if data is None:
                print('Warning: dataset "%s" was passed value: None'
                      '... ignoring that dataset' % dataset_name)
            else:
                fh[dataset_name] = data

        # we don't want to close the file in serial mode
        # this file is the server's main (only) file
        if MODE == 'PARALLEL':
            fh.close()

        return


    def done(self):
        """
        Finish any final communication and join partial files
        (in parallel mode).
        """

        # >> finish communication
        if self._type == 'client':
            # we want to send the finish signal to the server
            if len(self._batch) > 0:
                self._srvcomm.send(self._batch, dest=0)
            self._srvcomm.send('done', dest=0)

        elif self._type == 'server':
            self._server.done()

        elif self._type == 'serial':
            self._server.handle(self._batch)
            self._server.done()

        # stuff only one process should do in parallel mode
        if MODE == 'PARALLEL':
            if self._type != 'other': # other = not smalldata (Mona)
                self._smalldata_comm.barrier()

                # CLIENT rank 0 does all final file writing
                # this is because this client may write to the file
                # during "save_summary(...)" calls, and we want
                # ONE file owner
                if self._type == 'client' and self._full_filename is not None:
                    if self._client_comm.Get_rank() == 0:
                        self.join_files()

        return


    def join_files(self):
        """
        """

        joined_file = self._get_full_file_handle()

        # locate the srv (partial) files we expect
        files = []
        for i in range(self._server_group.Get_size()):
            srv_fn = _format_srv_filename(self._dirname,
                                          self._basename,
                                          i)
            if os.path.exists(srv_fn):
                files.append(srv_fn)
            else:
                print('!!! WARNING: expected partial (srv) file:')
                print(srv_fn)
                print('NOT FOUND. Trying to proceed with remaining data...')
                print('This almost certainly means something went wrong.')
        print('Joining: %d files --> %s' % (len(files), self._basename))

        # discover all the dataset names
        file_dsets = {}

        def assign_dset_info(name, obj):
            # TODO check if name contains unaligned, if so ignore
            if isinstance(obj, h5py.Dataset):
                tmp_dsets[obj.name] = (obj.dtype, obj.shape)

        all_dsets = []
        for fn in files:
            tmp_dsets = {}
            f = h5py.File(fn, 'r')
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

                # this happens if a dataset is completely missing in a file.
                # to maintain alignment, we need to extend the length by the
                # appropriate number and it will be filled in with the
                # "fillvalue" argument below.  if it's unaligned, then
                # we don't need to extend it at all.
                elif not is_unaligned(dset_name):
                    if '/timestamp' in dsets:
                        total_events += dsets['/timestamp'][1][0]

            combined_shape = (total_events,) + shape[1:]

            layout = h5py.VirtualLayout(shape=combined_shape, 
                                        dtype=dtype)

            # part (2): now that the number of events is known for this
            # dataset, fill in the "soft link" that points from the
            # master file to all the smaller files.
            index_of_last_fill = 0
            for fn in files:

                dsets = file_dsets[fn]

                if dset_name in dsets.keys():
                    _, shape = dsets[dset_name]
                    vsource = h5py.VirtualSource(fn, dset_name, shape=shape)
                    layout[index_of_last_fill:index_of_last_fill+shape[0], ...] = vsource
                    index_of_last_fill += shape[0]

                else:
                    # only need to pad aligned data with "fillvalue" argument below
                    if is_unaligned(dset_name):
                        pass
                    else:
                        if '/timestamp' in dsets:
                            n_timestamps = dsets['/timestamp'][1][0]
                            index_of_last_fill += n_timestamps

            joined_file.create_virtual_dataset(dset_name,
                                               layout,
                                               fillvalue=_get_missing_value(dtype)) 

        joined_file.close()

        return


