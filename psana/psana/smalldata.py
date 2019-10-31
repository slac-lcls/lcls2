
import os
import numpy as np
import h5py
import collections

# -----------------------------------------------------------------------------

from psana.psexp.tools import mode
SIZE = 1
if mode == 'mpi':
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()

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
        if isinstance(v, collections.MutableMapping):
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


# FOR NEXT TIME
# CONSIDER MAKING A FileServer CLASS
# CLASS BASECLASS METHOD THEN HANDLES HDF5

class CacheArray:

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

    def __init__(self, filename=None, smdcomm=None, cache_size=5,
                 callbacks=[]):

        self.filename   = filename
        self.smdcomm    = smdcomm
        self.cache_size = cache_size
        self.callbacks  = callbacks

        # dsets maps dataset_name --> (dtype, shape)
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
                                               dtype=dtype)

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
                 filename=None, batch_size=5, cache_size=5,
                 callbacks=[]):

        self.batch_size = batch_size
        self._batch = []
        self._previous_timestamp = -1

        self._full_filename = filename
        self._basename = os.path.basename(filename)
        self._dirname  = os.path.dirname(filename)

        if MODE == 'PARALLEL':

            self._server_group = server_group
            self._client_group = client_group

            # hide intermediate files -- join later via VDS
            self._hidden_filename = os.path.join(self._dirname,
                                                 '.' + str(self._server_group.Get_rank()) + '_' + self._basename)

            self._comm_partition()
            if self._type == 'server':
                self._server = Server(filename=self._hidden_filename, 
                                      smdcomm=self._srvcomm, 
                                      cache_size=cache_size,
                                      callbacks=callbacks)
                self._server.recv_loop()

        elif MODE == 'SERIAL':
            self._hidden_filename = self._full_filename # dont hide file
            self._type = 'serial'
            self._server = Server(filename=self._hidden_filename,
                                  cache_size=cache_size,
                                  callbacks=callbacks)

        return


    def _comm_partition(self):

        self._smalldata_group = MPI.Group.Union(self._server_group, self._client_group)
        self._smalldata_comm  = COMM.Create(self._smalldata_group)

        # partition into comms
        n_srv = self._server_group.size
        if n_srv < 1:
            raise Exception('Attempting to run smalldata with no servers'
                            ' set env var PS_SRV_NODES to be 1 or more')

        if self._server_group.rank != MPI.UNDEFINED: # if in server group
            self._type = 'server'
            self._srv_color = self._server_group.rank
            self._srvcomm = self._smalldata_comm.Split(self._srv_color, 0) # rank=0
        elif self._client_group.rank != MPI.UNDEFINED: # if in client group
            self._type = 'client'
            self._srv_color = self._client_group.rank % n_srv
            self._srvcomm = self._smalldata_comm.Split(self._srv_color, 
                                                 RANK+1) # keep rank order
        else:
            # we are some other node type
            self._type = 'other'

        return


    def event(self, event, *args, **kwargs):

        # collect all new data to add
        event_data_dict = {}
        event_data_dict.update(kwargs)
        for d in args:
            event_data_dict.update( _flatten_dictionary(d) )


        # check to see if the timestamp indicates a new event

        # multiple calls to self.event(...), same event as before
        if event.timestamp == self._previous_timestamp:
            self._batch[-1].update(event_data_dict)

        # we have a new event
        elif event.timestamp > self._previous_timestamp:

            # if we have a "batch_size", ship events
            # (this avoids splitting events if we have multiple
            #  calls to self.event)
            if len(self._batch) >= self.batch_size:
                if MODE == 'SERIAL':
                    self._server.handle(self._batch)
                elif MODE == 'PARALLEL':
                    self._srvcomm.send(self._batch, dest=0)
                self._batch = []           

            event_data_dict['timestamp'] = event.timestamp
            self._previous_timestamp = event.timestamp
            self._batch.append(event_data_dict)


        else:
            raise IndexError('event data is "old", event timestamps'
                             ' must increase monotonically'
                             ' previous timestamp: %d, current: %d'
                             '' % (previous_timestamp, event.timestamp))


        return


    def done(self):

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

        if MODE == 'PARALLEL':
            if self._type != 'other': # other = Mona
                self._smalldata_comm.barrier()
                if self._smalldata_comm.Get_rank() == 0:
                    self.join_files()

        return


    def join_files(self):
        """
        """

        joined_file = h5py.File(self._full_filename, 'w', libver='latest')

        # locate the hidden files we expect
        files = []
        for i in range(self._server_group.Get_size()):
            hidden_fn = os.path.join(self._dirname,
                                     '.' + str(i) + '_' + self._basename)
            if os.path.exists(hidden_fn):
                files.append(hidden_fn)
            else:
                print('!!! WARNING: expected hidden file:')
                print(hidden_fn)
                print('NOT FOUND. Trying to proceed with remaining data...')
                print('This almost certainly means something went wrong.')
        print('Joining: %d files --> %s' % (len(files), self._basename))

        # h5py requires you declare the size of the VDS at creation
        # so: we must first loop over each file to find # events
        #     that come from each file
        # then do a second loop to join the data together

        # part (1) : discover the size of the timestamps in each file
        file_dsets = {}

        def assign_dset_info(name, obj):
            # TODO check if name contains unaligned, if so ignore
            if isinstance(obj, h5py.Dataset):
                dsets[obj.name] = (obj.dtype, obj.shape)

        for fn in files:
            dsets = {}
            f = h5py.File(fn, 'r')
            f.visititems(assign_dset_info)
            file_dsets[fn] = dsets
            f.close()

        # part (2) : loop over datasets and combine them into a vds
        for dset_name in dsets.keys():

            # inspect the first file (w data) to get basic shape & dtype

            total_events = 0
            for fn in files:
                dsets = file_dsets[fn]
                if dset_name in dsets.keys():
                    dtype, shape = dsets[dset_name]
                    total_events += shape[0]

                # we need to reserve space for missing data for aligned data
                elif not is_unaligned(dset_name):
                    total_events += dsets['/timestamp'][1][0]

            combined_shape = (total_events,) + shape[1:]

            layout = h5py.VirtualLayout(shape=combined_shape, 
                                        dtype=dtype)

            # add data for "dset", from each file, in order
            index_of_last_fill = 0
            for fn in files:

                dsets = file_dsets[fn]

                if dset_name in dsets.keys():
                    _, shape = dsets[dset_name]
                    vsource = h5py.VirtualSource(fn, dset_name, shape=shape)
                    layout[index_of_last_fill:index_of_last_fill+shape[0], ...] = vsource
                    index_of_last_fill += shape[0]

                else:
                    if is_unaligned(dset_name):
                        pass
                    else: # should be aligned
                        n_timestamps = dsets['/timestamp'][1][0]
                        index_of_last_fill += n_timestamps

            joined_file.create_virtual_dataset(dset_name,
                                               layout,
                                               fillvalue=_get_missing_value(dtype)) 

        joined_file.close()

        return
