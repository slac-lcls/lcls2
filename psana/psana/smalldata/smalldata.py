
import os
import numpy as np
import h5py
import collections

# -----------------------------------------------------------------------------

from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

if SIZE > 1:
    MODE = 'PARALLEL'
    # these are all functions that provide xface to MPI pool
    from psana.psexp.node import node_type
    from psana.psexp.node import srv_group
    from psana.psexp.node import bd_group

    smalldata_group = MPI.Group.Union(srv_group(), bd_group())
    smalldata_comm  = COMM.Create(smalldata_group)

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



# FOR NEXT TIME
# CONSIDER MAKING A FileServer CLASS
# CLASS BASECLASS METHOD THEN HANDLES HDF5


class Server: # (hdf5 handling)

    def __init__(self, filename=None, smdcomm=None):

        self.filename = filename
        self.smdcomm  = smdcomm

        # dsets maps dataset_name --> (dtype, shape)
        self._dsets = {}
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
            self.num_events_seen += 1

            # TODO > CALLBACKS HERE <

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
                    self.append_to_dset(dataset_name, data)

                for dataset_name in to_backfill:
                    self.backfill(dataset_name, 1)

        return


    def new_dset(self, dataset_name, data):

        if type(data) == int:
            shape = (0,)
            maxshape = (None,)
            dtype = 'i8'
        elif type(data) == float:
            shape = (0,)
            maxshape = (None,)
            dtype = 'f8'
        elif hasattr(data, 'dtype'):
            shape = (0,) + data.shape
            maxshape = (None,) + data.shape
            dtype = data.dtype
        else:
            raise TypeError('Type: %s not compatible' % type(data))

        self._dsets[dataset_name] = (dtype, shape)

        dset = self.file_handle.create_dataset(dataset_name,
                                               shape,
                                               maxshape=maxshape,
                                               dtype=dtype)

        self.backfill(dataset_name, self.num_events_seen)

        return


    def append_to_dset(self, dataset_name, data):
        dset = self.file_handle.get(dataset_name)
        new_size = (dset.shape[0] + 1,) + dset.shape[1:]
        dset.resize(new_size)
        dset[-1,...] = data
        return


    def backfill(self, dataset_name, num_to_backfill):
        return


    def done(self):
        if (self.filename is not None):
            self.file_handle.close()
        return



class SmallData: # (client)

    def __init__(self, filename=None, batch_size=5):

        self.batch_size = batch_size
        self._batch = []

        if MODE == 'PARALLEL':
            self._comm_partition()
            if self._type == 'server':
                self._server = Server(filename=filename, smdcomm=self._smdcomm)
                self._server.recv_loop()

        elif MODE == 'SERIAL':
            self._type = 'serial'
            self._server = Server(filename=filename)

        return


    def _comm_partition(self):
    
        # partition into comms
        _srv_group = srv_group()
        n_srv = _srv_group.size
        if n_srv < 1:
            raise Exception('Attempting to run smalldata with no servers'
                            ' set env var PS_SRV_NODES to be 1 or more')

        if node_type() == 'srv':
            self._type = 'server'
            self._srv_color = _srv_group.rank
            self._smdcomm = smalldata_comm.Split(self._srv_color, 0) # rank=0
        elif node_type() == 'bd':
            self._type = 'client'
            self._srv_color = bd_group().rank % n_srv
            self._smdcomm = smalldata_comm.Split(self._srv_color, 
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
        self._batch.append(event_data_dict)

        if len(self._batch) >= self.batch_size:
            if MODE == 'SERIAL':
                self._server.handle(self._batch)
            elif MODE == 'PARALLEL':
                self._smdcomm.send(self._batch, dest=0)
            self._batch = []

        return


    def done(self):

        if self._type == 'client':
            # we want to send the finish signal to the server
            if len(self._batch) > 0:
                self._smdcomm.send(self._batch, dest=0)
            self._smdcomm.send('done', dest=0)
        elif self._type == 'server':
            self._server.done()

        return


