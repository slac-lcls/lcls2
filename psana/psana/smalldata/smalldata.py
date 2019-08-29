
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


def _h5py_create_or_append(file_handle, dataset_name, data):
    """
    Add data to an HDF5 dataset. If the dataset exists, append to it, otherwise
    create a new dataset then add to it.

    Parameters
    ----------
    file_handle : h5py.File
        The h5py File object

    dataset_name : str
        The dataset name

    data : np.ndarray
        The data
    """

    dset = file_handle.get(dataset_name)

    # create
    if dset is None:
        dset = file_handle.create_dataset(dataset_name, 
                                          (1,) + data.shape, 
                                          maxshape=(None,) + data.shape,
                                          dtype=data.dtype)  # TODO will not work for native float/int
                                          #chunks=np.product(data.shape)) # TODO optimize
        dset[:] = data

    # append
    else:
        if not dset.dtype == data.dtype:
            raise TypeError('dataset (%s) type does not match new data type'
                            '' % dset.name)
        if not dset.shape[1:] == data.shape:
            raise ValueError('dataset (%s) shape does not match new data shape'
                            '' % dset.name)
        new_size = (dset.shape[0] + 1,) + dset.shape[1:]
        dset.resize(new_size)
        dset[-1,...] = data

    return


class SmallData:

    def __init__(self, filename=None, batch_size=5):

        self.filename = filename
        self.batch_size = batch_size
        self._batch = []

        if MODE == 'PARALLEL':
            self._comm_partition()

            if self._type == 'server':
                if (self.filename is not None):
                    self.file_handle = h5py.File(self.filename)
                self._recv_loop()

        elif MODE == 'SERIAL':
            self._type = 'serial'
            if (self.filename is not None):
                self.file_handle = h5py.File(self.filename)            

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


    def done(self):

        if self._type == 'server':
            self.file_handle.close()

        elif self._type == 'client':
            # we want to send the finish signal to the server
            if len(self._batch) > 0:
                self._send(self._batch)
            self._smdcomm.send('done', dest=0)

        return


    def event(self, event, *args, **kwargs):

        # TODO : missing rows AND columns

        # collect all new data to add
        event_data_dict = {}
        event_data_dict.update(kwargs)
        for d in args:
            event_data_dict.update( _flatten_dictionary(d) )
        self._batch.append(event_data_dict)

        if len(self._batch) >= self.batch_size:
            if MODE == 'SERIAL':
                self._handle(self._batch)
            elif MODE == 'PARALLEL':
                self._send(self._batch)
            self._batch = []

        return


    def _send(self, batch):
        self._smdcomm.send(batch, dest=0)
        return


    def _recv_loop(self):

        num_clients_done = 0
        num_clients = self._smdcomm.Get_size() - 1
        while num_clients_done < num_clients:
            msg = self._smdcomm.recv(source=MPI.ANY_SOURCE)
            if type(msg) is list:
                self._handle(msg)
            elif msg == 'done':
                num_clients_done += 1

        return


    def _handle(self, batch):

        for event_data_dict in batch:

            for dataset_name, data in event_data_dict.items():
                if self.filename is not None:
                    _h5py_create_or_append(self.file_handle,
                                           dataset_name,
                                           np.array(data))

                # TODO also call all callbacks

        return


    def summary(self, *args, **kwargs):
        return


