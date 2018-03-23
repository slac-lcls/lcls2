
import sys
import abc
import zmq
import threading
from mpi4py import MPI
from enum import IntEnum

from ami.data import MsgTypes, Message, DataTypes, Datagram


# Dan and TJ:
# we don't think we need this any more

#class MpiHandler(object):
#    def __init__(self, col_rank):
#        """
#        col_rank : int
#            The rank of the target process that recieves data from
#            this process
#        """
#        self.col_rank = col_rank
#    
#    def send(self, msg):
#        MPI.COMM_WORLD.send(msg, dest=self.col_rank)
#
#    def recv(self):
#        return MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE)


class ResultStore(object):
    """
    This class is a AMI /graph node that collects results
    from a single process and has the ability to send them
    to another (via MPI). The sending end point is typically
    a Collector object.
    """

    def __init__(self, collector_rank):
        self.name = "resultstore"
        self._store = {}
        self._updated = {}
        self.collector_rank = collector_rank

    def collect(self):
        for name, result in self._store.items():
            if self._updated[name]:
                self.message(MsgTypes.Datagram, result)
                self._updated[name] = False

    def send(self, msg):
        MPI.COMM_WORLD.send(msg, dest=self.collector_rank)

    def message(self, mtype, payload):
        self.send(Message(mtype, payload))

    def create(self, name, datatype=DataTypes.Unset):
        if name in self._store:
            raise ValueError("result named %s already exists in ResultStore"%name)
        else:
            self._store[name] = Datagram(name, datatype)
            self._updated[name] = False

    def is_updated(self, name):
        return self._updated[name]

    def get_dgram(self, name):
        return self._store[name]

    def get(self, name):
        return self._store[name].data

    def put_dgram(self, dgram):
        self.put(dgram.name, dgram.data)

    def put(self, name, data):
        datatype = DataTypes.get_type(data)
        if name in self._store:
            if datatype == self._store[name].dtype or self._store[name].dtype == DataTypes.Unset:
                self._store[name].dtype = datatype
                self._store[name].data = data
                self._updated[name] = True
            else:
                raise TypeError("type of new result (%s) differs from existing"
                                " (%s)"%(datatype, self._store[name].dtype))
        else:
            self._store[name] = Datagram(name, datatype, data)
            self._updated[name] = True

    def clear(self):
        self._store = {}


class Collector(abc.ABC):
    """
    This class gathers (via MPI) results from many
    ResultsStores. But rather than use gather, it employs
    an async send/recieve pattern.
    """

    def __init__(self):
        return

    def recv(self):
        return MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE)

    @abc.abstractmethod
    def process_msg(self, msg):
        return

    def run(self):
        while True:
            msg = self.recv()
            self.process_msg(msg)
