
import sys
import zmq
import threading
from mpi4py import MPI
from enum import IntEnum
from ami.data import MsgTypes, Occurrences, Message, Datagram


class ZmqPorts(IntEnum):
    Collector = 0
    FinalCollector = 1
    Graph = 2
    Command = 3
    Offset = 10
    StartingPort = 15000


class ZmqConfig(object):
    def __init__(self, platform, binds, connects):
        self.platform = platform
        self.binds = self._fixup(binds)
        self.connects = self._fixup(connects)
        self.base = "tcp://%s:%d"

    def _fixup(self, cfg):
        for key, value in cfg.items():
            cfg[key] = (value[0], self.get_port(value[1]))
        return cfg

    def get_port(self, port_enum):
        return ZmqPorts.StartingPort + ZmqPorts.Offset * self.platform + port_enum

    def bind_addr(self, zmq_type):
        if zmq_type in self.binds:
            return self.base % self.binds[zmq_type]
            raise ValueError("Unsupported zmq type for bind:", zmq_type)

    def connect_addr(self, zmq_type):
        if zmq_type in self.connects:
            return self.base % self.connects[zmq_type]
        else:
            raise ValueError("Unsupported zmq type for connect:", zmq_type)
        

class ZmqBase(object):
    def __init__(self, name, zmq_config, zmqctx=None):
        self.name = name
        self.zmq_config = zmq_config
        if zmqctx is None:
            self._zmqctx = zmq.Context()
        else:
            self._zmqctx = zmqctx

    def bind(self, zmq_type, *opts):
        sock = self._zmqctx.socket(zmq_type)
        for opt in opts:
            sock.setsockopt_string(*opt)
        sock.bind(self.zmq_config.bind_addr(zmq_type))
        return sock

    def connect(self, zmq_type, *opts):
        sock = self._zmqctx.socket(zmq_type)
        for opt in opts:
            sock.setsockopt_string(*opt)
        sock.connect(self.zmq_config.connect_addr(zmq_type))
        return sock


class ZmqSender(ZmqBase):
    def __init__(self, name, zmq_config, zmqctx=None):
        super(__class__, self).__init__(name, zmq_config, zmqctx)
        self.sock = self.connect(zmq.PUSH)

    def send(self, msg):
        self.sock.send_pyobj(msg)


class ZmqReceiver(ZmqBase):
    def __init__(self, name, zmq_config, zmqctx=None):
        super(__class__, self).__init__(name, zmq_config, zmqctx)
        self.sock = self.bind(zmq.PULL)

    def recv(self):
        return self.sock.recv_pyobj()


class MpiHandler(object):
    def __init__(self, col_rank):
        """
        col_rank : int
            The rank of the target process that recieves data from
            this process
        """
        self.col_rank = col_rank
    
    def send(self, msg):
        MPI.COMM_WORLD.send(msg, dest=self.col_rank)

    def recv(self):
        return MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE)


class ResultStore(object):
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

    def create(self, name, datatype):
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
        self.put(dgram.name, dgram.dtype, dgram.data)

    def put(self, name, datatype, data):
        if name in self._store:
            if datatype == self._store[name].dtype:
                self._store[name].data = data
                self._updated[name] = True
            else:
                raise TypeError("type of new result (%s) differs from existing (%s)"%(datatype, self._store[name].dtype))
        else:
            self._store[name] = Datagram(name, datatype, data)
            self._updated[name] = True

    def clear(self):
        self._store = {}


class ZmqListener(ZmqBase):
    def __init__(self, name, topic, callback, zmq_config, zmqctx=None):
        super(__class__, self).__init__(name, zmq_config, zmqctx)
        self.listen_topic = topic
        self.listen_evt = False
        self._listen_cb = callback

    def listen(self):
        while True:
            topic = self._listen_sock.recv_string()
            payload = self._listen_sock.recv_pyobj()
            if topic == self.listen_topic:
                print("%s: listerner recieved new payload with topic:"%self.name, topic)
                if self._listen_cb is not None:
                    self._listen_cb(payload)
                self.listen_evt = True
            else:
                print("%s: recieved unexpected topic: %s"%(self.name, topic))


#class Listener(MpiHandler):
#    """
#    Listener with MPI
#    """
#    def __init__(self, name, topic, callback, handler):
#        super(__class__, self).__init__(name, handler)
#        self.listen_topic = topic
#        self.listen_evt = False
#        self._listen_cb = callback
#
#    def listen(self):
#        while True:
#            #topic = self._listen_sock.recv_string()
#            #payload = self._listen_sock.recv_pyobj()
#            payload = self.recv()
#            print("%s: listerner recieved new payload with topic:"%self.name, topic)
#            if self._listen_cb is not None:
#                self._listen_cb(payload)
#            self.listen_evt = True


class Collector(object):

    def __init__(self):
        self._transition_handler = None
        self._occurrence_handler = None
        self._datagram_handler = None

    def recv(self):
        return MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE)

    # TJ says: we should not use callbacks

    def set_transition_handler(self, handler):
        self._transition_handler = handler

    def set_occurrence_handler(self, handler):
        self._occurrence_handler = handler

    def set_datagram_handler(self, handler):
        self._datagram_handler = handler

    def set_handlers(self, handler):
        self._transition_handler = handler
        self._occurrence_handler = handler
        self._datagram_handler = handler
        
    def run(self):
        while True:
            msg = self.recv()
            if msg.mtype == MsgTypes.Transition:
                if self._transition_handler is not None:
                    self._transition_handler(msg)
            elif msg.mtype == MsgTypes.Occurrence:
                if self._occurrence_handler is not None:
                    self._occurrence_handler(msg)
            elif msg.mtype == MsgTypes.Datagram:
                if self._datagram_handler is not None:
                    self._datagram_handler(msg)
