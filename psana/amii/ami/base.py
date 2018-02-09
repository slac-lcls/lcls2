import zmq
import threading
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
        else:
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


class ResultStore(ZmqBase):
    def __init__(self, zmq_config, zmqctx=None):
        super(__class__, self).__init__("resultstore", zmq_config, zmqctx)
        self._store = {}
        self._updated = {}
        self._sink = self.connect(zmq.PUSH)

    def collect(self):
        for name, result in self._store.items():
            if self._updated[name]:
                self.message(MsgTypes.Datagram, result)
                self._updated[name] = False

    def forward(self, msg):
        self._sink.send_pyobj(msg)

    def message(self, mtype, payload):
        self._sink.send_pyobj(Message(mtype, payload))

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
        self.listen_evt = threading.Event()
        self._listen_cb = callback
        self._listen_sock = self.connect(zmq.SUB, (zmq.SUBSCRIBE, self.listen_topic))
        self._listen_thread = threading.Thread(name="%s-listen"%self.name, target=self.listen)
        self._listen_thread.daemon = True
        self._listen_thread.start()

    def listen(self):
        while True:
            topic = self._listen_sock.recv_string()
            payload = self._listen_sock.recv_pyobj()
            if topic == self.listen_topic:
                print("%s: listerner recieved new payload with topic:"%self.name, topic)
                if self._listen_cb is not None:
                    self._listen_cb(payload)
                self.listen_evt.set()
            else:
                print("%s: recieved unexpected topic: %s"%(self.name, topic))


class ZmqCollector(ZmqBase):
    def __init__(self, name, zmq_config, zmqctx=None):
        super(__class__, self).__init__(name, zmq_config, zmqctx)
        self._transition_handler = None
        self._occurrence_handler = None
        self._datagram_handler = None 
        self.collect = self.bind(zmq.PULL)

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
            msg = self.collect.recv_pyobj()
            if msg.mtype == MsgTypes.Transition:
                if self._transition_handler is not None:
                    self._transition_handler(msg)
            elif msg.mtype == MsgTypes.Occurrence:
                if self._occurrence_handler is not None:
                    self._occurrence_handler(msg)
            elif msg.mtype == MsgTypes.Datagram:
                if self._datagram_handler is not None:
                    self._datagram_handler(msg)
