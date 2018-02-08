import zmq
import threading
from enum import IntEnum
from ami.data import MsgTypes, Occurrences


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
    def __init__(self, name, host_config, zmqctx=None):
        self.name = name
        self.host_config = host_config
        if zmqctx is None:
            self._zmqctx = zmq.Context()
        else:
            self._zmqctx = zmqctx

    def bind(self, zmq_type, *opts):
        sock = self._zmqctx.socket(zmq_type)
        for opt in opts:
            sock.setsockopt_string(*opt)
        sock.bind(self.host_config.bind_addr(zmq_type))
        return sock

    def connect(self, zmq_type, *opts):
        sock = self._zmqctx.socket(zmq_type)
        for opt in opts:
            sock.setsockopt_string(*opt)
        sock.connect(self.host_config.connect_addr(zmq_type))
        return sock

class ZmqListener(ZmqBase):
    def __init__(self, name, topic, callback, host_config, zmqctx=None):
        super(__class__, self).__init__(name, host_config, zmqctx)
        self.listen_topic = topic
        self.listen_evt = threading.Event()
        self._listen_cb = callback
        self._listen_sock = self.connect(zmq.SUB, (zmq.SUBSCRIBE, self.listen_topic))
        self._listen_thread = threading.Thread(name="%s-listen"%self.name, target=self.listen)
        self._listen_thread.daemon = True
        self._listen_thread.start()

    def listen(self):
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
    def __init__(self, name, host_config, zmqctx=None):
        super(__class__, self).__init__(name, host_config, zmqctx)
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
