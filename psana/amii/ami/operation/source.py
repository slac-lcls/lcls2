import re
import zmq
import time
import threading
import numpy as np
from ami.operation.base import Operation, Datagram


class ExternDataSrc(Operation):
    def __init__(self, opid, ops):
        super(ExternDataSrc, self).__init__(opid, ops)


class FixedSrc(ExternDataSrc):
    def __init__(self, opid, ops):
        super(FixedSrc, self).__init__(opid, ops)
        self.config.require('sources', 'pedestal', 'width')
        self.node_slice = None
        self.num_slice = None
        self.config_lock = None
        self.src_thread = threading.Thread(target=self._start)
        self.src_thread.daemon = True

    def configure(self):
        if self.node_slice is None:
            self.event_id = 0
        else:
            self.event_id = self.node_slice
        return True

    def start(self, config_lock):
        self.config_lock = config_lock
        if not self.src_thread.is_alive():
            self.src_thread.start()

    def _update_outputs(self):
        if not self.outputs:
            for name, shape in self.config['sources']:
                self.outputs[name] = np.random.normal(self.config['pedestal'], self.config['width'], shape)

    def _start(self):
        while True:
            self.config_lock.acquire()
            if self.num_slice is None:
                self.event_id += 1
            else:
                self.event_id += self.num_slice
            self._update_outputs()
            for operation_id, input_name, output_id  in self.config['outputs']:
                dg = Datagram(self.event_id, self.config_id, self.outputs[output_id])
                self.ops[operation_id]._run(input_name, dg)
            self.config_lock.release()
            time.sleep(1)


class RandomSrc(FixedSrc):
    def __init__(self, opid, ops):
        super(RandomSrc, self).__init__(opid, ops)

    def _update_outputs(self):
        for name, shape in self.config['sources']:
            self.outputs[name] = np.random.normal(self.config['pedestal'], self.config['width'], shape)


class ZmqSrc(ExternDataSrc):
    def __init__(self, opid, ops):
        super(ZmqSrc, self).__init__(opid, ops)
        self.config.require('host', 'port')
        self.node_slice = None
        self.topics = {}
        self.context = zmq.Context()
        self.receiver = None
        self.config_lock = None
        self.src_thread = threading.Thread(target=self._start)
        self.src_thread.daemon = True

    def _add_outputs(self):
        self._clear_outputs()
        self.topics.clear() 
        if 'outputs' in self.config:
            for opkey, inpkey, topic in self.config['outputs']:
                self.outputs[opkey] = (self.ops[opkey], inpkey)        
                if topic in self.topics:
                    self.topics[topic].append((self.ops[opkey], inpkey))
                else:
                    self.topics[topic] = [(self.ops[opkey], inpkey)]

    def configure(self):
        if self.receiver is None:
            self.receiver = self.context.socket(zmq.SUB)
            if self.node_slice is None:
                self.topic_matcher = re.compile("(?P<topic>.*)\0")
                self.receiver.setsockopt_string(zmq.SUBSCRIBE, u"")
            else:
                self.topic_matcher = re.compile("(?:slice%03d\0)(?P<topic>.*)\0"%self.node_slice)
                self.receiver.setsockopt_string(zmq.SUBSCRIBE, u"slice%03d\0"%self.node_slice)
            self.receiver.connect("tcp://%s:%d"%(self.host, self.port))
        elif self.receiver.get_string(zmq.LAST_ENDPOINT) != "tcp://%s:%d"%(self.host, self.port):
            self.receiver.disconnect(self.receiver.get_string(zmq.LAST_ENDPOINT))
            self.receiver.connect("tcp://%s:%d"%(self.host, self.port))
        return True

    def start(self, config_lock):
        self.config_lock = config_lock
        if not self.src_thread.is_alive():
            self.src_thread.start()

    def _start(self):
        while True:
            topic = self.receiver.recv_string()
            outdata = self.receiver.recv_pyobj()
            topic_match = self.topic_matcher.match(topic)
            if topic_match:
                topic = topic_match.group('topic')
                if topic in self.topics:
                    self.config_lock.acquire()
                    for output, key in self.topics[topic]:
                        output._run(key, outdata)
                    self.config_lock.release()


class GRBase(Operation):
    def __init__(self, opid, ops, target):
        super(GRBase, self).__init__(opid, ops)
        self.config.require('port')
        self.gr = None
        self.topics = {}
        self.context = zmq.Context()
        self.count = {}
        self.running_data = {}
        self.evt_ids = {}
        self.config_lock = None
        self.gather_thread = threading.Thread(target=target)
        self.gather_thread.daemon = True

    def _add_outputs(self):
        self._clear_outputs()
        self.topics.clear()
        if 'outputs' in self.config:
            for opkey, inpkey, topic in self.config['outputs']:
                self.outputs[opkey] = (self.ops[opkey], inpkey)
                if topic in self.topics:
                    self.topics[topic].append((self.ops[opkey], inpkey))
                else:
                    self.topics[topic] = [(self.ops[opkey], inpkey)]

    def clear_gather(self):
        self.count.clear()
        self.running_data.clear()
        self.evt_ids.clear()

    def configure(self):
        self.clear_gather()
        if self.gr is None:
            self.gr = self.context.socket(zmq.PULL)
            self.gr.bind("tcp://*:%d"%self.port)
        elif self.gr.get_string(zmq.LAST_ENDPOINT) != "tcp://0.0.0.0:%d"%self.port:
            self.gr.unbind(self.gr.get_string(zmq.LAST_ENDPOINT))
            self.gr.bind("tcp://*:%d"%self.port)
        return True

    def start(self, config_lock):
        self.config_lock = config_lock
        if not self.gather_thread.is_alive():
            self.gather_thread.start()


class Reduce(GRBase):
    def __init__(self, opid, ops):
        super(Reduce, self).__init__(opid, ops, self._start)
        self.config.require('num_average')

    def _start(self):
        while True:
            topic = self.gr.recv_string()
            outdata = self.gr.recv_pyobj()
            if topic in self.topics:
                self.config_lock.acquire()
                if topic in self.count:
                    self.count[topic] += 1
                    self.evt_ids[topic].append(outdata.event_id)
                    self.running_data[topic] += outdata.data
                else:
                    self.count[topic] = 1
                    self.evt_ids[topic] = [outdata.event_id]
                    self.running_data[topic] = outdata.data
                if self.count[topic] == self.num_average:
                    outdata.event_id = self.evt_ids[topic]
                    outdata.data = self.running_data[topic] / self.count[topic]
                    for output, key in self.topics[topic]:
                        output._run(key, outdata)
                    del self.count[topic]
                self.config_lock.release()
