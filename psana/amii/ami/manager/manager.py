import zmq
import json

class Manager(object):
    def __init__(self, config_id, config):
        self.count = 0
        self.config = config
        self.config_id = config_id
        self.context = zmq.Context()
        self.comm = self.context.socket(zmq.REP)
        self.comm.bind("tcp://*:5555")
        self.sock = self.context.socket(zmq.PUB)
        self.sock.bind("tcp://*:5556")

    def send_cfg(self):
        if self.config is None:
            raise ValueError("must pass a configuration object if none is set in order to send")
        self.config_id += 1
        self.sock.send_string('config\0', zmq.SNDMORE)
        self.sock.send_pyobj(self.config_id, zmq.SNDMORE)
        self.sock.send_pyobj(self.config)

    def start(self):
        while True:
            request = self.comm.recv_string()
            if request == 'get_config':
                self.comm.send_pyobj(self.config)
            elif request == 'set_config':
                self.config = self.comm.recv_pyobj()
                self.comm.send_string('ok')
            elif request == 'apply_config':
                self.send_cfg()
                self.comm.send_string('ok')
            else:
                self.comm.send_pyobj('error')
