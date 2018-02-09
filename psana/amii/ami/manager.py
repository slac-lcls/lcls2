import re
import sys
import zmq
import argparse
import threading
from ami.base import ZmqPorts, ZmqConfig, ZmqCollector


class Manager(ZmqCollector):
    def __init__(self, name, host_config, zmqctx=None):
        super(__class__, self).__init__("manager", host_config, zmqctx)
        self.feature_store = {}
        self.feature_req = re.compile("feature:(?P<name>.*)")
        self.graph = {}
        self.comm = self.bind(zmq.REP)
        self.graph_comm = self.bind(zmq.PUB)
        self.set_datagram_handler(self.publish)
        #self.set_occurence_handler(self.publish)
        self.cmd_thread = threading.Thread(name="%s-command"%name, target=self.command_listener)
        self.cmd_thread.daemon = True
        self.cmd_thread.start()

    @property
    def features(self):
        dets = {}
        for key, value in self.feature_store.items():
            dets[key] = value.dtype
        return dets

    def publish(self, msg):
        self.feature_store[msg.payload.name] = msg.payload
        #print(msg.payload)

    def apply_graph(self):
        self.graph_comm.send_string('graph', zmq.SNDMORE)
        self.graph_comm.send_pyobj(self.graph)

    def feature_request(self, request):
        matched = self.feature_req.match(request)
        if matched:
            if matched.group('name') in self.feature_store:
                self.comm.send_string('ok', zmq.SNDMORE)
                self.comm.send_pyobj(self.feature_store[matched.group('name')].data)
            else:
                self.comm.send_string('error')
            return True
        else:
            return False

    def command_listener(self):
        while True:
            request = self.comm.recv_string()
            
            # check if it is a feature request
            if not self.feature_request(request):
                if request == 'get_features':
                    self.comm.send_pyobj(self.features)
                elif request == 'get_graph':
                    self.comm.send_pyobj(self.graph)
                elif request == 'set_graph':
                    self.graph = self.comm.recv_pyobj()
                    self.comm.send_string('ok')
                elif request == 'apply_graph':
                    self.apply_graph()
                    self.comm.send_string('ok')
                else:
                    self.comm.send_string('error')


def main():
    parser = argparse.ArgumentParser(description='AMII Manager App')

    parser.add_argument(
        '-H',
        '--host',
        default='*',
        help='interface the AMII manager listens on (default: all)'
    )

    parser.add_argument(
        '-p',
        '--platform',
        type=int,
        default=0,
        help='platform number of the AMII - selects port range to use (default: 0)'
    )

    args = parser.parse_args()
    manager_cfg = ZmqConfig(
        platform=args.platform,
        binds={zmq.PULL: (args.host, ZmqPorts.FinalCollector), zmq.PUB: (args.host, ZmqPorts.Graph), zmq.REP: (args.host, ZmqPorts.Command)},
        connects={}
    )

    try:
        manager = Manager("manager", manager_cfg)
        return manager.run()
    except KeyboardInterrupt:
        print("Worker killed by user...")
        return 0


if __name__ == '__main__':
    sys.exit(main())
