import re
import sys
import zmq
import argparse
from ami.comm import Ports, Collector
from ami.data import MsgTypes


class Manager(Collector):
    """
    An AMI graph Manager is the control point for an
    active "tree" of workers. It is the final collection
    point for all results, broadcasts those results to 
    clients (e.g. plots/GUIs), and handles requests for
    configuration changes to the graph.
    """

    def __init__(self, collector_addr, graph_addr, comm_addr):
        """
        protocol right now only tells you how to communicate with workers
        """
        super(__class__, self).__init__(collector_addr)
        self.feature_store = {}
        self.feature_req = re.compile("feature:(?P<name>.*)")
        self.graph = {}

        self.comm = self.ctx.socket(zmq.REP)
        self.comm.bind(comm_addr)
        self.register(self.comm, self.client_request)
        self.graph_comm = self.ctx.socket(zmq.PUB)
        self.graph_comm.bind(graph_addr)

    def process_msg(self, msg):
        if msg.mtype == MsgTypes.Datagram:
            self.feature_store[msg.payload.name] = msg.payload
        return

    @property
    def features(self):
        dets = {}
        for key, value in self.feature_store.items():
            dets[key] = value.dtype
        return dets

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

    def client_request(self):
        request = self.comm.recv_string()
        # check if it is a feature request
        if not self.feature_request(request):
            if request == 'get_features':
                self.comm.send_pyobj(self.features)
            elif request == 'get_graph':
                self.comm.send_pyobj(self.graph)
            elif request == 'set_graph':
                self.graph = self.recv_graph()
                if self.apply_graph():
                    self.comm.send_string('ok')
                else:
                    self.comm.send_string('error')
            else:
                self.comm.send_string('error')

    def recv_graph(self):
        return self.comm.recv_pyobj() # zmq for now, could be EPICS in future?

    def apply_graph(self):
        print("manager: sending requested graph...")
        try:
            self.graph_comm.send_string("graph", zmq.SNDMORE)
            self.graph_comm.send_pyobj(self.graph)
        except Exception as exp:
            print("manager: failed to send graph -", exp)
            return False
        print("manager: sending of graph completed")
        return True


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
        '--port',
        type=int,
        default=Ports.Comm,
        help='port for GUI-Manager communication (default: %d)'%Ports.Comm
    )

    parser.add_argument(
        '-g',
        '--graph',
        type=int,
        default=Ports.Graph,
        help='port for graph communication (default: %d)'%Ports.Graph
    )

    parser.add_argument(
        '-c',
        '--collector',
        type=int,
        default=Ports.Collector,
        help='port for final collector (default: %d)'%Ports.Collector
    )

    args = parser.parse_args()

    collector_addr = "tcp://%s:%d"%(args.host, args.collector)
    graph_addr = "tcp://%s:%d"%(args.host, args.graph)
    comm_addr = "tcp://%s:%d"%(args.host, args.port)

    try:
        manager = Manager(collector_addr, graph_addr, comm_addr)
        return manager.run()
    except KeyboardInterrupt:
        print("Manager killed by user...")
        return 0


if __name__ == '__main__':
    sys.exit(main())
