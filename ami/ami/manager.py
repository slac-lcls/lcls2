import re
import sys
import zmq
import argparse
import threading
from ami.comm import Collector
from ami.data import MsgTypes


class Manager(Collector):
    """
    An AMI graph Manager is the control point for an
    active "tree" of workers. It is the final collection
    point for all results, broadcasts those results to 
    clients (e.g. plots/GUIs), and handles requests for
    configuration changes to the graph.
    """

    def __init__(self, gui_port):
        """
        protocol right now only tells you how to communicate with workers
        """
        super(__class__, self).__init__()
        self.feature_store = {}
        self.feature_req = re.compile("feature:(?P<name>.*)")
        self.graph = {}

        # ZMQ setup
        self.ctx = zmq.Context()
        self.comm = self.ctx.socket(zmq.REP)
        self.comm.bind("tcp://*:%d" % gui_port)

        self.client_listener_thread = threading.Thread(target=self.client_listener)
        self.client_listener_thread.daemon = True
        self.client_listener_thread.start()

    def process_msg(self, msg):
        if msg.mtype == MsgTypes.Datagram:
            self.feature_store[msg.payload.name] = msg.payload
            print(msg.payload)
            sys.stdout.flush()
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

    def client_listener(self):
        print('*** started client listen thread')
        sys.stdout.flush()
        while True:
            request = self.comm.recv_string()
            
            # check if it is a feature request
            if not self.feature_request(request):
                if request == 'get_features':
                    self.comm.send_pyobj(self.features)
                elif request == 'get_graph':
                    self.comm.send_pyobj(self.graph)
                elif request == 'set_graph':
                    self.graph = self.recv_graph()
                    self.apply_graph()
                else:
                    self.comm.send_string('error')

    def recv_graph(self):
        self.comm.recv_pyobj() # zmq for now, could be EPICS in future?
        return graph

    def apply_graph(self):
        MPI.COMM_WORLD.send(self.graph)
        return


# TJL note to self:
# we do not need this any more

#def main():
#    parser = argparse.ArgumentParser(description='AMII Manager App')
#
#    parser.add_argument(
#        '-H',
#        '--host',
#        default='*',
#        help='interface the AMII manager listens on (default: all)'
#    )
#
#    parser.add_argument(
#        '-p',
#        '--port',
#        type=int,
#        default=5557,
#        help='port for GUI-Manager communication'
#    )
#
#    args = parser.parse_args()
#
#    try:
#        manager = Manager(args.port)
#        return manager.run()
#    except KeyboardInterrupt:
#        print("Manager killed by user...")
#        return 0
#
#
#if __name__ == '__main__':
#    sys.exit(main())


