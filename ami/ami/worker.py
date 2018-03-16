import re
import sys
import zmq
import time
import json
import argparse
import threading
from mpi4py import MPI
import multiprocessing as mp
from ami.manager import Manager
from ami.graph import Graph, GraphConfigError, GraphRuntimeError
from ami.comm import Collector, ResultStore
from ami.data import MsgTypes, DataTypes, Transitions, Occurrences, Message, Datagram, Transition, StaticSource


class Worker(object):
    def __init__(self, idnum, src, collector_rank):
        """
        idnum : int
            a unique integer identifying this worker
        src : object
            object with an events() method that is an iterable (like psana.DataSource)
        """
        self.idnum = idnum
        self.src = src
        self.collector_rank = collector_rank
        self.store = ResultStore(self.collector_rank)
        self.graph = Graph(self.store)
        self.new_graph_available = False

    def update_graph(self, new_graph):
        self.graph.update(new_graph)

    def run(self):
        partition = self.src.partition()
        self.store.message(MsgTypes.Transition, 
                           Transition(Transitions.Allocate, partition))
        for name, dtype in partition:
            self.store.create(name, dtype)
        for msg in self.src.events():
            # check to see if the graph has been reconfigured after update
            if msg.mtype == MsgTypes.Occurrence and msg.payload == Occurrences.Heartbeat:

                # TJL 3/9/18
                # THIS IS BROKEN
                # We need to connect the Manager's ZMQ to a special
                # kind of MPI message here saying there is a new graph

                if self.new_graph_available:
                    print("%s: Received new configuration"%self.name)
                    try:
                        self.graph.configure()
                        print("%s: Configuration complete"%self.name)
                    except GraphConfigError as graph_err:
                        print("%s: Configuration failed reverting to previous config:"%self.name, graph_err)
                        # if this fails we just die
                        self.graph.revert()
                    self.new_graph_available = False
                self.store.send(msg)
            elif msg.mtype == MsgTypes.Datagram:
                updates = []
                for dgram in msg.payload:
                    self.store.put_dgram(dgram)
                    updates.append(dgram.name)
                try:
                    self.graph.execute(updates)
                except GraphRuntimeError as graph_err:
                    print("%s: Failure encountered executing graph:"%self.name, graph_err)
                    return 1
                self.store.collect()
            else:
                self.store.forward(msg)


#class NodeCollector(Collector):
#    def __init__(self, num_workers, col_handler):
#        super(__class__, self).__init__("collector", col_handler)
#        self.num_workers = num_workers
#        # this does not really work - need a better way
#        self.counts = { MsgTypes.Transition: 0, MsgTypes.Occurrence: 0 }
#        self.set_handlers(self.store_msg)
#        self.upstream = ResultStore(col_handler)
#
#    def store_msg(self, msg):
#        if msg.mtype == MsgTypes.Transition:
#            print("%s: Seen Transition of type"%self.name, msg.payload.ttype)
#            self.counts[MsgTypes.Transition] += 1
#            if self.counts[MsgTypes.Transition] == self.num_workers:
#                if msg.payload.ttype == Transitions.Allocate:
#                    for name, dtype in msg.payload.payload:
#                        self.upstream.create(name, dtype)
#                #self.upstream.forward(msg)
#                self.counts[MsgTypes.Transition] = 0
#        elif msg.mtype == MsgTypes.Occurrence:
#            print("%s: Seen Occurence of type"%self.name, msg.payload)
#            self.counts[MsgTypes.Occurrence] += 1
#            if self.counts[MsgTypes.Occurrence] == self.num_workers:
#                if msg.payload == Occurrences.Heartbeat:
#                    self.upstream.collect()
#                #self.upstream.forward(msg)
#                self.counts[MsgTypes.Occurrence] = 0
#        elif msg.mtype == MsgTypes.Datagram:
#            print(msg.payload)
#            #self.upstream.put_dgram(msg.payload)

               
def run_worker(num, source, collector_rank=0):

    print('Starting worker # %d, sending to collector %d' % (num, collector_rank))
    sys.stdout.flush()

    if source[0] == 'static':
        try:
            with open(source[1], 'r') as cnf:
                src_cfg = json.load(cnf)
        except OSError as os_exp:
            print("worker%03d: problem opening json file:"%num, os_exp)
            return 1
        except json.decoder.JSONDecodeError as json_exp:
            print("worker%03d: problem parsing json file (%s):"%(num, source[1]), json_exp)
            return 1

        src = StaticSource(num, 
                           src_cfg['interval'], 
                           src_cfg['heartbeat'], 
                           src_cfg["init_time"], 
                           src_cfg['config'])
    else:
        print("worker%03d: unknown data source type:"%num, source[0])
    worker = Worker(num, src, collector_rank)
    sys.exit(worker.run())

def main():
    parser = argparse.ArgumentParser(description='AMII Worker/Collector App')

    parser.add_argument(
        '-H',
        '--host',
        default='localhost',
        help='hostname of the AMII Manager'
    )

    parser.add_argument(
        '-p',
        '--port',
        type=int,
        default=5556,
        help='port for GUI-Manager communication (via zmq)'
    )

    parser.add_argument(
        '-n',
        '--num-workers',
        type=int,
        default=1,
        help='number of worker processes'
    )

    parser.add_argument(
        '-N',
        '--node-num',
        type=int,
        default=1,
        help='node identification number'
    )

    parser.add_argument(
        'source',
        metavar='SOURCE',
        help='data source configuration (exampes: static://test.json, psana://exp=xcsdaq13:run=14)'
    )

    args = parser.parse_args()

    try:
        src_url_match = re.match('(?P<prot>.*)://(?P<body>.*)', args.source)
        if src_url_match:
            src_cfg = src_url_match.groups()
        else:
            print("Invalid data source config string:", args.source)
            return 1
            
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if rank == 0:
            m = Manager(args.port)
            m.run()
        else:
            run_worker(rank, src_cfg, collector_rank=0)

    except KeyboardInterrupt:
        print("Worker killed by user...")
        return 0


if __name__ == '__main__':
    sys.exit(main())

 
