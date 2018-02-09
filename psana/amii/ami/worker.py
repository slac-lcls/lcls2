import re
import sys
import zmq
import time
import json
import argparse
import threading
import multiprocessing as mp
from ami.graph import Graph, GraphConfigError, GraphRuntimeError
from ami.base import ZmqPorts, ZmqConfig, ZmqBase, ZmqListener, ZmqCollector, ResultStore
from ami.data import MsgTypes, DataTypes, Transitions, Occurrences, Message, Datagram, Transition, StaticSource


class Worker(ZmqListener):
    def __init__(self, idnum, host_config, src, zmqctx=None):
        super(__class__, self).__init__("worker%03d"%idnum, 'graph', self.update_graph, host_config, zmqctx)
        self.idnum = idnum
        self.src = src
        self.store = ResultStore(host_config, self._zmqctx)
        self.graph = Graph(self.store)

    def update_graph(self, new_graph):
        self.graph.update(new_graph)

    def run(self):
        partition = self.src.partition()
        self.store.message(MsgTypes.Transition, Transition(Transitions.Allocate, partition))
        for name, dtype in partition:
            self.store.create(name, dtype)
        for msg in self.src.events():
            # check to see if the graph has been reconfigured after update
            if self.listen_evt.is_set():
                print("%s: Received new configuration"%self.name)
                try:
                    self.graph.configure()
                    print("%s: Configuration complete"%self.name)
                except GraphConfigError as graph_err:
                    print("%s: Configuration failed reverting to previous config:"%self.name, graph_err)
                    # if this fails we just die
                    self.graph.revert()
                self.listen_evt.clear()
            if msg.mtype == MsgTypes.Datagram:
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


class Collector(ZmqCollector):
    def __init__(self, num_workers, host_config, zmqctx=None):
        super(__class__, self).__init__("collector", host_config, zmqctx)
        self.num_workers = num_workers
        # this does not really work - need a better way
        self.counts = { MsgTypes.Transition: 0, MsgTypes.Occurrence: 0 }
        self.set_handlers(self.store_msg)
        self.upstream = ResultStore(host_config, self._zmqctx)

    def store_msg(self, msg):
        if msg.mtype == MsgTypes.Transition:
            print("%s: Seen Transition of type"%self.name, msg.payload.ttype)
            self.counts[MsgTypes.Transition] += 1
            if self.counts[MsgTypes.Transition] == self.num_workers:
                if msg.payload.ttype == Transitions.Allocate:
                    for name, dtype in msg.payload.payload:
                        self.upstream.create(name, dtype)
                self.upstream.forward(msg)
                self.counts[MsgTypes.Transition] = 0
        elif msg.mtype == MsgTypes.Occurrence:
            print("%s: Seen Occurence of type"%self.name, msg.payload)
            self.counts[MsgTypes.Occurrence] += 1
            if self.counts[MsgTypes.Occurrence] == self.num_workers:
                if msg.payload == Occurrences.Heartbeat:
                    self.upstream.collect()
                self.upstream.forward(msg)
                self.counts[MsgTypes.Occurrence] = 0
        elif msg.mtype == MsgTypes.Datagram:
            self.upstream.put_dgram(msg.payload)

                
def run_worker(num, host_config, source):
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
        src = StaticSource(num, src_cfg['interval'], src_cfg['heartbeat'], src_cfg["init_time"], src_cfg['config'])
    else:
        print("worker%03d: unknown data source type:"%num, source[0])
    worker = Worker(num, host_config, src)
    sys.exit(worker.run())


def run_collector(num, host_config):
    col = Collector(num, host_config)
    sys.exit(col.run())


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
        '--platform',
        type=int,
        default=0,
        help='platform number of the AMII - selects port range to use (default: 0)'
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

    procs = []
    failed_worker = False
    worker_cfg = ZmqConfig(
        args.platform,
        binds={},
        connects={zmq.PUSH: ('localhost', ZmqPorts.Collector), zmq.SUB: (args.host, ZmqPorts.Graph)}
    )
    collector_cfg = ZmqConfig(
        args.platform,
        binds={zmq.PULL: ('*', ZmqPorts.Collector)},
        connects={zmq.PUSH: (args.host, ZmqPorts.FinalCollector)}
    )

    try:
        src_url_match = re.match('(?P<prot>.*)://(?P<body>.*)', args.source)
        if src_url_match:
            src_cfg = src_url_match.groups()
        else:
            print("Invalid data source config string:", args.source)
            return 1
            

        for i in range(args.num_workers):
            proc = mp.Process(name='worker%03d-n%03d'%(i, args.node_num), target=run_worker, args=(i, worker_cfg, src_cfg))
            proc.daemon = True
            proc.start()
            procs.append(proc)

        collector_proc = mp.Process(name='collector-n%03d'%args.node_num, target=run_collector, args=(args.num_workers, collector_cfg))
        collector_proc.daemon = True
        collector_proc.start()
        procs.append(collector_proc)

        for proc in procs:
            proc.join()
            if proc.exitcode == 0:
                print('%s exited successfully'%proc.name)
            else:
                failed_worker = True
                print('%s exited with non-zero status code: %d'%(proc.name, proc.exitcode))

        # return a non-zero status code if any workerss died
        if failed_worker:
            return 1

        return 0
    except KeyboardInterrupt:
        print("Worker killed by user...")
        return 0


if __name__ == '__main__':
    sys.exit(main())
