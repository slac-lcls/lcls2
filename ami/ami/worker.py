import re
import os
import sys
import zmq
import time
import json
import shutil
import tempfile
import argparse
import threading
import numpy as np
import multiprocessing as mp
from ami.graph import Graph, GraphConfigError, GraphRuntimeError
from ami.comm import Ports, Collector, ResultStore
from ami.data import MsgTypes, DataTypes, Transitions, Occurrences, Message, Datagram, Transition, StaticSource
import numpy as np

class Request(object):
    def __init__(self, store, name, number):
        self.name = name
        self.number = number
        self.progress = 0
        self.store = store
        self.sum = None
    def update(self):
        data = self.store.get(self.name)
        if data is None: return False
        if self.sum is None:
            self.sum = data
        else:
            self.sum += data
        self.progress +=1
        if self.progress==self.number:
            self.store.put(self.name+'_avg_'+str(self.number),self.sum)
            return True
        return False

def simple_tree(json):
    """
    Generate a dependency tree from a json input

    Parameters
    ----------
    json : json object

    Returns
    -------
    graph : dict
        dict where the keys are operations, the value are upstream
        (dependent) operations that should be done first
    """
    graph = {}
    for op in json:
        graph[op] = json[op]['inputs']
    return graph


def resolve_dependencies(dependencies):
    """
    Dependency resolver

    Parameters
    ----------
    dependencies : dict
        the values are the dependencies of their respective keys

    Returns
    -------
    order : list
        a list of the order in which to resolve dependencies
    """

    # this just makes sure there are no duplicate dependencies
    d = dict( (k, set(dependencies[k])) for k in dependencies )

    # output list
    r = []

    while d:

        # find all nodes without dependencies (roots)
        t = set(i for v in d.values() for i in v) - set(d.keys())

        # and items with no dependencies (disjoint nodes)
        t.update(k for k, v in d.items() if not v)

        # both of these can be resolved
        r.extend(t)

        # remove resolved depedencies from the tree
        d = dict(((k, v-t) for k, v in d.items() if v))

    return r
   

class Worker(object):
    def __init__(self, idnum, src, collector_addr, graph_addr):

        """
        idnum : int
            a unique integer identifying this worker
        src : object
            object with an events() method that is an iterable (like psana.DataSource)
        """

        self.idnum = idnum
        self.src = src
        self.ctx = zmq.Context()
        self.store = ResultStore(collector_addr, self.ctx)

        # >>> NON PYTHONBACKEND
        #self.graph = Graph(self.store)

        self.graph = {}
        self.graph_comm = self.ctx.socket(zmq.SUB)
        #self.graph_comm.setsockopt_string(zmq.SUBSCRIBE, "graph")
        self.graph_comm.setsockopt_string(zmq.SUBSCRIBE, "")
        self.graph_comm.connect(graph_addr)
        self.requests = []

    def run(self):

        for msg in self.src.events():

            # check to see if the graph has been reconfigured after update
            if msg.mtype == MsgTypes.Occurrence and msg.payload == Occurrences.Heartbeat:
                new_graph = None
                while True:
                    try:
                        topic = self.graph_comm.recv_string(flags=zmq.NOBLOCK)
                        payload = self.graph_comm.recv_pyobj()
                        if topic == "graph":
                            new_graph = payload
                        if topic == "request":
                            self.requests.append(Request(self.store,*payload))
                        else:
                            print("worker%d: No handler for received topic: %s"%(self.idnum, topic))
                    except zmq.Again:
                        break
                if new_graph is not None:

                    # >>> NON PYTHONBACKEND
                    #self.graph.update(new_graph)

                    self.graph = new_graph
                    print("worker%d: Received new configuration"%self.idnum)


                    # >>> NON PYTHONBACKEND
                    # >>> TODO figure out how to check graph
                    #try:
                    #    self.graph.configure()
                    #    print("worker%d: Configuration complete"%self.idnum)
                    #except GraphConfigError as graph_err:
                    #    print("worker%d: Configuration failed reverting to previous config:"%self.idnum, graph_err)
                    #    # if this fails we just die
                    #    self.graph.revert()

                    self.new_graph_available = False
                self.store.send(msg)
            elif msg.mtype == MsgTypes.Datagram:
                for dgram in msg.payload:
                    self.store.put_dgram(dgram)

                # loop over operations in the correct order
                for op in resolve_dependencies(simple_tree(self.graph)):

                    # this takes care of "base" data (e.g. given by the DAQ)
                    if op in self.store.namespace:
                        continue

                    # at the end of the executed code, inject outputs in to the feature store
                    store_put_list = ['store.put("%s", %s)'%(output, output) for output in self.graph[op]['outputs']]
                    store_put = '\n' + '; '.join(store_put_list)

                    # generate the local & global namespaces for the execution of the graph operation
                    loc = {'store': self.store}
                    loc.update(self.store.namespace)
                    glb = {"np" : np} # TODO be smarter :)

                    # execute the operation code (graph is the json)
                    exec(self.graph[op]['code'] + store_put, glb, loc)

                #try:
                #    self.graph.execute(updates)
                #except GraphRuntimeError as graph_err:
                #    print("worker%s: Failure encountered executing graph:"%self.idnum, graph_err)
                #    return 1

                self.store.collect()
                for r in self.requests:
                    if (r.update()):
                        self.requests.remove(r)
                        print(self.store.get('cspad_avg_3').shape)
            else:
                self.store.send(msg)


class NodeCollector(Collector):
    def __init__(self, node, num_workers, collector_addr, upstream_addr):
        super(__class__, self).__init__(collector_addr)
        self.node = node
        self.counts = { MsgTypes.Transition: 0, MsgTypes.Occurrence: 0 }
        self.num_workers = num_workers
        self.upstream = ResultStore(upstream_addr, self.ctx)

    def process_msg(self, msg):
        if msg.mtype == MsgTypes.Transition:
            self.counts[MsgTypes.Transition] += 1
            if self.counts[MsgTypes.Transition] == self.num_workers:
                if msg.payload.ttype == Transitions.Allocate:
                    for name, dtype in msg.payload.payload:
                        self.upstream.create(name, dtype)
                self.upstream.send(msg)
                self.counts[MsgTypes.Transition] = 0
        elif msg.mtype == MsgTypes.Occurrence:
            self.counts[MsgTypes.Occurrence] += 1
            if self.counts[MsgTypes.Occurrence] == self.num_workers:
                if msg.payload == Occurrences.Heartbeat:
                    self.upstream.collect()
                self.upstream.send(msg)
                self.counts[MsgTypes.Occurrence] = 0
        elif msg.mtype == MsgTypes.Datagram:
            self.upstream.put_dgram(msg.payload)


def run_worker(num, source, collector_addr, graph_addr):

    print('Starting worker # %d, sending to collector at %s' % (num, collector_addr))

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
        return 1
    worker = Worker(num, src, collector_addr, graph_addr)
    return worker.run()


def run_collector(node_num, num_workers, collector_addr, upstream_addr):
    print('Starting collector on node # %d' % node_num)
    collector = NodeCollector(node_num, num_workers, collector_addr, upstream_addr)
    return collector.run()


def main():
    parser = argparse.ArgumentParser(description='AMII Worker/Collector App')

    parser.add_argument(
        '-H',
        '--host',
        default='localhost',
        help='hostname of the AMII Manager (default: localhost)'
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

    parser.add_argument(
        '-n',
        '--num-workers',
        type=int,
        default=1,
        help='number of worker processes (default: 1)'
    )

    parser.add_argument(
        '-N',
        '--node-num',
        type=int,
        default=0,
        help='node identification number (default: 0)'
    )

    parser.add_argument(
        'source',
        metavar='SOURCE',
        help='data source configuration (exampes: static://test.json, psana://exp=xcsdaq13:run=14)'
    )

    args = parser.parse_args()
    ipcdir = tempfile.mkdtemp()
    collector_addr = "ipc://%s/node_collector"%ipcdir
    upstream_addr = "tcp://%s:%d"%(args.host, args.collector)
    graph_addr = "tcp://%s:%d"%(args.host, args.graph)
    comm_addr = "tcp://%s:%d"%(args.host, args.port)

    procs = []

    try:
        src_url_match = re.match('(?P<prot>.*)://(?P<body>.*)', args.source)
        if src_url_match:
            src_cfg = src_url_match.groups()
        else:
            print("Invalid data source config string:", args.source)
            return 1

        for i in range(args.num_workers):
            proc = mp.Process(
                name='worker%03d-n%03d'%(i, args.node_num),
                target=run_worker,
                args=(i, src_cfg, collector_addr, graph_addr)
            )
            proc.daemon = True
            proc.start()
            procs.append(proc)

        collector_proc = mp.Process(
            name='manager-n%03d'%args.node_num,
            target=run_collector,
            args=(args.node_num, args.num_workers, collector_addr, upstream_addr)
        )
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
    finally:
        if ipcdir is not None and os.path.exists(ipcdir):
            shutil.rmtree(ipcdir)


if __name__ == '__main__':
    sys.exit(main())

 
