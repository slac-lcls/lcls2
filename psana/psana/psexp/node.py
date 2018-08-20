import os
import numpy as np
from psana.smdreader import SmdReader
from psana.eventbuilder import EventBuilder
from psana.event import Event

mode = os.environ.get('PS_PARALLEL', 'mpi')
MPI = None
legion = None
if mode == 'mpi':
    from mpi4py import MPI
    def task(fn): return fn # Nop when not using Legion
elif mode == 'legion':
    import legion
    from legion import task
else:
    raise Exception('Unrecognized value of PS_PARALLEL %s' % mode)

class Node(object):
    def __init__(self, mpi=None):
        self.mpi = mpi

class Smd0(Node):
    """ Sends blocks of smds to smd_node
    Identifies limit timestamp of the slowest detector then
    sends all smds within that timestamp to an smd_node.
    """
    def __init__(self, mpi, fds, n_smd_nodes, max_events=0):
        Node.__init__(self, mpi)
        self.fds = fds
        assert len(self.fds) > 0
        self.smdr = SmdReader(fds)

        self.n_smd_nodes = n_smd_nodes

        self.n_events = int(os.environ.get('PS_SMD_N_EVENTS', 100))
        self.max_events = max_events
        if self.max_events:
            if self.max_events < self.n_events:
                self.n_events = self.max_events

        if mode == 'mpi':
            self.run_mpi()
        elif mode == 'legion':
            self.run_legion()

    def run_mpi(self):
        rank = self.mpi.rank
        comm = self.mpi.comm

        rankreq = np.empty(1, dtype='i')

        for chunk in self.chunks():
            comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            comm.Send(chunk, dest=rankreq[0], tag=12)

        for i in range(self.n_smd_nodes):
            comm.Recv(rankreq, source=MPI.ANY_SOURCE)
            comm.Send(bytearray(b'eof'), dest=rankreq[0], tag=12)

    def run_legion(self, configs, batch_size=1, filter=0):
        for chunk in self.chunks():
            SmdNode.run_legion_task(chunk, configs, batch_size, filter)

    def chunks(self):
        processed_events = 0
        got_events = -1
        while got_events != 0:
            self.smdr.get(self.n_events)
            got_events = self.smdr.got_events
            processed_events += got_events
            views = bytearray()
            for i in range(len(self.fds)):
                view = self.smdr.view(i)
                if view != 0:
                    views.extend(view)
                    if i < len(self.fds) - 1:
                        views.extend(b'endofstream')
            if views:
                yield views

            if self.max_events:
                if processed_events == self.max_events:
                    break

class SmdNode(Node):
    """Handles both smd_0 and bd_nodes
    Receives blocks of smds from smd_0 then assembles
    offsets and dgramsizes into a numpy array. Sends
    this np array to bd_nodes that are registered to it."""
    def __init__(self, mpi, configs, n_bd_nodes, batch_size=1, filter=0):
        Node.__init__(self, mpi)
        self.configs = configs
        self.n_bd_nodes = n_bd_nodes
        self.batch_size = batch_size
        self.filter = filter

    def run_mpi(self):
        rank = self.mpi.rank
        comm = self.mpi.comm
        rankreq = np.empty(1, dtype='i')
        view = 0
        while True:
            # handles requests from smd_0
            comm.Send(np.array([rank], dtype='i'), dest=0)
            info = MPI.Status()
            comm.Probe(source=0, tag=12, status=info)
            count = info.Get_elements(MPI.BYTE)
            view = bytearray(count)
            comm.Recv(view, source=0, tag=12)
            if view.startswith(b'eof'):
                break

            # send offset/size(s) to bd_node
            views = self.split_chunk(view)

            n_views = len(views)
            assert n_views == len(self.configs)

            # build batch of events
            for batch in self.batches(views):
                comm.Recv(rankreq, source=MPI.ANY_SOURCE, tag=13)
                comm.Send(batch, dest=rankreq[0])

            view = 0

        for i in range(self.n_bd_nodes):
            comm.Recv(rankreq, source=MPI.ANY_SOURCE, tag=13)
            comm.Send(bytearray(b'eof'), dest=rankreq[0])

    def run_legion(self, view):
        views = self.split_chunk(view)

        n_views = len(views)
        assert n_views == len(self.configs)

        # build batch of events
        for batch in self.batches(views):
            BigDataNode.run_legion(batch)

    @task
    def run_legion_task(chunk, configs, batch_size, filter):
        smd = SmdNode(None, configs, 1, batch_size, filter)
        smd.run_legion(chunk)

    def split_chunk(self, chunk):
        return chunk.split(b'endofstream')

    def batches(self, chunk):
        eb = EventBuilder(chunk, self.configs)
        batch = eb.build(batch_size=self.batch_size, filter=self.filter)
        while eb.nevents:
            yield batch
            batch = eb.build(batch_size=self.batch_size, filter=self.filter)

class BigDataNode(Node):
    def __init__(self, mpi, smd_configs, dm, smd_node_id):
        Node.__init__(self, mpi)
        self.smd_configs = smd_configs
        self.dm = dm
        self.smd_node_id = smd_node_id

    def run_mpi(self):
        rank = self.mpi.rank
        comm = self.mpi.comm
        while True:
            comm.Send(np.array([rank], dtype='i'), dest=self.smd_node_id, tag=13)
            info = MPI.Status()
            comm.Probe(source=self.smd_node_id, tag=MPI.ANY_TAG, status=info)
            count = info.Get_elements(MPI.BYTE)
            view = bytearray(count)
            comm.Recv(view, source=self.smd_node_id)
            if view.startswith(b'eof'):
                break

            for event in self.events(view):
                yield event

    def run_legion(self, batch):
        for event in self.events(batch):
            pass # do something

    @task
    def run_legion_task(batch, configs):
        bd = BigDataNode(None, configs, None, None) # FIXME: Need dm
        bd.run_legion(batch)

    def events(self, view):
        views = view.split(b'endofevt')
        for event_bytes in views:
            if event_bytes:
                evt = Event().from_bytes(self.smd_configs, event_bytes)
                # get big data
                ofsz = np.asarray([[d.info.offsetAlg.intOffset, d.info.offsetAlg.intDgramSize] \
                        for d in evt])
                bd_evt = self.dm.next(offsets=ofsz[:,0], sizes=ofsz[:,1], read_chunk=False)
                yield bd_evt

def analyze(ds, event_fn=None, start_run_fn=None):
    if mode != 'legion':
        for run in ds.runs():
            if start_run_fn is not None:
                start_run_fn(run)
            for event in ds.events():
                if event_fn is not None:
                    event_fn(event)
    else:
        assert False
