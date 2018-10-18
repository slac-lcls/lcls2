from .tools import mode
MPI = None
legion = None
if mode == 'mpi':
    from mpi4py import MPI
    # Nop when not using Legion
    def task(fn=None, **kwargs):
        if fn is None:
            return lambda fn: fn
        return fn
elif mode == 'legion':
    import legion
    from legion import task
else:
    raise Exception('Unrecognized value of PS_PARALLEL %s' % mode)

from psana.smdreader import SmdReader
from psana.eventbuilder import EventBuilder
from psana.event import Event
import numpy as np
import os

def run_smd0(run):
    fds = run.smd_dm.fds
    max_events = run.max_events
    n_events = int(os.environ.get('PS_SMD_N_EVENTS', 1000))
    if max_events:
        if max_events < n_events:
            n_events = max_events

    smdr = SmdReader(fds)
    got_events = -1
    processed_events = 0
    while got_events != 0:
        smdr.get(n_events)
        got_events = smdr.got_events
        processed_events += got_events
        views = bytearray()
        for i in range(len(fds)):
            view = smdr.view(i)
            if view != 0:
                views.extend(view)
                if i < len(fds) - 1:
                    views.extend(b'endofstream')

        if views:
            run_smd_task(views, run)

        if max_events:
            if processed_events >= max_events:
                break

@task
def run_smd_task(views, run): 
    run_smd(views, run)

def run_smd(view, run):
    views = view.split(b'endofstream')
    eb = EventBuilder(views, run.smd_configs)
    batch = eb.build(batch_size=run.batch_size, filter=run.filter_callback)
    while eb.nevents:
        run_bigdata_task(batch, run)
        batch = eb.build(batch_size=run.batch_size, filter=run.filter_callback)

@task
def run_bigdata_task(batch, run):
    run_bigdata(batch, run)

def run_bigdata(batch, run):
    evt_byte_arr = batch.split(b'endofevt')
    for i, evt_byte in enumerate(evt_byte_arr):
        if evt_byte:
            evt = Event().from_bytes(run.smd_configs, evt_byte)
            # get big data
            ofsz = np.asarray([[d.info.offsetAlg.intOffset, d.info.offsetAlg.intDgramSize] \
                    for d in evt])
            bd_evt = run.dm.jump(ofsz[:,0], ofsz[:,1])
            run.event_fn(bd_evt, run.det)

run_to_process = []
def analyze(run, event_fn=None, start_run_fn=None, det=None):
    run.event_fn = event_fn
    run.start_run_fn = start_run_fn
    run.det = det
    run_to_process.append(run)

@task(top_level=True)
def legion_main():
    for run in run_to_process:
        run_smd0(run)
