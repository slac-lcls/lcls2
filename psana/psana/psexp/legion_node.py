from .tools import mode

legion = None
if mode == 'mpi':
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

from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.eventbuilder_manager import EventBuilderManager
from psana.psexp.event_manager import EventManager
import numpy as np
import os

def run_smd0(run):
    smdr_man = SmdReaderManager(run.smd_dm.fds, run.max_events)
    for chunk in smdr_man.chunks():
        run_smd_task(chunk, run)

@task
def run_smd_task(view, run):
    eb_man = EventBuilderManager(run.smd_configs, run.batch_size, run.filter_callback)
    for batch in eb_man.batches(view):
        run_bigdata_task(batch, run)

@task
def run_bigdata_task(batch, run):
    evt_man = EventManager(run.smd_configs, run.dm, run.filter_callback)
    for event in evt_man.events(batch):
        run.event_fn(event, run.det)

run_to_process = []
def analyze(run, event_fn=None, start_run_fn=None, det=None):
    run.event_fn = event_fn
    run.start_run_fn = start_run_fn
    run.det = det
    if legion.is_script:
        run_smd0(run)
    else:
        run_to_process.append(run)

if legion is not None and not legion.is_script:
    @task(top_level=True)
    def legion_main():
        for run in run_to_process:
            run_smd0(run)
