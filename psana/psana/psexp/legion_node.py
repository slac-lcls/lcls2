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

@task(inner=True)
def run_smd0_task(run):
    global_procs = legion.Tunable.select(legion.Tunable.GLOBAL_PYS).get()

    smdr_man = SmdReaderManager(run.smd_dm.fds, run.max_events)
    for i, chunk in enumerate(smdr_man.chunks()):
        run_smd_task(chunk, run, point=i)
    # Block before returning so that the caller can use this task's future for synchronization
    legion.execution_fence(block=True)

@task(inner=True)
def run_smd_task(view, run):
    eb_man = EventBuilderManager(run.smd_configs, batch_size=run.batch_size, filter_fn=run.filter_callback, destination=0) #FIXME: needs to talk to Elliott how to handle cube data in legion
    for i, batch_dict in enumerate(eb_man.batches(view)):
        batch, _ = batch_dict[0]
        run_bigdata_task(batch, run, point=i)

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
        num_procs = legion.Tunable.select(legion.Tunable.GLOBAL_PYS).get()

        bar = legion.c.legion_phase_barrier_create(legion._my.ctx.runtime, legion._my.ctx.context, num_procs)
        legion.c.legion_phase_barrier_arrive(legion._my.ctx.runtime, legion._my.ctx.context, bar, 1)
        global_task_registration_barrier = legion.c.legion_phase_barrier_advance(legion._my.ctx.runtime, legion._my.ctx.context, bar)
        legion.c.legion_phase_barrier_wait(legion._my.ctx.runtime, legion._my.ctx.context, bar)

        return run_smd0_task(run)
    else:
        run_to_process.append(run)

if legion is not None and not legion.is_script:
    @task(top_level=True)
    def legion_main():
        for run in run_to_process:
            run_smd0_task(run, point=0)
