from psana.psexp import mode

pygion = None
if mode == 'legion':
    import pygion
    from pygion import task
else:
    # Nop when not using Legion
    def task(fn=None, **kwargs):
        if fn is None:
            return lambda fn: fn
        return fn

from psana.psexp import EventBuilderManager, TransitionId, Events

def smd_chunks(run):
    for smd_chunk, update_chunk in run.smdr_man.chunks():
        yield smd_chunk

@task(inner=True)
def run_smd0_task(run):
    global_procs = pygion.Tunable.select(pygion.Tunable.GLOBAL_PYS).get()

    for i, smd_chunk in enumerate(smd_chunks(run)):
        run_smd_task(smd_chunk, run, point=i)
    # Block before returning so that the caller can use this task's future for synchronization
    pygion.execution_fence(block=True)

def smd_batches(smd_chunk, run):
    eb_man = EventBuilderManager(smd_chunk, run.configs, run.dsparms) 
    for smd_batch_dict, step_batch_dict in eb_man.batches():
        smd_batch, _ = smd_batch_dict[0]
        yield smd_batch

@task(inner=True)
def run_smd_task(smd_chunk, run):
    for i, smd_batch in enumerate(smd_batches(smd_chunk, run)):
        run_bigdata_task(smd_batch, run, point=i)

def batch_events(smd_batch, run):
    batch_iter = iter([smd_batch, bytearray()])
    def get_smd():
        for this_batch in batch_iter:
            return this_batch

    events = Events(run.configs, run.dm, run.dsparms, 
            filter_callback=run.dsparms.filter, get_smd=get_smd)
    for evt in events:
        if evt.service() != TransitionId.L1Accept: continue
        yield evt

@task
def run_bigdata_task(batch, run):
    for evt in batch_events(batch, run):
        run.event_fn(evt, run.det)

run_to_process = []
def analyze(run, event_fn=None, det=None):
    run.event_fn = event_fn
    run.det = det
    if pygion.is_script:
        num_procs = pygion.Tunable.select(pygion.Tunable.GLOBAL_PYS).get()

        bar = pygion.c.legion_phase_barrier_create(pygion._my.ctx.runtime, pygion._my.ctx.context, num_procs)
        pygion.c.legion_phase_barrier_arrive(pygion._my.ctx.runtime, pygion._my.ctx.context, bar, 1)
        global_task_registration_barrier = pygion.c.legion_phase_barrier_advance(pygion._my.ctx.runtime, pygion._my.ctx.context, bar)
        pygion.c.legion_phase_barrier_wait(pygion._my.ctx.runtime, pygion._my.ctx.context, bar)

        return run_smd0_task(run)
    else:
        run_to_process.append(run)
    

if pygion is not None and not pygion.is_script:
    @task(top_level=True)
    def legion_main():
        for run in run_to_process:
            run_smd0_task(run, point=0)
