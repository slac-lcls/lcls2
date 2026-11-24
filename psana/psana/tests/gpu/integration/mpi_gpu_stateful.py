"""A stateful GpuTask under MPI: per-BD workers, summed across ranks.

Every stateful check so far has been single-process -- the device tests, and
the run 77 drivers in validation/gpu-stateful-20261118/. So the stateful
finalization path had never run with real device-group collectives in
cleanup, which is where R4 lives: a finalizer error must reach the fatal
guard before shared-constant cleanup barriers over that group.

Four ranks: smd0 + EB + 2 BDs, one shared GPU. Each BD builds its own
TaskWorkers (one instance per execution stream) and its own reducer, and sees
only its share of the events. So the per-step counts are summed across BDs
before being compared with run 77's structure: three steps of 1000.

Run with PS_PARALLEL=mpi, PS_EB_NODES=1, PS_SRV_NODES=0.

Uncapped on purpose. A capped run stops before EndRun, and EndRun is what
makes run_status 'complete'; it is also the only way finalization runs after
a full stream. The capped case is covered single-process by
validation/gpu-stateful-20261118/stateful_run77_capped.py.
"""
import os
import sys
from functools import partial

PER_STEP = [1000, 1000, 1000]      # run 77 (job 42149659)
FIRST_STEP = 1
N_BINS = 8


def main():
    from mpi4py import MPI
    from psana import DataSource
    from psana.gpu import GpuTask, GpuTaskWorker
    comm = MPI.COMM_WORLD
    assert comm.size == 4, f'expected 4 ranks, got {comm.size}'

    results, sink = {}, []

    class StepCounter(GpuTaskWorker):
        def __init__(self, store):
            self.store = store
            self.counts = None
            self.hooks = []

        def begin_run(self, ctx, stream):
            import cupy as cp
            self.counts = ctx.state.zeros((N_BINS,), cp.uint64)

        def begin_step(self, ctx, stream):
            self.hooks.append(('begin_step', ctx.step_scope.index))

        def process(self, batch, stream):
            self.counts[batch.step_scope.index] += batch.size

        def end_step(self, ctx, stream):
            k = ctx.step_scope.index
            self.hooks.append(('end_step', k))
            ctx.emit('cube', key='events', count=self.counts[k:k + 1])

        def end_run(self, ctx, stream):
            import cupy as cp
            self.store[ctx.stream_index] = (ctx.run_status,
                                            cp.asnumpy(self.counts).tolist(),
                                            self.hooks)

        def close(self):
            self.counts = None

    task = GpuTask(factory=partial(StepCounter, results),
                   inputs=['jungfrau.raw'], state_bytes_per_stream=1 << 20,
                   aggregate_outputs={'cube': {'count': 'sum'}},
                   aggregate_sink=sink.append)
    ds = DataSource(exp='mfx100852324', run=77,
                    dir='/sdf/data/lcls/ds/prj/public01/xtc',
                    detectors=['jungfrau', 'timing'], gpu_det='jungfrau',
                    gpu_fn=task, batch_size=20, n_gpu_streams=2, max_events=0)
    print(f'rank {comm.rank}: DataSource built', flush=True)
    run = next(ds.runs())
    role = run.comms.node_type()
    print(f'rank {comm.rank}: role={role}, run opened', flush=True)

    # EVERY rank iterates, not just the BDs. smd0 and the EB yield nothing,
    # but only after their blocking service loops have run; gating this on
    # role == 'bd' parks the BDs in Probe() waiting for batches nobody builds
    # while smd0 and the EB wait in the gather below -- a circular wait with
    # no timeout. See mpi_gpu_steps.py for the observed hang.
    n_events = sum(1 for _ in run.events())
    print(f'rank {comm.rank}: {n_events} events', flush=True)
    if role != 'bd':
        assert n_events == 0, (role, n_events)
        assert 'cupy' not in sys.modules, role

    bins = [sum(results[i][1][b] for i in results) for b in range(N_BINS)] \
        if results else [0] * N_BINS
    steps = [r for r in sink if r.begin_ts is not None]
    row = {'role': role, 'n_events': n_events, 'bins': bins,
           'statuses': [results[i][0] for i in sorted(results)],
           'instances': len(results),
           'step_records': [(r.fields['count'].tolist()[0], r.complete, r.truncated)
                            for r in steps]}
    rows = comm.gather(row, root=0)
    comm.Barrier()
    run.close_shared_memory()

    if comm.rank != 0:
        return 0

    failures = []
    bd = [r for r in rows if r['role'] == 'bd']
    if len(bd) != 2:
        failures.append(f'expected 2 BD ranks, got {len(bd)}')

    for i, r in enumerate(bd):
        if r['instances'] != 2:
            failures.append(f'BD {i}: {r["instances"]} worker instances, expected 2')
        if any(s != 'complete' for s in r['statuses']):
            failures.append(f'BD {i}: run_status {r["statuses"]}, expected complete')
        if not r['step_records']:
            failures.append(f'BD {i}: no step records reached its sink')
        if any(t for _, _, t in r['step_records']):
            failures.append(f'BD {i}: a step was truncated in an uncapped run')

    total_events = sum(r['n_events'] for r in bd)
    summed = [sum(r['bins'][b] for r in bd) for b in range(N_BINS)]
    per_step = summed[FIRST_STEP:FIRST_STEP + 3]
    rest = summed[0] + sum(summed[FIRST_STEP + 3:])
    # Each BD's reducer is its own, so the sink totals add across BDs too.
    sink_total = sum(c for r in bd for c, _, _ in r['step_records'])

    print(f'MPI_STATEFUL events={total_events} per_step={per_step} '
          f'pre_or_beyond={rest} sink_total={sink_total}', flush=True)
    for i, r in enumerate(bd):
        print(f'MPI_STATEFUL bd{i} events={r["n_events"]} '
              f'bins={r["bins"][:5]} records={r["step_records"]}', flush=True)

    if per_step != PER_STEP:
        failures.append(f'per_step {per_step}, expected {PER_STEP}')
    if total_events != sum(PER_STEP):
        failures.append(f'{total_events} events, expected {sum(PER_STEP)}')
    if rest:
        failures.append(f'{rest} events outside the three steps')
    if sink_total != sum(PER_STEP):
        failures.append(f'sink totals {sink_total}, expected {sum(PER_STEP)}')

    for f in failures:
        print(f'MPI_STATEFUL_FAIL: {f}', flush=True)
    print('MPI_STATEFUL_PASS' if not failures else 'MPI_STATEFUL_FAIL', flush=True)
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
