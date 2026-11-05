"""MPI `run.steps()` on GPU BD ranks: the loop GPU-140-03 made yield nothing.

Four ranks: 1 smd0 + 1 EB + 2 BD, one shared GPU. Before #153 PR B,
RunParallel.steps() returned immediately on a GPU BD rank, so this loop ran
zero times and reported no error. Two BD ranks matter: the teardown that
run.steps() now shares with run.events() closes the batch iterator, then the
manager (which barriers on device_comm), then frees device_comm, and a
single-rank run cannot observe a break in that order.

Run with PS_PARALLEL=mpi, PS_EB_NODES=1, PS_SRV_NODES=0.

Dataset is selectable, because the two available ones test different things:

  PS_STEPS_DATASET=51 (default) -- mfx100848724 run 51, the dataset every
      other GPU MPI driver uses. Proves the mechanism: steps() yields, the
      shared teardown releases cleanly at 2 BD ranks.
  PS_STEPS_DATASET=77 -- mfx100852324 run 77, a 3-step Jungfrau pedestal scan
      with 1000 events per step (verified: job 42149659). The real
      multi-step case. Its Jungfrau Configure records used to fail the GPU
      config parser -- "field jungfrau.config.user.gainMode:gainModeEnum
      segment 19 is owned by more than one routed Configure Names record"
      -- because the gain-mode scan declares segments 11-31 twice under
      "config" in stream 3. detector_field_handles() now omits config fields
      outright (gpudgram/config.py), since its only caller discarded them a
      line later; run 77 opens in serial as of job 42996269.

Raw structure only -- runs 77-79 have nonzero pixel_status on every g1/g2
pixel, so a calibrated image can come out entirely masked. This asserts step
structure and event accounting, not calibrated values.
"""
import os
import sys


def unused(batch, stream):
    # No publications are read here; this driver is about step iteration.
    pass


def main():
    from mpi4py import MPI
    from psana import DataSource
    from psana.gpu import GpuTask
    comm = MPI.COMM_WORLD
    assert comm.size == 4, f'expected 4 ranks, got {comm.size}'

    task = GpuTask(unused, ['jungfrau.raw'], [])
    which = os.environ.get('PS_STEPS_DATASET', '51')
    # max_events has to clear the steps being counted, or the assertions below
    # test the cap instead of the stream. Run 77 is 1000 events per step
    # (job 42149659), so 120 stops inside step 0 and no number of BD ranks can
    # ever see 3 steps. 2400 reaches step 2 with room for the split between
    # BDs while still being well short of the 3000-event run.
    #
    # PS_STEPS_NO_CAP=1 drops the cap entirely, which is the only way this
    # driver reaches EndRun -- and EndRun is where delivery order was wrong:
    # the manager released a packet's transitions before its events whenever
    # the execution pool was idle, which is the final packet every time
    # (EndRun drains the pool, and the EB appends EndRun to the batch holding
    # the run's last events). Both Run.events() and _steps_impl return on
    # EndRun, so those events were lost. A capped run stops before EndRun and
    # cannot see it.
    no_cap = os.environ.get('PS_STEPS_NO_CAP') == '1'
    options = (dict(exp='mfx100852324', run=77, detectors=['jungfrau', 'timing'],
                    max_events=0 if no_cap else 2400)
               if which == '77' else
               dict(exp='mfx100848724', run=51, detectors=['jungfrau'],
                    max_events=0 if no_cap else 120))
    ds = DataSource(dir='/sdf/data/lcls/ds/prj/public01/xtc',
                    gpu_det='jungfrau', gpu_fn=task,
                    batch_size=20, **options)
    print(f'rank {comm.rank}: DataSource built', flush=True)
    run = next(ds.runs())
    role = run.comms.node_type()
    print(f'rank {comm.rank}: role={role}, run opened', flush=True)

    # EVERY rank must enter run.steps(), not just the BD ranks. smd0 and the
    # EB yield nothing from it, but only after RunParallel.start() has run
    # their blocking service loops -- smd0.start() reads chunks and feeds the
    # EB, eb_node.start() builds batches and ends by sending each BD the empty
    # message that terminates it. Gating this loop on role=='bd' skips those
    # loops entirely, so the BDs park in bd_comm.Probe(source=0) waiting for a
    # batch nobody will build while smd0 and the EB sit in the gather below:
    # a circular wait with no timeout, which hangs until the job's time limit
    # (observed: job 42992578, 40+ min with no output, both BDs in PMPI_Probe
    # and smd0/EB already in PMPI_Gather/PMPI_Barrier). Every other GPU MPI
    # driver iterates on all roles for this reason -- mpi_input_only.py:27,
    # mpi_gpu_publication.py:34, mpi_calib_pixel_exact.py:138.
    per_step = []
    for step in run.steps():
        # Reached on BD ranks only. Before PR B this body never executed.
        n = sum(1 for _ in step.events())
        per_step.append(n)
        print(f'rank {comm.rank}: step {len(per_step) - 1} -> {n} events',
              flush=True)
    if role != 'bd':
        # The non-BD roles drove the stream without ever touching the device.
        assert not per_step, (role, per_step)
        assert 'cupy' not in sys.modules, role

    rows = comm.gather({'role': role, 'per_step': per_step}, root=0)
    comm.Barrier()
    run.close_shared_memory()

    if comm.rank != 0:
        return 0

    failures = []
    bd_rows = [r for r in rows if r['role'] == 'bd']
    if len(bd_rows) != 2:
        failures.append(f'expected 2 BD ranks, got {len(bd_rows)}')

    for i, row in enumerate(bd_rows):
        # The headline: steps() yielded at all. This is GPU-140-03.
        if not row['per_step']:
            failures.append(f'BD rank {i} saw no steps: run.steps() yielded '
                            f'nothing, which is the defect PR B fixes')

    counts = [len(row['per_step']) for row in bd_rows]
    if len(set(counts)) != 1:
        # Every BD receives every BeginStep -- smd0 and the EB prepend missing
        # history to each recipient -- so the step COUNT must agree even
        # though the events are split between them.
        failures.append(f'BD ranks disagree on step count: {counts}')
    if which == '77' and counts and counts[0] != 3:
        failures.append(f'run 77 has 3 steps, saw {counts[0]}')

    # Events are partitioned, not duplicated: each L1 goes to exactly one BD.
    max_events = options['max_events']
    total = sum(sum(row['per_step']) for row in bd_rows)
    if max_events and total > max_events:
        failures.append(f'delivered {total} events, more than '
                        f'max_events={max_events}')
    if total == 0:
        failures.append('no events delivered inside any step')

    # Uncapped, the run runs to EndRun and every event must arrive. This is
    # the assertion that catches a transition released ahead of the events it
    # follows: EndRun delivered early makes Run.events()/_steps_impl return
    # before the final packet's events, so the total comes up short.
    RUN_SIZES = {'77': 3000}
    if no_cap and which in RUN_SIZES and total != RUN_SIZES[which]:
        failures.append(
            f'uncapped run {which} delivered {total} events, expected '
            f'{RUN_SIZES[which]}: events after the last BeginStep are lost '
            f'if EndRun is delivered before them')

    for row in bd_rows:
        # Uncapped, no step may be empty -- including the last, which a cap
        # would otherwise truncate.
        counted = row['per_step'] if no_cap else row['per_step'][:-1]
        if any(n == 0 for n in counted):
            failures.append(f'a step delivered no events: {row["per_step"]}')

    print('MPI_STEPS_RESULT', {'counts': counts, 'total': total,
                               'per_step': [r['per_step'] for r in bd_rows]},
          flush=True)
    if failures:
        for failure in failures:
            print('FAIL', failure, flush=True)
        return 1
    print('MPI_STEPS_PASS', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
