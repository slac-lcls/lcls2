# Stateful GPU task on real data (PR #270 phase 1, PR #344)

Results below are from job 45895215 on an A100, at commit `530a3ee5a` (the
scope-0 change, the `final`/`truncated` record flags and the budgeted
`StepCube` scratch bank all included). Earlier job numbers are kept where a
row records something one of those runs settled.

`stateful_run77.py` runs mfx100852324 run 77 uncapped through
`GpuTask(factory=...)` with two execution streams. Each worker instance counts
the events of every subbatch it processes into `bins[step_scope.index]`, held
in `ctx.state` inside the setup reservation, and reports the bins from
`end_run`. Summed over both instances the three steps must each hold 1000
events and the total must equal what `run.events()` delivered.

    PS_PARALLEL=none python3 validation/gpu-stateful-20261118/stateful_run77.py

| | result (A100, job 45895215) |
| --- | --- |
| events delivered | 3000 |
| instances, status | 2, both `complete` |
| per-instance bins (scopes 0..4) | `[0, 500, 500, 500, 0]` each |
| per step (scopes 1..3) | `[1000, 1000, 1000]` |
| pre-step or beyond | 0 |
| step hooks, per stream | `begin_step`/`end_step` for scopes 1, 2, 3, once each, in order; none for scope 0, which had no events |
| count read inside `end_step` (job 45734339) | 500, 500, 500 -- equal to each stream's final count for that scope |
| `ctx.emit` -> sink, per scope | one record each, `complete` (so also `final`, `seq 0`, not `truncated`), merged over both streams: `[1000]`, `[1000]`, `[1000]` |
| `ctx.emit` in `end_run` -> sink | one record, `[0, 1000, 1000, 1000]`, complete, streams (0, 1) |

The `end_step` rows are the "Hook placement" contract measured: by the time a
stream's `end_step` runs, every `process` for that scope on that stream has
completed, with no drain and no barrier -- the hook is simply enqueued after
them on the same stream. The sink rows add the export path: `ctx.emit` copies
on that same stream and waits for its own copy only, the per-BD reducer merges
the two streams and learns completion from the framework (a stream's
`end_step` returning), and the sink is called inline once per complete scope.
Scope 0 -- before the first BeginStep -- gets a hook pair, and so a record,
only if some stream processed events in it; run 77 has none, so the sink sees
exactly one record per step. (Earlier runs in this series exported a zero
scope-0 record; that was changed on review.) A step the run ended inside, and
the run record of a run cut short, are delivered `truncated` and so not
`complete`; a key the host byte cap delivered early arrives as several
records, the last `final`. Neither happened here.

Two things this run settled:

- `StepScope.index` is 0 *before* the first BeginStep and counts BeginSteps
  from there, so run 77's steps are scopes 1, 2 and 3. Scope 0 holds pre-step
  events, of which run 77 has none. A step-scan worker should key its bins by
  scope, not assume the first step is 0.
- `run_status` was "incomplete" on the first attempt although EndRun was
  reached: the flag was assigned after the yield that delivers EndRun, and
  `Run.events()` returns on that envelope, so the assignment never ran. It is
  now set before the drain; `test_run_completeness_is_recorded_before_endrun_is_yielded`
  pins it.

The pool-level counterpart, with real CUDA streams and a synthetic input, is
`psana/psana/tests/gpu/integration/test_gpu_task_worker_device.py` (same job).

## StepCube against a CPU reference (step D)

`cube_run77.py` runs `psana.gpu.examples.step_cube.StepCube` -- per-pixel
uint64 sums and per-segment presence counts of `jungfrau.raw`, one bank per
stream plus a same-sized scratch bank the per-batch reduction lands in (so no
temporary is allocated outside the budget), zeroed at `begin_step`, emitted
at `end_step` -- then re-reads run 77
on the plain CPU path with `run.steps()` and sums each present segment's raw
words per step. Run 77's jungfrau has 32 segments, each `(1, 512, 1024)`.

    PS_PARALLEL=none python3 validation/gpu-stateful-20261118/cube_run77.py

| | result (A100, job 45895215; first measured in 45757000) |
| --- | --- |
| GPU records | 3, one per step, complete, streams (0, 1), key = dense segment ids 0..31 |
| per-segment counts, every step | 1000 on both sides |
| per-pixel sums, every step | `np.array_equal` True, 32 x 512 x 1024 uint64 |
| events | 3000 on both sides |

Two things the first attempt taught, both now in the code: `ctx.state`
charges each array at the CuPy pool's 512-byte rounding, so a reservation
summed from `nbytes` is short by a few hundred bytes per array --
`psana.gpu.state_bytes(...)` applies the same rounding and is what
`state_bytes_per_stream` should be computed with; and `det.raw.raw(evt)`
returns None for any partial event, so a reference that must match the GPU's
per-segment presence reads the per-segment dict instead.

## The PR #344 review fixes (R1-R7)

Job **46874797** on an A100 (`sdfampere004`), at `d3ca9d372` plus the review
fixes in the working tree. Nine stages, every exit code zero. The two runs
above are stages 3 and 4 of it, unchanged, as the regression check.

| stage | check | result |
| --- | --- | --- |
| 1 | GPU unit + service tests, on device | 1451 passed, 1 deselected |
| 2 | full GPU integration suite | 558 passed, 5 deselected |
| 3 | `stateful_run77.py`, uncapped | `STATEFUL77_PASS` |
| 4 | `cube_run77.py` vs the CPU reference | `CUBE77_PASS` |
| 5 | `stateful_run77_capped.py`, `PS_CAP=1500` | `CAPPED77_PASS` |
| 6 | same, `PS_CAP=500` | `CAPPED77_PASS` |
| 7 | same, `PS_CAP=1500 PS_STREAMS=1` | `CAPPED77_PASS` |
| 8 | `mpi_gpu_steps.py` on run 77, 4 ranks, uncapped | `MPI_STEPS_PASS` |
| 9 | `mpi_gpu_stateful.py`, 4 ranks, uncapped | `MPI_STATEFUL_PASS` |

### Capped runs: the eligible frontier (R1)

`stateful_run77_capped.py` is the capped counterpart to `stateful_run77.py`,
and the only real-data check of the late cutoff. Every stateful run before it
went to EndRun with no cap, so read-ahead past the cutoff was never
exercised: once `max_events` is reached the manager still submits the
remaining subbatches of the packet, carrying no events. Counting those as
progress reported a step that was delivered whole as truncated, and gave the
NEXT step hooks and an exported record for events nobody received.

    PS_PARALLEL=none PS_CAP=1500 python3 validation/gpu-stateful-20261118/stateful_run77_capped.py

`PS_CAP` must land strictly inside a step; the script refuses a cap on a step
boundary, because the cutoff is found on the next step's first event and the
frontier then stays on the step before it. `PS_STREAMS` sets the stream count.

| | `PS_CAP=1500`, 2 streams | `PS_CAP=500` | `PS_CAP=1500`, 1 stream |
| --- | --- | --- | --- |
| events delivered | 1500 | 500 | 1500 |
| instances, status | 2, both `incomplete` | 2, `incomplete` | 1, `incomplete` |
| step hooks | scopes 1 and 2 only, on every stream | scope 1 only | scopes 1 and 2 |
| hooks for scope 3 | none | none | none |
| step records | `[1000]` complete, `[500]` truncated | `[500]` truncated | `[1000]` complete, `[500]` truncated |
| bins (scopes 0..4) | `[0, 1000, 500, 0, 0]` | `[0, 500, 0, 0, 0]` | `[0, 1000, 500, 0, 0]` |

The scope-3 row is the point: before the fix it received hooks and a record.
The single-stream column also covers the case where no other stream exists to
service an idle one, which is how `service_idle` is reached with `exclude`
unset.

### MPI: stateful finalization with real collectives

`mpi_gpu_stateful.py` is new. Every stateful check before it was
single-process -- the device tests and both run 77 drivers -- so finalization
had never run with device-group barriers in cleanup, which is where R4 lives.
Four ranks: smd0, one EB, two BDs, one shared GPU. Each BD builds its own
worker instances and its own reducer and sees only its share of the events,
so the counts are summed across BDs.

    PS_PARALLEL=mpi PS_EB_NODES=1 PS_SRV_NODES=0 \
      mpirun -n 4 python3 psana/psana/tests/gpu/integration/mpi_gpu_stateful.py

| | result (job 46874797) |
| --- | --- |
| events, summed over BDs | 3000 |
| per BD | 1500 each, bins `[0, 500, 500, 500, 0]` |
| per step, summed | `[1000, 1000, 1000]` |
| pre-step or beyond | 0 |
| sink records per BD | three, each `[500]`, complete, not truncated |
| sink totals, summed | 3000 |
| instances, status | 2 per BD, all `complete` |

Uncapped on purpose: a capped run stops before EndRun, and EndRun is what
makes the status `complete`. A capped stateful MPI run is deliberately not
here -- the cap interacts with how events divide across BDs, which makes the
assertions fragile -- so the capped case is covered single-process above.

### Not covered

Run 77 has jungfrau in every event, so no subbatch is ever rowless. R7's
starved-stream case therefore has no real-data run; the unit tests carry it
(`test_gpu_task_worker_finish.py`). A dataset where some events lack the task
detector would be the end-to-end check.
