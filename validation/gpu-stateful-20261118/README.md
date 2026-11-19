# Stateful GPU task on real data (PR #270 phase 1, step A)

`stateful_run77.py` runs mfx100852324 run 77 uncapped through
`GpuTask(factory=...)` with two execution streams. Each worker instance counts
the events of every subbatch it processes into `bins[step_scope.index]`, held
in `ctx.state` inside the setup reservation, and reports the bins from
`end_run`. Summed over both instances the three steps must each hold 1000
events and the total must equal what `run.events()` delivered.

    PS_PARALLEL=none python3 validation/gpu-stateful-20261118/stateful_run77.py

| | result (A100, jobs 45720049 step A, 45734339 step B, 45748485 step C, 45757000 step D) |
| --- | --- |
| events delivered | 3000 |
| instances, status | 2, both `complete` |
| per-instance bins (scopes 0..4) | `[0, 500, 500, 500, 0]` each |
| per step (scopes 1..3) | `[1000, 1000, 1000]` |
| pre-step or beyond | 0 |
| step hooks, per stream | `begin_step`/`end_step` for scopes 1, 2, 3, once each, in order; none for scope 0, which had no events |
| count read inside `end_step` (job 45734339) | 500, 500, 500 -- equal to each stream's final count for that scope |
| `ctx.emit` -> sink, per scope | one **complete** record each, merged over both streams: `[1000]`, `[1000]`, `[1000]`, all `seq 0` |
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
the run record of a run cut short, are delivered with `complete=False`.

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
stream, zeroed at `begin_step`, emitted at `end_step` -- then re-reads run 77
on the plain CPU path with `run.steps()` and sums each present segment's raw
words per step. Run 77's jungfrau has 32 segments, each `(1, 512, 1024)`.

    PS_PARALLEL=none python3 validation/gpu-stateful-20261118/cube_run77.py

| | result (A100, job 45757000) |
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
