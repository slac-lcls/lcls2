# Stateful GPU task on real data (PR #270 phase 1, step A)

`stateful_run77.py` runs mfx100852324 run 77 uncapped through
`GpuTask(factory=...)` with two execution streams. Each worker instance counts
the events of every subbatch it processes into `bins[step_scope.index]`, held
in `ctx.state` inside the setup reservation, and reports the bins from
`end_run`. Summed over both instances the three steps must each hold 1000
events and the total must equal what `run.events()` delivered.

    PS_PARALLEL=none python3 validation/gpu-stateful-20261118/stateful_run77.py

| | result (A100, jobs 45720049 step A, 45734339 step B, 45748485 step C) |
| --- | --- |
| events delivered | 3000 |
| instances, status | 2, both `complete` |
| per-instance bins (scopes 0..4) | `[0, 500, 500, 500, 0]` each |
| per step (scopes 1..3) | `[1000, 1000, 1000]` |
| pre-step or beyond | 0 |
| step hooks, per stream | `begin_step`/`end_step` for scopes 0, 1, 2, 3, once each, in order |
| count read inside `end_step` | 0, 500, 500, 500 -- equal to each stream's final count for that scope |
| `ctx.emit` -> sink, per scope | one **complete** record each, merged over both streams: `[0]`, `[1000]`, `[1000]`, `[1000]`, all `seq 0` |
| `ctx.emit` in `end_run` -> sink | one record, `[0, 1000, 1000, 1000]`, complete, streams (0, 1) |

The `end_step` rows are the "Hook placement" contract measured: by the time a
stream's `end_step` runs, every `process` for that scope on that stream has
completed, with no drain and no barrier -- the hook is simply enqueued after
them on the same stream. The sink rows add the export path: `ctx.emit` copies
on that same stream and waits for its own copy only, the per-BD reducer merges
the two streams and learns completion from the framework (a stream's
`end_step` returning), and the sink is called inline once per complete scope.
Scope 0 -- before the first BeginStep -- is exported like any other, here with
a zero count.

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
