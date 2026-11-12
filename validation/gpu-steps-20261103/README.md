# `run.steps()` on GPU BD ranks under MPI

Integration evidence for issue #153, GPU-140-03: `RunParallel.steps()`
returned immediately on a GPU BD rank, so `run.steps()` silently yielded
nothing on the GPU path. PR B puts ordered `BeginStep`/`EndStep` envelopes in
the stream and `_steps_impl` consumes them as the CPU path does.

Two BD ranks, not one. The teardown that `run.steps()` now shares with
`run.events()` closes the batch iterator, then the manager (which barriers on
`device_comm`), then frees `device_comm`. A single-rank run cannot observe a
break in that order.

Run from the repository root after building:

    sbatch validation/gpu-steps-20261103/run.sbatch
    PS_STEPS_DATASET=77 sbatch --export=ALL,PS_STEPS_DATASET=77 \
      validation/gpu-steps-20261103/run.sbatch

## Cases

Two datasets, because they establish different things.

| Dataset | Events | Asserts | Evidence |
| --- | --- | --- | --- |
| 51 (`mfx100848724` r51) | 120 | the mechanism: `steps()` yields at all, shared teardown releases cleanly at 2 BD ranks | job 43000204, `MPI_STEPS_PASS` in ~90 s |
| 77 (`mfx100852324` r77) | 2400 | the real multi-step case: a 3-step Jungfrau gain-mode scan | job 43001391, `MPI_STEPS_PASS` in ~150 s |

Run 51 has one step, so it proves delivery and cleanup but not step
*boundaries*. Run 77 is the structural case:

    per_step: [[540, 500, 200], [460, 500, 200]]   counts: [3, 3]

Steps 0 and 1 sum to 1000 each, matching the 1000-events-per-step structure
recorded independently in job 42149659. Events are partitioned between the two
BDs, not duplicated. Step 2 is truncated to 400 by `max_events=2400`, which is
the expected cut of a trailing step.

`max_events` must clear the steps being counted or the assertions measure the
cap rather than the stream: at the 120 used for run 51, run 77 stops inside
step 0 and no number of BD ranks can ever see 3 steps (job 43000712).

## The uncapped case, which is the one that matters

Every capped run stops **before** `EndRun`, and `EndRun` is the record whose
misordering loses data: it was delivered ahead of the final packet's events,
and both `Run.events()` and `_steps_impl` return on it. So no capped run can
catch that, however many steps it checks.

    PS_STEPS_NO_CAP=1 drops max_events entirely:

    SBATCH_ACCOUNT=lcls:prjdat21 SBATCH_QOS=normal sbatch \
      --export=ALL,PS_STEPS_DATASET=77,PS_STEPS_NO_CAP=1 \
      validation/gpu-steps-20261103/run.sbatch

    python3 validation/gpu-steps-20261103/serial_events.py   # serial counterpart

Both pass on run 77:

| Path | Result |
| --- | --- |
| serial `run.events()` | 3000 of 3000 events |
| serial `run.steps()` | `per_step [1000, 1000, 1000]` |
| MPI, 2 BD ranks | `counts [3, 3]`, `per_step [[500,500,500],[500,500,500]]`, total 3000 |

Job **43386685**, after the transition-release fix. Job 43349533 is the same
pair of cases on `c0e0d5d9a`, *before* that fix, and also passed — worth
recording, because it shows what this harness does and does not cover: the
delivery-order defects it missed needed a memory-pressure flush mid-packet,
which a normal run does not provoke. See the note below.

The serial event count is the sharpest check in the whole harness: one
number, against a known run size, on the path that returns at `EndRun`.

## The memory-pressure path, which the two runs above do NOT cover

Neither uncapped run hits `GpuMemoryPressureError`, so neither exercises the
mid-packet flush where the delivery-order defects lived — both passed before
that fix. `pressure_events.py` exists for exactly that path. It runs run 77
uncapped, serial, with a per-BD budget small enough that
`_retire_issue_and_yield` takes its recovery path, and it **counts the
recoveries**: a run that never hit pressure proves nothing, so it prints
`PRESSURE77_INCONCLUSIVE` and fails rather than passing vacuously.

    PS_PARALLEL=none PS_PRESSURE_BATCH=200 PS_PRESSURE_GB=0.25 PS_PRESSURE_STREAMS=4 \
      python3 validation/gpu-steps-20261103/pressure_events.py

Budget and batch size matter together: at these batch sizes 0.5 GiB never
recovers (inconclusive), 0.25 GiB recovers on nearly every subbatch. Too small
and admission raises "cannot fit alone", a different and legitimate error.

| batch | budget | streams | result |
| --- | --- | --- | --- |
| 200 | 0.25 GiB | 4 | **999 recoveries**, 3000 of 3000, `per_step [1000, 1000, 1000]` |
| 500 | 0.25 GiB | 8 | **999 recoveries**, 3000 of 3000, `per_step [1000, 1000, 1000]` |
| 200 / 500 | 0.5 GiB | 4 | 0 recoveries — inconclusive, as designed |

Identical on job **43402513** (Nov 5, before the rebase onto #262) and job
**44524024** (Nov 12, at `a29258975` with #262 and the Copilot-review fixes
in). Unit coverage of the same path: `test_gpu_transition_order.py`,
`test_gpu_step_delivery_properties.py`, `test_gpu_process_batch_transitions.py`.

Note the account override. The `#SBATCH` header requests `preemptable`, but
`lcls:prjdat21@ampere` is the association that grants `normal` — without the
override the job waits behind priority.

## Two failures this harness exists to catch

**Run 77 was unopenable.** Its Configure records failed the GPU config parser
-- `field jungfrau.config.user.gainMode:gainModeEnum segment 19 is owned by
more than one routed Configure Names record` -- because the gain-mode scan
declares jungfrau segments 11-31 twice under `config` in stream 3.
`detector_field_handles()` now omits config fields, whose handles the only
caller discarded a line later anyway. Fixed separately; run 77 opens in serial
as of job 42996269.

**Every rank must enter `run.steps()`.** smd0 and the EB yield nothing from
it, but only after `RunParallel.start()` has run their blocking service loops:
`smd0.start()` reads chunks and feeds the EB, and `eb_node.start()` ends by
sending each BD the empty message that terminates it. Gating the loop on
`role == 'bd'` skips those loops, so the BDs park in `bd_comm.Probe(source=0)`
waiting for a batch nobody will build while smd0 and the EB sit in the gather
below -- a circular wait with no timeout of its own, which hangs until the
job's time limit. Observed as job 42992578: 40+ minutes with no output, both
BDs in `PMPI_Probe` and smd0/EB already in `PMPI_Gather`/`PMPI_Barrier`.

Hence the `timeout` in `run.sbatch`, and the dataset-scaled limit: an MPI
deadlock produces no output and no error, so without a cap it is
indistinguishable from a slow job until the wall clock expires.
