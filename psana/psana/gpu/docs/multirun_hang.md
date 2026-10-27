# Multi-run MPI jobs hang at the second run

**Status:** pre-existing on `features/psana2-gpu`, this branch's parent. Not
introduced by the device-placement / CUDA-IPC work, and not fixed by it.
Tracked separately as [P2 #209](https://github.com/slac-lcls/lcls2/issues/209).

**Effect:** a psana2 MPI job given more than one run processes the first run,
then stops. The second run's BD workers each receive zero events, the EB rank
never returns from it, and the job hangs until the launcher kills it.

## What was measured

A small synthetic dataset -- two `xtcwriter` runs, 20 events each, ~50 KiB
per file -- driven through `DataSource(exp=..., run=[14, 15])` with
`PS_EB_NODES=1`. Event counts come from the summary each case emits;
`rc=124` is `timeout` killing a hung job.

| Revision of `psana/psana/psexp` | Result |
|---|---|
| master fork point `f743ef5be` (pre-GPU) | **both runs delivered**, 20 + 20 events |
| parent `features/psana2-gpu` `9b32dda19` | hang, `rc=124`, no summary |
| parent + this branch's `mpi_ds.py` | hang, `rc=124`, no summary |
| this branch, working tree | hang, `rc=124`, no summary |

`mpi_ds.py` is the only file under `psexp/` that this branch changes relative
to its parent, and swapping just that file onto the parent changes nothing:
the parent hangs either way. The failure is therefore inherited.

Controls, same dataset and same driver:

* **a one-element run list** (`run=[14]`, 6 ranks) -- passes, 20 events. So
  neither the list form nor the driver is at fault; only crossing a run
  boundary fails.
* **3 ranks** (one BD worker) -- hangs identically, so it does not depend on
  how many workers there are.
* **with and without `max_events`** -- hangs either way, so the cap is not the
  trigger. (`max_events` is per-run in serial mode: `run=[77,78,79]` with
  `max_events=10` yields 10 events from each of the three runs, because
  `_setup_run` builds a fresh `SmdReaderManager` per run number.)

The same hang reproduces on real data (`mfx101629726` r0015/r0016, 6 ranks,
CPU-only, no `gpu_det`), so it is not an artefact of synthetic files.

## Where it is

`BigDataNode._batch_envelopes` (`psexp/node.py`) sends its next batch request
to the EB *before* yielding the current envelope -- a one-message look-ahead
that hides EB build latency behind the BD's I/O. `start()`'s `finally` drains
that outstanding request to EB termination when the consumer stops.

This look-ahead does not exist at the master fork point, which is the only
revision tested that completes two runs. That makes it the mechanism to
examine: an outstanding request, or a drain that consumes the termination
message belonging to the *next* run, leaves the EB and the BD workers
disagreeing about which run they are in. The exact interaction has not been
traced -- only localised to the GPU lineage and to the run boundary.

## Consequence for the test suite

`psana/psana/tests/gpu/integration/mpi_multirun_placement.py` exists to cover
per-run rediscovery -- that run N+1's budget is not sized against a card still
holding run N's memory (review findings N1, N2 and finding 1). It cannot pass
until this is fixed, because psana does not reach run 2 at all.

It is kept, and wired into `run_cases.py`, for two reasons: it is the
reproducer for this bug, and it is the coverage those findings asked for the
moment the multi-run path works. The case is marked `expect='hang-known'` so
the harness marks reproduction of the *known* failure as an expected outcome.
A suite-level PASS therefore does not mean successful multi-run processing;
if multi-run starts working, this case fails and says to update the expectation.
