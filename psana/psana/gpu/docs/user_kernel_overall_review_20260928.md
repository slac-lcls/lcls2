# Overall user-kernel support review — 2026-09-28

**Resolution update:** Stage 4 is committed as `9730a9982`. Finding 1 is fixed
by `10173a263` ([geometry validation](geometry_cache_fallback_20260928.md));
finding 2 is fixed by `10df4c6e3`
([serial cleanup validation](serial_gpu_close_20260928.md)). Current API docs
have been refreshed. The dated review below records the original findings and
review scope; it does not describe those fixes as still open. Stage 5's scientific
example and performance campaigns are now complete. See the
[Stage 6 consolidated acceptance](user_kernel_stage6_20260929.md) for final lifecycle checks.


Review basis: pre-user-kernel baseline `480f7074c` through `6291d21ed`, plus the
current uncommitted Stage 4 runtime, tests, and performance evidence. This is a
read-only runtime review; no implementation fixes or commits were made.

**Updated classification after comparison with master:** the geometry hazard
already exists in local `master` (`a233e797f`) and the locally available
`origin/master` (`f743ef5be`, dated September 25). Both seed only the default
index variant and use the same collective cache-miss path. No remote fetch was
performed. The pre-user-kernel GPU branch had an additional warmup that masked
this hazard; Stage 1b removed it. Consequently, this is a loss of protection
relative to that GPU branch baseline, not a newly introduced master regression.
The initial P1 merge-blocker classification was too broad and is withdrawn.
Track geometry cache-miss safety as an existing issue rather than requiring GPU
geometry startup to be restored for user-kernel support. Resolve the inherited
serial cleanup gap before relying on early-close ownership guarantees for the
new public task API. Recorded test and performance results remain valid.

## Findings

### 1. Existing master issue — Unseeded geometry requests can hang MPI consumers

Location: `psana/psana/psexp/mpi_ds.py:342`,
`RunParallel._setup_jungfrau_shared_caches`.

Stage 1b removed `iface._pixel_coord_indexes(all_segs=True)` along with automatic
GPU geometry setup. That warmup also ran for CPU-only and hybrid Jungfrau
detectors. They still receive a shared geometry cache, but now only the default
`all_segs=False` index variant is seeded. The two variants have different cache
keys even when all detector segments are present.

When event-processing code subsequently calls
`det.raw._pixel_coord_indexes(all_segs=True)` on a BD rank, the cache miss in
`AreaDetector._pixel_coord_indexes` enters `shm_comm.bcast` and later a barrier
(`areadetector.py:230`). If other ranks in that shared-memory communicator do
not make the same call, the job hangs. This applies with default `PS_GEO_SHARE=1`
and a communicator with multiple ranks. It is not a claim that ordinary
`det.raw.image(evt)` always uses this path.

Evidence: a focused probe extracts the actual warmup calls from the baseline
and current startup methods, runs them through the real `AreaDetector` method
and `SharedGeoCache`, then simulates a BD-only all-segment request. The baseline
is a cache hit with no collective; current code attempts a collective. MPI
completion is simulated to report the call rather than deliberately hang a job;
no new multi-rank hang experiment was run.

Possible fixes: retain the all-segment warmup for remaining CPU/hybrid consumers, or make
later geometry cache misses safe without participation by unrelated ranks.
The existing exclusion of exclusively GPU-owned detectors can still avoid
their automatic geometry work. Add a regression covering a BD-only request
after shared startup, including CPU-only and hybrid selection.

### 2. P2 — Public serial iterator close does not reach task/output cleanup

Locations: `psana/psana/psexp/run.py:142`,
`psana/psana/gpu/gpu_events.py:1039` and `:1068`.

`Run.events()` iterates over its manager without a cleanup `finally`. Calling
`events.close()` after the first delivered event therefore leaves the manager
suspended, with execution slots occupied and `PublicationD2H.close()` and
requested-constant cleanup uncalled. A loop-body exception or abandoned
iteration has the same missing deterministic cleanup route. There is no public
serial `run.close()` counterpart to the private manager close used by tests.

There is a second part to the same lifecycle gap: the manager's final
`yield from self.finish()` is in a `try/except/else` **else** block. Closing its
internal generator while it is yielding final deliveries bypasses the exception
cleanup. `EventPool.flush()` retires the currently yielded slot in its `finally`,
but does not drain the remaining slots or complete the rest of `finish()`.

Evidence: the real Python iterator, manager, and EventPool control flow with
simulated stream synchronization and two occupied slots produced:

| Action | Manager closed | Occupied slots | Output close called |
|---|---:|---:|---:|
| Close public `Run.events()` after one event | false | 2 | no |
| Explicitly close private manager afterward | true | 0 | yes |
| Close internal iterator during final flush | false | 1 | no |

**This plumbing gap already exists at the baseline.** It is not a new Stage 4
regression. It now also governs user scratch, requested constants, and published
output buffers, and limits the new guide's “closing a run” guarantee. The probe
demonstrates missing cleanup, not a measured CUDA use-after-free. Existing
serial device tests explicitly call `manager.close()` in their own `finally`,
which bypasses this public integration gap. MPI has explicit manager cleanup.

Fix: define a public serial close/iteration cleanup contract, connect it to
manager close, and make final draining interruption-safe. Preserve resumable
iteration deliberately if required; simply breaking a Python loop is not by
itself a reliable close protocol. Add public early-close, loop-error, and
final-flush interruption tests with multiple live executions and retained host
results, without private cleanup masking the behavior under test.

### Documentation follow-up

`GpuTask`'s docstring (`gpu_task.py:47`) and `EventPool.submit` still say public
processing is gated until Stage 4, although Stage 4 enables it. Update these
descriptions and specify the public early-close protocol when fixing finding 2.

## Scope and conclusions by component

| Component | Overall change | Review conclusion |
|---|---|---|
| Stage 1 / 1b | Separate dense raw preparation; remove built-in calibration, image scattering, automatic calibrated-output D2H and calibration IPC | Removal is intentional; finding 1 concerns collateral CPU/hybrid geometry behavior |
| Stage 2 | Host-only `GpuTask` declarations, exact input/constant selectors, per-BD requested constant uploads and refresh | Validation, atomic replacement, bytewise change detection, and transition draining are consistent |
| Stage 3 / 3a–3c | Selected/aligned rows; one callback per execution subbatch; lazy batched metadata; publication groups and scratch retention | Dispatch occurs outside per-event delivery; no remaining per-event callback or user-kernel scheduling was found |
| Stage 4 | Grouped D2H, bounded pinned output staging, exact named host results and public serial/MPI integration | Mapping and completion ownership are consistent; serial early-close integration remains incomplete |

Reviewed dense/generic input alignment, missing fields and physical segment IDs,
max-events tails, BeginStep refresh/draining, declared constant lifetimes,
scratch registration, borrowed-input publications, source and destination
retention, partial-copy/event-record/drain failures, quarantine/retry,
sparse/reordered publications, scalar/empty rows, shape mutation, exact naming,
pinned cap/fallback, retained result materialization, CPU service imports, and
MPI exclusive/hybrid selection. No additional blocking issue was identified
in those paths beyond the findings above.

The implementation schedules one callback per **selected execution subbatch**,
not necessarily one per whole EventBuilder batch: memory admission can split it.
For multiple publication groups there is one payload copy per nonempty group
and one output completion event per execution. Per-event host lookup and NumPy
row materialization remain in delivery. User GPU allocation policy remains the
user's responsibility; the psana device budget does not bound arbitrary user
scratch/output allocations.

## Verification and performance limits

- Re-ran task, batch-context, producer, publication-D2H, and result-lifetime unit
  tests: **87 passed**. Imported runtime files were hash-matched to this checkout.
- Added review-only [probes](../../../../validation/user-kernel-overall-review-20260928/review_probes.py)
  and captured [output](../../../../validation/user-kernel-overall-review-20260928/review_probes.log).
  These use actual Python control flow with simulated CUDA/MPI completion;
  they do not claim device or multi-rank validation of the suspected failures.
- Existing Stage 4 evidence remains: 466 local units, 533 main tests plus four
  byhand tests, 135 A100 tests, and public-task MPI checks. Those earlier runs
  were not repeated for this review, and did not cover the two scenarios above.
- Recorded batching counts support the intended scheduling change. GPU tests
  instrument actual `.get()` submissions and completion-event creation;
  benchmark publication-group counts alone are not native copy traces.
- Performance is workload-dependent. N=20 compact results improve public-loop
  scheduling substantially; the real DataSource comparison has a median paired
  loop reduction of about 7.45%, with considerable I/O variability. Full-image
  fallback is about 23–26% slower than a reference that reuses its ordinary
  host destination, but about 2% faster with matched fresh-host allocation.
  N=1 small workloads also have overhead. These are documented policy/workload
  limits, not evidence for a universal no-regression claim.
- Input-only comparisons show small changes amid substantial A/A variability;
  a small regression cannot be excluded. No new performance job was needed to
  establish the control-flow findings.

After fixes, run focused geometry/early-close regression tests and the relevant
CPU, A100, and MPI suites. Recheck startup timing if geometry prewarming changes;
repeat throughput testing only if the fixes alter steady-state scheduling or
copy behavior. Keep the existing performance campaigns as evidence for the
unchanged batching and output policies.
