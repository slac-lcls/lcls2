# Stage 3c: batched scheduling accepted

The corrected Stage 3 satisfies both gates: one user invocation per selected
execution subbatch, and no repeatable performance regression above the 5%
investigation threshold in the matched measurements. Stage 4 is authorized.
Small differences and isolated outliers remain; this is not a claim that every
sample is faster. The public user-event-loop comparison belongs to Stage 4.

## Runtime and review

Stage 3b `a1c7ae252` was reviewed for aligned selection, callback expiry,
publication validation/mapping, shared backing ownership, completion ordering,
and failure quarantine. No correctness blocker was found. Measurement then
exposed unnecessary metadata work for dense-only callbacks.

The accepted correction builds host identities immediately and uploads the one
bulk device metadata block only on first device-identity/generic-field access
inside the callback. Dense inputs, presence, host identities and constants do
not trigger it. EventPool seals initialization before recording producer
completion. Failed initialization cannot expose partial views or retry. Owners
remain retained before async upload, including failed-drain quarantine; admission
still reserves the conservative metadata bound. This changes task-only paths.

The source base is `be8541fdb` plus the four runtime/test hashes in the
[validation record](user_kernel_stage3c_validation_20260927.json). The frozen
runtime is `/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage3c-lazy-20260927-r1/python`.
No native rebuild was needed; verified Stage 1b native dependencies were inherited.

## Correctness and scheduling

- Local GPU unit suite: 449 passed.
- CPU job 39318857: 516 passed, 128 skipped, seven deselected; byhand: four passed.
- A100 job 39318858: 128 passed.
- Four-rank MPI job 39318859: exact 13-event exclusive/hybrid input delivery;
  task setup and constant uploads confined to BD ranks. Public callback delivery
  was still guarded, so this is not Stage 4 MPI execution acceptance.
- Callback job 39319264: 864 counted numerical preflights and 864 timed samples
  across six balanced rounds, 72 cases, with diagnostics excluded from timing.

For N=20, fresh scratch/publication examples have one callback, one allocation
and one user kernel per subbatch, versus 20 of each in reference `c64fcb2ba`.
Empty callback has one invocation; preallocated examples have zero timed user
allocations. Both paths retain one gather and one producer completion event.
Dense-only candidate metadata uploads are zero; the metadata-requesting device
suite separately verifies one bulk upload. No automatic output D2H exists yet;
preflight verification copies are diagnostic only.

## Input-only regression

Job 39317728 completed on sdfampere023 in 1h32m14s, exit 0. Frozen Stage 2
`f5b4cfb0e` versus Stage 3b `a1c7ae252`, no task, identical benchmark-only dense
preparation. The accepted metadata correction leaves this no-task path unchanged;
the callback campaign also tests no task directly against the corrected runtime.

Jungfrau mfx101210926 run 387: 10,000 events, batch 20, depth 1, one A100,
1/2/4 BDs warm and four BDs cold, bulk reads, eight KvikIO workers, 1 MiB task
and bulk thresholds, compatibility CPU fallback. Strict per-file warm >=99%
and cold <=1% conditioning passed. Eight pixel preflights and 40 timed samples
completed; manifests verified before/after. Root:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-stage3c-regression-20260927-r1`.

| Case | Median paired loop change |
|---|---:|
| Warm, 1 BD | +2.79% |
| Warm, 2 BDs | +1.37% |
| Warm, 4 BDs | -2.82% |
| Cold, 4 BDs | -1.28% |
| Identical-code A/A, warm 1 BD | +4.47% |

Positive means slower. Timing varies between pairs; a ratio of separate runtime
medians can differ from the median paired change. No repeatable >5% loop
regression is established beyond observed control variation. Startup is separate:
median paired maximum-rank setup deltas are -0.001/+0.039/+0.097/+0.012 s for
warm 1/2/4 and cold 4 respectively. Both round-2 four-BD candidate samples had
about +2.4 s setup outliers; all other four-BD setup pairs range -0.13 to +0.16 s.
These are preserved, not removed or counted as loop time.

## Callback performance

Job 39319264, sdfampere033, one dedicated A100 and 16 dedicated CPU cores,
`srun --cpu-bind=cores`. Six balanced rounds reverse runtime, profile and mode
order. Profiles are a 900-pixel fixture and synthetic full Jungfrau shape
`(32,512,1024)` uint16, without I/O/calibration. Sizes 1/3/20, depths 1/2;
none, empty, scratch, publication, and matched preallocated variants.
Scratch computes raw+1 with uint16 wrap; publication computes one uint32 scalar
per event from raw[300]+1. Each slot retires before preallocated storage reuse.

Each sample has 500 immediate warmup submissions, then 2,000 micro or 200
full-frame measured submissions. Host GC resets outside timing and remains
on inside it. Compilation, parsing, upload, preallocation and numerical checks
are outside timing; loop timing includes submission and full retirement.
Reports retain submit/retire/loop microseconds per event, paired differences,
and source hashes. Logical user-payload bounds exclude allocator rounding;
after-drain CuPy counters are not peaks.

See the [complete compact performance evidence](user_kernel_stage3c_performance_20260928.json)
and [validation/provenance](user_kernel_stage3c_validation_20260927.json).
Root: `/sdf/scratch/users/m/monarin/gpu-validation/jf-stage3c-lazy-callback-comparison-20260927-r1`.
The 6,197-entry source/native/harness manifest verified before and after timing.

## Failed candidate and measurement history

Original eager-metadata job 39318349 completed on sdfampere019 in 41m29s,
exit 0, but failed performance acceptance: depth-1 micro batch-1
empty/scratch/publication costs rose about 30%/24%/26%; batch-3 empty rose 20%.
Batch-20 scratch/publication improved 42%, which did not excuse small-batch
regressions. Full-frame batch-1 publication rose 11%. Its root and successful
correctness evidence remain under `jf-stage3c-callback-comparison-20260927-r2`.

Whole-node job 39317958 was cancelled while pending; no timed results. Separate
smoke 39317997 passed all 288 counted checks on sdfampere019 in 10m16s. Corrected
pilot 39318894 passed in 48s on sdfampere028; variability prompted the longer
warmup/sample protocol above. Initial local unit attempts used incomplete
native/MPI environments; the corrected environment passed all 449 tests without
changing or skipping tests. Main CPU tests ran in an isolated scratch directory.

The accepted callback job completed in **49m25s**, exit 0. Across all 72 cases,
the largest median paired slowdown was **1.64%**, below the investigation gate.
At batch 20, micro empty-callback loop time fell **11.8%**, scratch **42.8–43.0%**,
and publication **43.2–50.3%** across depths 1/2. Preallocated scratch/publication
improved **33.7–33.8% / 38.2–38.5%**, confirming savings beyond allocation alone.
Micro batch-1 cases stayed within -3.9% to +0.7%. Full-frame batch-20 scratch and
publication improved about **1.3–2.8%**; full-frame batch-1 task cases ranged
-0.4% to +1.6%. These are loop timings, not process startup or end-to-end file I/O.
