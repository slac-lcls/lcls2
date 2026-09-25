# CPU attribution for stream-group overhead

2026-09-25. Campaign **39067790**, one A100 on `sdfampere014`.
The experiment profiles the frozen 4 MiB runtime used by job 39038746:
`c8f6b6cdf` plus the existing bulk-target patch. The later transition-drain
fix is validated separately and is not part of this profiling baseline.
Job completed with exit `0:0` in **16m33s**. All **16 samples** and final
hash/placement checks passed.

## Control results

Rates are calculated from median loop time across the two unprofiled controls.

| Bulk | Cache | R1 / R2 events/s | Median-time events/s | Median loop seconds | Profile median seconds |
|---|---|---:|---:|---:|---:|
| Off | Cold | 134.60 / 136.48 | 135.53 | 7.3782 | 7.6462 |
| On | Cold | 119.99 / 121.45 | 120.72 | 8.2839 | 8.9912 |
| Off | Warm | 221.73 / 207.10 | 214.17 | 4.6692 | 4.8998 |
| On | Warm | 170.84 / 177.93 | 174.31 | 5.7369 | 6.5478 |

Bulk-on control loop time is **12.3% higher cold** and **22.9% higher warm**.
Instrumentation adds approximately 3.6–14.1% to the corresponding median
control times, so the profile costs are not direct speedup predictions.
The severe historical warm depth-1 slowdown did not recur on this allocation;
these measurements do not establish its cause or isolate a cross-allocation
4 MiB versus 1 MiB effect.

## Method

One BD, three MPI ranks, 1,000 JF+feespec events, batch 100, execution depth 1,
8 GiB budget, eight KvikIO workers, compatibility mode ON, bulk target and
KvikIO task size both 4 MiB. Private Weka FFB SSD input and frozen CPU
calibration/reference data match the earlier campaign.

Sixteen samples cover bulk off/on, cold/warm, two rounds, each with an
ordinary control and a separate cProfile sample. Round two reverses cache,
variant, and instrumentation order. Both modes retain identical lightweight
read counters. Profiles run only on the BD Python thread around the measured
loop and its existing terminal device synchronization. Warmup, cache
preparation, and profile serialization are excluded. Lazy pipeline setup
and BeginStep calibration inside `run.events()` remain included.

Every sample retains the existing timestamp/feespec checksum and payload/API
request checks, 200-event warmup array checks, cache residency gates, and cold
physical NIC-byte check. The harness checks runtime/reference/script hashes
and Weka placement before and after the campaign.

cProfile observes Python execution and blocking calls on that thread. It does
not measure asynchronous GPU kernel duration or KvikIO worker execution.
Cumulative function times overlap and cannot be summed. Use unprofiled
controls for throughput; instrumented costs identify candidates rather than
predicting recoverable wall time.

## Repeated profile findings

All four bulk-on profiles (two cold, two warm) identify repeated metadata work:

- `InputGroupPool.plan_slots`: 0.591–0.603 seconds cumulative per sample. It
  rebuilds a free-slot list and searches cached capacities for every requested
  group. Roughly 518k best-fit size-key evaluations occur in one sample.
- `KvikioGpuReader._prune_files`: 0.398–0.406 seconds cumulative, called 5,019
  times from `wait_batch`. Each call scans remaining pending read handles.
- `_coalesced_plan`: 0.238–0.265 seconds cumulative, also 5,019 calls. The
  new group adapter recreates a legacy read plan after stream grouping.
- `GpuXtcBatchPool.parse_groups`: 0.313–0.318 seconds cumulative across
  29 batched submissions. Parser launches remain shared across groups.
- `gpu_budget.py` accounts for only about 0.04 seconds additive self time in
  these bulk-on profiles. Reservation counts alone do not identify the main
  Python cost; allocation/native waiting can appear in other functions.

CPU calibration construction contributes approximately 1.1 seconds cumulative
in both modes, inside the measured loop. It is a shared short-run cost rather
than an explanation specific to bulk on. The earlier large warm depth-1
slowdown has not been causally established by this profile.

These findings motivated the three completed changes below. The profile
remains historical baseline evidence; see the [current checklist](../stream_read_refactor_cleanup.md)
for ownership acceptance, completed cleanup and deferred performance work.

## Follow-up implementation status

1. Avoid reconstructing and rescanning all raw-slot candidates per group.
   Preserve best-fit reuse, deterministic ties, small-stream credits, busy
   slot exclusion, and full replacement reservations.
   Implemented and measured: [slot-selection results](stream_read_slot_index.md)
   record over 93% less profiled selection time and variable warm throughput.
2. Maintain pending-file ownership without a full pending-list scan after
   every completed group. Preserve old-file handles until every associated
   future drains, including short reads and partial submission failures.
   Implemented with bulk-on reference counts: [cleanup results](stream_read_file_refs.md)
   record correctness and the isolated performance comparison.
3. Remove repeated legacy-plan construction in the group adapter only after
   sharing its descriptor validation and preserving transition/file fences,
   byte bounds, empty rows, descriptor order, and bulk-off behavior.
   Implemented and measured: [direct group results](stream_read_direct_group.md).
   Correctness passed; throughput acceptance remains unresolved.

Each change received an isolated comparison; the later
[10k bulk-off/on comparison](stream_read_current_10k.md) still found higher
bulk-on loop times. Broader ownership acceptance and legacy cleanup are now
complete, as recorded in [the checklist](../stream_read_refactor_cleanup.md).
Ownership charges and completion dependencies remain required.

## Artifacts

`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-profile-20260925`:

- `builds.json`, `source.patch`, `python/`: frozen runtime identity.
- `run.sbatch`, `job-39067790.log`: exact launch and sample status.
- `job-39067790/results.json`: validated samples, full function/caller tables,
  and hashes of each raw profile.
- `job-39067790/*.pstats`: original per-sample call graphs.
- `job-39067790/provenance.json`, `summary.md`, `summary.json`: campaign
  provenance and comparison.

Maintained harness: `scripts/feespec_bulk_benchmark/acceptance.py --study profile`,
`bench.py --python-profile`, and `profile_summary.py`. The harness's 13 CPU
tests pass, including caller-edge parsing and control/profile separation.
