# JF warm/cold block timing

2026-09-23. Runtime `ac87a93b2`, integration checkpoint `fb1a221b6`.
Production runtime sources are unchanged. Benchmark-only instrumentation is
in [bulk_phase_timing.py](../../scripts/bulk_phase_timing.py).

## Timing locations and interpretation

| Block | Source | What the host timer measures |
|---|---|---|
| Upstream batch | `GpuEventManager._next_batch` | Time requesting/receiving the next upstream batch |
| Admission | `_split_subbatches`, `_reserve_gpu_subbatch` | Execution/residency planning and growth reservations |
| Resident input | `_start_resident_input` | Inclusive resident setup, read submission, read completion and parsing; exclusive time removes those nested children |
| File and read planning | `GpuFileEpochs.resolve`, `KvikioGpuReader._coalesced_plan`, remaining `issue_batch` work | File identity, coalescing, descriptors and Python bookkeeping |
| Reader buffer | `KvikioGpuReader._ensure_slot_buffer` | Buffer capacity/replacement work |
| Read submission | `issue_batch`'s `for r in ranges` | File lookup, destination views and KvikIO `pread` calls, including any time blocked inside those calls |
| Read completion | `KvikioGpuReader.wait_batch` | Future completion waits, byte checks and bookkeeping; no new wait introduced |
| Allocation | `gpu_allocation.owned_empty`, including imported aliases | Owned allocation calls and host duration, including allocator pool reuse; not a count of `cudaMalloc` |
| Parser | `GpuXtcBatchPool.parse_window/parse`, slot preparation, `GpuEventBatch.__init__/_locate_configured` | Metadata construction/upload and walker/locator submission; not elapsed CUDA kernel execution |
| Event views | `GpuEventDgrams.from_windows` | Mapping the execution to input owners |
| Detector | Gather-map preparation, canonical gather, calibration and cleanup calls | Host map construction/upload and CUDA submissions |
| Existing synchronization | Null-stream sync in `EventPool.submit`, producer sync in retirement/flush, final device sync | Host time blocked by synchronization already present in the baseline |
| Consumer retirement | Lease-join loops in `finish_retire_next` and `flush` | Consumer completion and ownership release |
| Trim | `GpuEventManager._trim_gpu_caches` | Reader/parser/detector cache trimming after execution drain |
| Delivery/orchestration | Resumptions of `_process_batch`, `_flush_event_pool`, `_yield_ready`, `GPUDetector.process_batch`, `EventPool.flush` | Active generator work only; user time while paused at yield is excluded |

The bulk-on resident path drains the previous execution pool and trims caches
before starting a resident read. It then waits for that read before parsing and
submitting the execution. Bulk off follows the nonresident submission path.
These are candidate explanations to test, not attribution from code alone.

Per the user's direction, faster read completion from warm DRAM than cold
NVMe is expected. Keep those waits as context and focus interpretation on
admission, parser work, gather/calibration submission, allocation/trim, event
ownership and existing GPU drain waits. Report GPU synchronization separately
from CPU work. Subtracting read waits is a host-time partition, not a prediction
of throughput with I/O removed, since asynchronous operations can overlap.

## Measurement method

Job **38917703**, one A100 on **sdfampere027**. JF only, `mfx101210926` run 387,
streams 5–9, 10,000 events, batch 20, depth 1, 8 GiB budget, one BD plus SMD0/EB.
KvikIO CPU fallback with eight threads and 1 MiB tasks. No automatic/user D2H
in measured loops. Each fresh MPI process warms 100 events first.

Two rounds of integrated bulk off/on, warm/cold, timers enabled/disabled give
16 samples. Variant, cache and timer order reverse in the second round.
The scheduler CPU mask is fixed across ranks/cases. Private local-NVMe files
are verified at least 99% resident before/after warm samples and at most 1%
resident before cold samples. File-cache preparation, staging and post-loop
cleanup are outside the event-loop timer.

Timers use nested host wall clocks. Exclusive time subtracts child scopes and
partitions BD wall time; inclusive times overlap. Both startup and steady
scopes are retained. No CUDA events or additional synchronization are added.
Existing generator return/send/throw/close semantics are preserved, with each
timer closing before a yield. Statement rewriting rejects changed source
boundaries, and source hashes/range inventories are saved in every result.

Ten CPU tests cover nesting, exclusion of caller time at yields, generator
return/exception/close behavior, source boundaries, rejection of changed
code, and reader-counter accumulation across EB-batch resets. The instrumentation installed successfully against the frozen runtime;
all 26 GPU Python runtime files match the current checkpoint.

## Initial collector correction

The first attempt, job 38916680, completed one cold sample but its post-run
audit rejected the reader byte count. The reader resets I/O statistics for
each EB batch; the collector had retained only the last snapshot. The corrected
collector accumulates completion deltas and request counts, with a CPU test
covering multiple reads and batch resets. Original scripts/logs are preserved
in `validation/bulk-phase-timing-20260923`; its sample is excluded below.
No production code or timing scopes changed.

## Results

Job **38917703** completed all **16 samples**: eight instrumented and eight
timer-disabled controls. Both instrumented CPU-reference preflights passed.
All timestamp, useful-byte, cache-residency, placement, runtime/source-hash
and timing-partition audits passed. Each table entry below is the median
of two real 10,000-event runs. These are exclusive BD host seconds;
kernel submission is distinct from GPU execution and synchronization.

### Selected non-I/O phases

| Phase | Bulk off warm | Bulk off cold | Bulk on warm | Bulk on cold |
|---|---:|---:|---:|---:|
| Admission and resident setup | 2.228 | 2.184 | 3.015 | 3.087 |
| Owned device allocation calls | 0.006 | 0.007 | 0.176 | 0.169 |
| Cache trim | 0.000 | 0.000 | 0.031 | 0.033 |
| Parser metadata and kernel submission | 0.368 | 0.352 | 0.355 | 0.336 |
| Gather map, gather and calibration submission | 1.505 | 1.487 | 1.476 | 1.510 |
| Event views, delivery, ownership and orchestration | 1.771 | 1.740 | 1.772 | 1.802 |
| Existing GPU synchronization | 1.978 | 2.341 | 2.968 | 3.627 |
| Setup and transitions | 1.779 | 2.143 | 1.880 | 1.652 |

**BD wall time after subtracting read-completion waits:** Integrated-off warm 11.050 s, Integrated-off cold 11.714 s, Integrated-on warm 13.472 s, Integrated-on cold 14.039 s.
This residual also includes read planning/submission and time outside
the selected scopes; it is not an I/O-free throughput estimate.

### What the measurements locate

1. **The parser and detector submission costs barely change warm versus
   cold.** After allowing for the expected input wait, there is no large
   cold-specific parser, gather-map, or calibration-submission bottleneck
   in this JF case.
2. **Bulk-on admission/resident setup adds 0.787 s warm and 0.903 s cold.**
   The measured scopes are `_split_subbatches`, `_reserve_gpu_subbatch`,
   and the exclusive part of `_start_resident_input`. The latter rebuilds
   event/cost metadata also visited during planning and reservation.
   Reusing that metadata is a candidate for a separate measured change;
   these scopes do not isolate the cost of each internal expression.
3. **Existing GPU synchronization adds 0.990 s warm and 1.286 s cold with bulk on.**
   The dominant call moves from `EventPool.begin_retire_next` to the
   producer synchronization in `EventPool.flush`, invoked before
   resident setup. This is additional host blocking at that boundary,
   not proof of additional GPU kernel work or an equally large available
   speedup. Inspect its placement/overlap before changing retirement.
4. **Allocation churn is real but small in measured host time.**
   Bulk on spends 0.176 s warm / 0.169 s cold in owned allocations,
   plus 0.031 / 0.033 s in cache trim.
   Preserve buffer reuse as a later optimization, but the call count
   alone does not make it the largest current CPU cost.

### Allocation and batching counts

| Variant | Fixed | Reader | Parser | Detector | Total owned calls |
|---|---:|---:|---:|---:|---:|
| Integrated-off | 10 | 1 | 4 | 4 | 19 |
| Integrated-on | 10 | 500 | 2000 | 2000 | 4510 |

Counts are identical in both rounds and both cache states. They count
`owned_empty` calls, including CuPy pool reuse, not fresh `cudaMalloc`.
Every instrumented sample retains **500 parser calls and 500 canonical
gathers**. No per-event/per-segment gather path was reintroduced.

### Loop-time controls

| Variant | Cache | Timers off R1 / R2, s | Timers on R1 / R2, s |
|---|---|---:|---:|
| Integrated-off | warm | 32.420 / 35.725 | 33.120 / 34.422 |
| Integrated-off | cold | 75.035 / 76.863 | 75.531 / 75.063 |
| Integrated-on | warm | 35.699 / 32.553 | 33.809 / 36.465 |
| Integrated-on | cold | 118.726 / 114.062 | 116.233 / 107.758 |

Controls contain run variability as well as instrumentation effects.
The warm results vary by several seconds; these two rounds locate
repeatable phase costs and do not establish a small throughput gain.
Read-wait differences between DRAM and NVMe are expected and are retained
in the complete artifact tables rather than used as an optimization claim.

### Recommended next scope

First inspect/reuse repeated admission metadata and study the existing
resident-boundary drain placement. Keep full growth reservations, input
owners, and producer/consumer completion dependencies intact. Allocation
reuse can follow, with a measured benefit requirement. This investigation
does not change scheduling, admission policy, or calibration batching.

## Artifacts

Scratch campaign:
`/sdf/scratch/users/m/monarin/gpu-validation/bulk-phase-timing-20260923-r2/`.
Worktree link: `validation/bulk-phase-timing-20260923-r2`.
Job results: `job-38917703/results.json`, `preflights.json`, `provenance.json`,
`primary/*.log`, and final `summary.md`/`summary.json`.
Reproduce the audit with `python summarize_phases.py job-38917703` from the
campaign directory. Completed prior campaigns remain immutable.
