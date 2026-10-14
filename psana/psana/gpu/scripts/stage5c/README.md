# Stage 5c matched calibration/integration performance campaign

Latest accepted measurements: [current performance report](../../docs/performance/user_kernels.md).

Benchmark harness only; no production runtime changes. Runtime baseline is
`6ba5fa586` (Stage 5b). The user callable is the same validated
`JungfrauAzimuthalIntegration` in both variants.

- `batched_task`: one user analysis invocation and two kernels per subbatch;
  normal psana histogram publication/D2H and public `.on_cpu` delivery.
- `event_loop`: the same two kernels per event in the public `Run.events()`
  consumer; a reused pinned destination is copied to an independent NumPy
  histogram for each event. Inputs are protected with actual `on_gpu_view`
  leases through the blocking copy.

Both variants declare the same dense input and original constants in GpuTask,
use the same parser/gather batching and physical segment map, and use the same
pipeline memory quota. In the reference, the framework callback only captures
read-only inputs/constants; benchmark-only `_submit_gpu` plumbing exposes a
leased dense input to the event loop. Its host exposure time is recorded
separately and remains included in loop time. This adapter is not a new public
input API and is not part of the external user algorithm. Framework callback
counts and user analysis invocation counts must be reported separately.

The two paths return the same independent float64 `(3,64)` histogram per event.
Per-event and batched numerical algorithms, masks, precision and bin maps are
identical. Every pair must have identical timestamp-associated histogram
hashes. Diagnostic processes additionally validate selected histograms against
independent NumPy calibration/reduction and compare raw inputs to the existing
run-387 CPU reference. Counts are exact; sums/means use Stage 5b's tolerances.

## Dataset and scope

Run 387, 10,000 events, the same five JF streams and immutable calibration
snapshot as the earlier JF scaling tests. The bin map is the real run-51
physical-layout radial map validated in Stage 5b, deliberately held fixed as
an integration workload. It is not represented as run-387 beam geometry or
q-space scientific calibration. Its source and SHA-256 are retained.

A private node-local prefix copy isolates cache preparation from other jobs.
Bulk reads are on, with eight KvikIO workers/BD and 1 MiB tasks/targets, one EB,
KvikIO CPU fallback, no feespec. Warm samples require >99% resident measured
prefixes; cold samples require <1%. Historical staging-only rates are not the
same-work denominator for this campaign.

## Matrix and acceptance gates

Warm configurations (GPUs, BDs, input batch size, stream depth):

```text
(1,1,20,2)  (1,1,5,2)  (1,1,20,1)
(1,2,20,2)  (1,4,20,2)
(2,2,20,2)  (2,4,20,2)  (4,4,20,2)
```

Each runs both variants over 1,000 events with counted/timed kernel submissions,
publication groups, output pinned-memory peak and CuPy live-allocation peak.
The shorter 200-event attempt left a BD idle in one topology; 1,000 events
provide more batches without forcing destinations or relaxing participation.
The diagnostic reference is independently derived from SMD records and checked
against the existing 200/10,000-event reference before freezing.
All 16 diagnostics must pass before timed samples start. Diagnostic event
synchronization and CPU comparisons are excluded from throughput samples.

The main one-BD batch-20/depth-2 case has six balanced warm pairs. Other warm
cases have four pairs each. The main case also has four cold pairs: 38 pairs,
76 timed samples in total. Pair order reverses by round; topology order also
reverses. Output, event count, topology, cache, batch size, depth, GPU bus map,
pipeline budget and diagnostic status must match before a pair is accepted.
Positive and negative performance differences are retained equally.

The explicit per-BD pipeline quota reserves `depth * batch_size * 80 MiB`
for user full-JF calibration/validity scratch plus 2 GiB for tables, allocation
slack and context. It is identical for both variants. A configuration with less
than 4 GiB left for the pipeline is rejected before DataSource construction.
These quotas do not change psana's user-allocation contract; device peaks are
also sampled with nvidia-smi every 250 ms (not an exact instantaneous maximum).

## Metrics and limits

Primary loop seconds include lazy first-use setup and event delivery; explicit
DataSource/Run setup is separate. Each worker also records first-event time,
first user-submission time (table preparation, upload and compilation), total
user submission time, I/O wait and resource accounting. Runner sample wall time
includes MPI startup/imports/initialization/cleanup but excludes prior cache
preparation. The historical after-first-event rate is retained as a secondary
metric with its explicit definition; it is not a perfectly isolated steady-state
measurement, especially when an entire first subbatch is already ready.

CUDA event timings are collected only in diagnostics, separately for calibration,
integration and output D2H. D2H markers include stream-side gaps incurred while
issuing the copy commands; they are not an isolated memcpy-bandwidth benchmark.
Actual user launches should be 40 versus 2 for a full 20-event subbatch. Results
also expose CPU handoff overhead, table memory and repeated first-use costs.
Do not equate small histogram byte volume with low scheduling cost or infer a
speedup before the balanced paired samples complete.

`run.py` freezes no files itself: the launch root contains copied Python/runtime
sources, scripts, native dependency links, input metadata, bin-map provenance,
and a hash manifest. It verifies those hashes before and after the campaign.
`results.json` updates after every sample; `pairs.json` only after both matching
variants pass; `summary.json` and `STAGE5C_CAMPAIGN_COMPLETE` require full success.
Only this invocation's private node-local staging directory is removed on exit.

## Completion review and scaling follow-up

`kernel_check.py` separately times the same kernels on the first real event,
with hot buffers and allocations/I/O/D2H outside the CUDA intervals. It checks
batch-1/batch-20 histogram equality and records contributing pixels and device
properties. This is supporting evidence, not another end-to-end sample.

`review.py` independently checks complete matrix coverage, actual diagnostic
counts, all paired outputs, and source/log hashes. `plausibility.py` compares
its kernel timers to that isolated check and rejects impossible loop times or
large unexplained slowdowns. Neither requires a positive batching speedup.
`followup.py` polls scheduler status, runs both gates, writes the review report,
and submits the explicitly authorized scaling jobs only on success. Submitted
IDs are saved after each launch, so restarting the follow-up skips those jobs.

`scaling.py` uses the batched user kernels for the previous full JF matrix and
the one-GPU 1/2/4-BD JF+feespec matrix, bulk OFF/ON, cold/warm, two repetitions,
batch 20/depth 1. Mixed processing retains the prior per-event feespec sum and
independent validation. It reserves batch scratch plus 1 GiB per BD outside
the framework budget and requires at least 1 GiB of framework quota. The
runtime may split execution subbatches to fit; actual callback sizes are saved.
All requested BDs must process diagnostic and timed events. Diagnostics use
4,000 events for six or more BDs and 1,000 otherwise, with SMD-derived references.
These scaling workloads include both user kernels; they are not matched-work
comparisons with the earlier staging-only rates.
