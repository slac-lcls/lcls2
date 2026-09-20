# Batched canonical gathering: implementation plan

Revised 2026-09-18 after review of stream-grouped batched locators and the
bulk-read branch through `8f94e3c7b`. The implementation now follows this plan; see the
[current review and call path](batched_canonical_gather_review.md).
The completed locator work is described in [the review](batched_locators_review.md)
and [the matched benchmark](performance/batched_locators_abo_sdf.md).

Completed through the on-demand locator-view follow-up on 2026-09-20.
See the [final pre-bulk review](batched_pre_bulk_review.md) and
[latest same-allocation comparison](performance/lazy_locator_wrappers_sdf.md).
The original implementation sequence and bulk integration contract follow.

## Objective and review boundary

Gather all canonical detector rows for the existing execution subbatch with
one GPU launch per supported detector. Remove the Python event/segment loop
from field access, dependency submission, and copy submission. Keep the current
execution sizes, pool depth, event delivery, calibration algorithm, and reader.
Calibration and missing-row cleanup remain per event in this first change.

Implement and measure on the locator branch first. Bulk-read integration is a
subsequent reviewed merge into the then-current bulk history; do not replay or
rewrite published commits. General public field-view materialization and its
existing locator-row readback are outside this detector gathering change.

The matched 10,000-event baseline is A 21.850 s, B 33.561 s, and B+ 28.743 s.
B+'s field-access/dependency/gather host scope is 7.241 s versus A's 0.853 s.
These separate instrumented medians identify a target, not a promised saving.

## Implementation sequence

1. **Compile the detector gather plan once.** From the existing binding and
   parser handle-to-output mapping, build a numeric table in canonical segment
   order: input XTC stream, stable locator handle index, expected type/rank,
   and canonical destination row. Upload once per binding/configuration lifetime,
   account for its bytes, and order its upload before consumption. Keep the
   current canonical ordering and supported uint16/float32 paths. Validate
   configured handles before the hot path; preserve lazy lookup for other APIs.
2. **Prepare execution indexing once per subbatch.** Build a compact mapping
   from the selected execution events and required streams to input-local dgram
   rows, with an explicit absent sentinel. Use existing host descriptors; never
   copy GPU locator metadata back to schedule gathering. Share this mapping
   among detector consumers where practical. Retain the current detector event
   selection, timestamps, and ordering when an event lacks all its sources.
3. **Expose an internal descriptor for combined locator storage.** Carry the
   backing array, active row count, allocated capacity stride, raw input base
   and byte length, ready dependency, and strong owner reference. Do not recover
   capacity from an active tail view or rely on segment order matching handle
   order. Keep the existing per-handle public locator API compatible.
4. **Submit dependencies once.** Gather follows location on the same CUDA
   stream without an extra wait when ordering is established. For another
   stream, wait once per distinct required ready event. Preserve additional
   lazy-locator dependencies when present. Never replace consumer-lifetime
   tracking with the locator-ready event.
5. **Launch one canonical gather.** Use a grid covering execution event,
   canonical segment, and pixel tile. Each work item resolves its input row and
   stable handle index, then reads the locator at
   `((handle_index * capacity) + dgram_row) * LOC_NCOLS`. Match the existing
   gather checks for FOUND status, type, rank, exact byte count, and raw-buffer
   bounds, including overflow-safe range checks. Preserve existing shape
   acceptance; stricter shape validation would be a separate behavior change.
6. **Write every output pixel and presence entry.** Valid rows copy pixels;
   absent or rejected rows write zero. Exactly one designated thread writes
   each presence byte, including zero. This avoids stale data on reused slots
   without racing a separate initialization phase inside the kernel. Keep the
   post-calibration missing-row cleanup: calibrating zero raw pixels need not
   produce zero output. Float32 passthrough gathers directly to its output.
7. **Reuse current calibration and delivery.** After the whole gather, loop
   through the existing event slices to submit calibration/cleanup and create
   EventContext results. Preserve input and output leases through all device
   consumers. Account for mapping/table buffers, growth peaks, rollback, and
   tail reuse; any asynchronous host upload source must remain alive until done.

Primary code: `gpu_detector.py:GPUDetector.process_batch`, its gather kernel,
`gpu_input.py:GpuDetectorBinding` and event mappings, and a small internal
descriptor interface in `gpudgram/parser.py` / `gpudgram/batch.py`. Change
`gpu_stream.py` only if shared execution-map ownership needs it. Avoid adding
a general user-facing scheduling API for this optimization.

## Expected call path and counts

```text
setup -> Configure/locator tables + canonical gather plan uploaded once
execution submit
  -> parse input (existing batched locators)
  -> prepare event/stream row map
  -> establish distinct input dependencies
  -> gather all events and canonical segments [one kernel per detector]
  -> existing per-event calibration and missing-row cleanup
  -> existing result delivery and lease-based retirement
```

For the measured dense Jungfrau workload (20 events, 32 segments), target
gather kernels per execution are 640 -> 1. The 640 gather-path locator waits
become zero extra waits on the producing stream, or one shared locator wait
for one input produced on another stream. Count other pool/walker dependencies
separately. Calibration and cleanup remain 20 launches each. Locator decoding
and initialization remain one launch each per newly parsed input. Record map
uploads and any initialization separately; do not hide replacement overhead.
Empty detector selections submit no gather kernel.

## Bulk-read integration note: parsing and execution have different boundaries

Reviewed commits: `a5f07ee38` (input owners), `763a8df1b` (admission),
`694b9ff2b` (resident inputs), and `4a26bc640` (small-dgram priority).

The bulk branch calls `gpu_events.py:_start_resident_input` to read all admitted
physical streams for the full EB batch and call `GpuXtcBatchPool.parse_window`
once. `_submit_gpu` parses only newly read transient input for each execution.
`EventPool.submit` waits on the supplied windows; `GpuEventDgrams.from_windows`
composes existing rows without reparsing. Without residency, EventPool creates
one input window for each execution. The bulk branch still uses the old
per-handle eager location loop; B+ is not integrated there yet.

For a full 1,000-event batch with 10 events per execution (100 executions),
assuming both detectors are present at every event:

| Residency | Newly parsed input windows | Expected B+ locator launch sets after integration |
| --- | ---: | ---: |
| None | 100 combined transient windows | 100 |
| epix100 resident, Jungfrau transient | 1 resident + 100 transient | 101 |
| All input resident | 1 combined resident window | 1 |

Each locator set is one initialization kernel, one decoding kernel, and one
shared ready event; each parsed window also has one walker. All admitted
streams share the resident window, and all transient streams for an execution
share its transient window. These are predicted counts, not a merged benchmark.
Physical read requests and KvikIO tasks do not determine parser launch counts.

At 32 MiB Jungfrau and 1 MiB epix100 raw input per event, resident epix100
needs roughly 1,000 MiB plus parser storage. Jungfrau can dominate the execution
budget: its current path adds approximately 32 MiB raw scratch and 64 MiB
calibrated output per event. Residency is chosen per physical stream, so some
Jungfrau streams can be resident and others transient. Admission uses total
working memory, fixed costs/headroom, and concurrency; sizes alone do not prove
that epix100 will fit or that the execution size will be ten.

**Integration contract:** locate once per input window; gather once per
detector execution subbatch. Resident locators are reused across executions.
Gathering a resident detector still follows execution boundaries, because
output/scratch capacity and delivery have those boundaries. In this example,
Jungfrau would still have 100 batched gather launches even with all inputs
resident. Epix100 remains on its current generic field-access path unless a
supported canonical gather consumer is explicitly added; no epix calibration
support is implied by this plan.

At integration, extend the execution map with an input-owner index and retain
each owner's independent raw base, locator base, capacity stride, and local
dgram numbering. A detector may span resident and transient windows. Resolve
each source through its owner; do not concatenate/reparse resident input or
assume one raw base. Deduplicate shared events in `InputWindow._ready`, retaining
distinct lazy and downstream dependencies. Update input-owner reservations,
allocation requirements, reporting, trimming, and failure-drain paths for the
combined backing and new mapping buffers. Release only after planned uses and
registered consumers finish. Preserve existing `from_windows` correctness
checks and input leases.

## Acceptance and measurement

- Compare raw and calibrated pixels against the current per-segment gather and
  CPU reference. Cover reordered/noncontiguous segment IDs, stream/Names
  isolation, sparse/absent sources, invalid locators, bounds/type/rank/size
  rejection, uint16 and float32 passthrough, empty input, tails, and slot reuse.
- Exercise same-stream and cross-stream execution, delayed consumers, growth
  failure/rollback, and explicit allocation strides. Verify complete zero and
  presence writes and post-calibration missing-row zeros. Tests should catch
  incorrect data or lifetime behavior rather than mirror table construction.
- Run focused CPU/GPU tests, applicable main/MPI regressions, and real-data
  pixel acceptance before timing. At the later bulk merge, add mixed/all/no
  residency tests, multi-owner detectors, different owner capacities and local
  row indices, and reuse of resident locators across many executions.
- Compare B+ versus B+gather warm in one allocation with matched dataset,
  batch 20, depth 1, 8 GiB budget, KvikIO settings, calibration, and cache guards.
  Alternate repeated clean samples. Retain A as a reference if available;
  isolate timing/NVTX/Nsight samples from clean throughput.
- Verify gather and dependency counts plus mapping/upload overhead. Report
  clean elapsed time, memory peaks, event order/count, and remaining host scopes.
  Do not assert timing thresholds or promise closure of the 6.893 s gap to A.

Stop after correctness, matched measurement, and a reviewable diff/call trace.
Gather/calibration fusion, calibration batching, new admission policy, larger
execution sizes, and bulk merging remain separate review decisions.
