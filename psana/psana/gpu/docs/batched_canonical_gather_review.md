# Batched canonical gathering: implementation review

Implemented 2026-09-18 on `codex/psana2-gpu-batched-locators`, based on
`b0c9c3c02` (B+). Bulk integration remains deferred. The
[implementation plan](batched_canonical_gather_plan.md) provides the scope and
future input-window contract. This document describes the current code.

A subsequent [combined pre-bulk review](batched_pre_bulk_review.md) fixes memory
reporting and records remaining optimization and integration work. Configured
Python locator views are now created only on demand; kernel and submission
behavior remain as measured below. The
[wrapper comparison](performance/lazy_locator_wrappers_sdf.md) records the final
A/gather/on-demand-view measurements separately.

## Call path

```text
GpuEventManager._setup_detectors
  -> construct detector bindings and GPUDetector instances
  -> construct GpuXtcBatchPool and stable handle -> output indices
  -> GPUDetector.configure_gather
     -> build canonical [stream column, handle index, type, rank] rows
     -> reserve/upload table once and record setup readiness

EventPool.submit (existing execution size/slot)
  -> GpuXtcBatchPool.parse
     -> walker + batched locator initialization/decoding
     -> shared locator-ready event, combined backing and explicit capacity
     -> retain combined descriptor; no per-handle Python wrapper loop
  -> GpuEventDgrams.from_batch (existing event/stream mapping)
  -> GPUDetector.process_batch
     -> select existing detector events and allocate/reuse output slots
     -> _GatherMap.prepare
        -> fill pinned [selected event, required stream] dgram-row map
        -> use -1 for missing streams; reject multiple input owners
        -> one H2D upload; preserve backing until execution-slot retirement
     -> _CanonicalGatherPlan.gather
        -> establish setup dependency once for each consuming stream
        -> same producer stream: no extra locator wait
        -> other stream: one shared locator-ready wait
        -> one gather_canonical_u16/f32 launch for all canonical output rows
     -> existing per-event calibration / missing-row cleanup / EventContext
  -> existing result-ready record and input/result leases

begin_retire_next -> expose results / register consumers -> finish_retire_next
  -> wait for terminal consumers, permit output and row-map reuse
```

## Kernel and compatibility

A flat block grid spans `(event, canonical segment, pixel tile)`. Canonical
routing chooses the stream-map column and stable field index. Each row reads
its locator at `((handle_index * allocated_capacity) + input_dgram_row) * 11`.
A smaller tail therefore retains the correct handle stride.

The new kernel preserves the prior gather's FOUND/type/rank/exact byte-count
and overflow-safe raw-buffer bounds checks. It intentionally preserves the
old shape acceptance policy rather than imposing new dimension checks. It
copies every valid pixel and writes every absent/rejected pixel as zero. One
thread writes each presence byte, avoiding races between zeroing and copying.
Post-calibration missing-row cleanup remains necessary: calibrated zero raw
pixels need not be zero. Float32 passthrough copies directly into calib output.

All device offsets stay on the GPU. Host scheduling reads only existing read
metadata. The original per-field gather helper remains as the test/reference
path; production no longer calls it or `dgram.locate()` per event/segment.
Public field lookup remains compatible: configured handles create/cache a view
on first access, sharing the already-recorded ready event; unconfigured handles
retain their single-field decoder. The general field-view API is unchanged.

## Ownership and memory

- Fixed canonical table: 32 bytes per canonical segment; uploaded once and
  charged to the shared GPU budget, reported under detector routing memory.
- Row map: eight bytes per selected event per required physical stream. Its
  device buffer and equally sized pinned host buffer are reused per execution
  slot. GPU map bytes appear in `raw_slots`; pinned bytes are host memory.
- Map growth reserves the full replacement while old storage is live. Device
  or pinned allocation failure releases the reservation and retains old buffers.
- The combined-locator descriptor retains its parsed batch, raw bytes, backing,
  ready event, and explicit capacity while being submitted. The EventPool owns
  that batch until retirement. Maps do not cache parsed batches across reuse,
  which would retain old parser storage during the next allocation growth.
- Existing output/presence allocations and consumer leases remain authoritative.
  Gather completion is ordered before calibration and the result-ready event.
- Admission estimates now include presence bytes and row-map bytes and count
  the actual dense canonical output shape. The existing constants/geometry
  accounting limitations at this B-era base are not changed here.

For 20 events, 32 Jungfrau segments, and five streams, new storage is a
1,024-byte fixed plan, an 800-byte device map per occupied slot, and an
800-byte pinned host map per slot. Gather submission becomes one kernel instead
of 640, and explicit target/presence clears disappear because gathering fully
writes both. Calibration and missing-row cleanup remain 20 kernels each.

## Scope of bulk integration

This implementation explicitly accepts one parsed owner per execution. It does
not silently assume that resident and transient windows share a raw base. The
later merge must add owner indices and independent bases/capacities/local rows
as described in the plan. Locate once per input window; gather once per detector
execution. Resident input can be reused across many gather submissions.
The future input owner must retain `configured_locations().ready` explicitly;
an empty per-handle wrapper cache does not mean location has completed.

## Validation

Final on-demand-view validation: **206 unit/GPU integration tests passed**, six
slow cases excluded, job **38676735**. The A/gather/on-demand-view performance
allocation **38676923** also passed 33 locator/gather tests, three CPU-reference
preflights, and all 18 timing-sample audits. The earlier gather implementation
and its broader acceptance evidence are recorded below; they are separate runs.

The first A100 correctness run, job `38559897` on sdfampere024, passed 203 tests
with six slow real-data cases deselected. This includes 16 initial new gather
cases, with exact raw/calib comparisons against the old per-segment helper,
independent synthetic raw expectations, float32 passthrough, missing streams,
reused Names IDs, reordered canonical IDs, multiple segments in one stream,
empty/tail/growing batches, malformed locator rejection, cross-stream readiness,
no per-handle lookup, one gather launch per nonempty execution, and map growth
rollback/accounting. The additional actual EventPool test with a delayed consumer passed in job
`38560120`. The main suite passed 234 tests (41 skipped, eight deselected), and
all four longer MPI tests passed. Two earlier lifetime-test attempts failed
because the new test omitted the empty RawKernel argument tuple and used
`query()` instead of the installed CuPy `Event.done` property; correcting these
test-only issues required no production edits. All six slow real-data pixel-exact acceptance cases also passed in job
`38560120` (sdfampere014), covering tails, two execution slots, automatic D2H
chunk sizes, and hybrid routing. Compute Sanitizer job `38561752` passed all 17 gather cases with zero reported
errors. The first benchmark allocation was interrupted by cudaErrorContained
in the frozen B+ control. Job `38561649` completed on sdfampere004 with the
same builds: four clean pairs give medians 37.275 → 28.578 s (23.3% lower
elapsed time). Both provenance and trace audits pass: gather kernels fall
640 → 1 per execution, total kernels 683 → 44, and 501 additional H2D
copies total 401,024 bytes over the run. See the
[performance report](performance/batched_canonical_gather_sdf.md) for all
repetitions, host scopes, memory, and profiler collection warnings.

The warm comparison freezes B+ in
`validation/batched-gather-20260918/install_baseline`, verifies its production
sources against `b0c9c3c02`, and compares it to the then-current gather installation
on one allocation. Harness instrumentation has seven AST-preservation/timing
tests. Clean throughput, host timings, and Nsight counts are separate samples.

Source review anchors (paths under `psana/psana/gpu`):

| File | Change |
| --- | --- |
| `gpu_detector.py` | `_canonical_gather_table`, `_GatherMap`, `_CanonicalGatherPlan`, one gather before the calibration loop, and the two typed canonical kernels. |
| `gpu_events.py` | Configure detector gather plans after constructing the parser pool; report routing, pinned maps, and aggregate detector high-water values. |
| `gpudgram/batch.py` | Build stable handle indices once and share them with parsed inputs. |
| `gpudgram/parser.py` | Expose combined locations with owner/capacity/readiness; create and cache configured views only on access. |
| `tests/gpu/unit/test_core.py` | Check dense output, presence, map estimates, pinned reporting, and multi-detector high-water aggregation. |
| `tests/gpu/integration/test_batched_locators.py` | Check on-demand wrapper construction, caching, backing strides, shared readiness, and decoder fallback. |
| `tests/gpu/integration/test_batched_gather.py` | Synthetic equivalence, rejection, reuse, dependencies, allocation rollback, and actual delayed-consumer retirement. |

The CPU loop over execution events remains for calibration and result objects;
map construction also iterates events and required streams. There is no longer
a Python loop over every segment submitting field access, waits, and copies.
Performance acceptance targets one BD/GPU at depth one; multi-BD contention,
true GDS, and throughput changes with deeper overlap remain unmeasured.
The internal gather plan requires eager configured handles. Unsupported owners
or a different handle layout fail explicitly; they do not silently select an
incorrect locator base. The general lazy locator API remains available.

## Separate geometry probe (not a production change)

Job `38560573` compared three gather launch/index geometries on a synthetic
20-event, 32-segment, 512-by-1024 uint16 workload on an A100. All three produced
exact raw pixels and presence flags. Eight GPU-event samples per geometry gave
approximately 4.119 ms for the production flat/64-bit indexing, 3.070 ms for
flat/32-bit indexing, and 2.746 ms for a direct three-dimensional grid. This
probe excludes XTC parsing, reads, calibration, event delivery, and host scopes.

Production remains the flat grid, which avoids CUDA's smaller y/z grid limits.
A later geometry change would need explicit large-grid fallback coverage and a
new matched full-pipeline comparison. Do not infer an end-to-end saving by
adding or subtracting these synthetic GPU durations from host timing scopes.
