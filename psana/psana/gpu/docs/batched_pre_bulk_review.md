# Combined B+ and B+gather review before bulk integration

Reviewed 2026-09-18 and finalized 2026-09-20 on
`codex/psana2-gpu-batched-locators`: B+ commit `b0c9c3c02` and the canonical
gather/on-demand-view follow-up, against parser B `803a70011`.
Bulk source was inspected read-only at `codex/psana2-gpu-xtc-parser`
(`8f94e3c7b`). Bulk integration and calibration changes remain deferred.
Final cleanup changed documentation only; production sources still match the
validated on-demand-view build.

## Findings and cleanup applied

Two memory-reporting defects were fixed in `gpu_events.py`:

1. `_snapshot_memory` omitted the detector's new routing allocation and pinned
   row-map upload buffers. GPUDetector already included routing/device maps in
   its device accounting; the manager now reports routing explicitly and adds
   host maps to pinned memory. `GPUDetector.pinned_bytes()` keeps host storage
   out of device totals. For the measured 20-event JF workload, the omissions
   were 1,024 fixed device bytes, plus 800 pinned-host bytes per occupied slot.
2. `log_memory` previously took the largest individual detector value when
   updating category high-water marks. It now sums detectors within each
   snapshot before updating the high-water mark. The previous behavior
   understated multi-detector usage; this defect predates B+gather.

The raw-slot documentation now includes presence masks and device row maps.
An unused GPUDetector copy of `canonical_segment_rows` was removed. The binding
remains the authoritative owner of canonical row information.

No additional pixel-correctness defect was identified in the reviewed B+/gather
changes. This is not a claim that all old B-era budget/error handling is complete:
see the integration requirements below. The reporting fixes change neither
reservations nor kernel submission, so they have no claimed throughput benefit.

## Call path and ownership checked

```text
run setup
  -> validate/group field handles; upload Configure and handle tables
  -> create stable handle indices and canonical detector gather table

execution slot submission
  -> parser config dependency
  -> upload descriptor metadata
  -> walk XTC
  -> initialize combined locator backing
  -> decode only the dgram stream's configured handles on GPU
  -> record one configured-location ready event
  -> retain backing, handle indices, and ready event; create no Python wrappers
  -> build event/stream views once for detector consumers
  -> select detector events; prepare slot-owned pinned/device row map
  -> establish gather dependency; launch one canonical gather
  -> per-event calibration, missing-row cleanup, and result objects
  -> result-ready event and input/result leases

retirement
  -> wait for producer and registered consumers
  -> release parsed input references; permit map/output/slot reuse

optional individual field access
  -> locate(handle): cache hit -> return wrapper
  -> configured cache miss -> create/cache one view with shared ready event
  -> unconfigured cache miss -> existing single-field decoder
```

Review checked stream/Names isolation, duplicate/status propagation, explicit
allocated-capacity strides on tails, full output/presence writes, missing-row
zeroing after calibration, same/cross-stream readiness, map upload lifetime,
replacement reservation rollback, and retirement ownership. Combined locators
and row maps reserve the full replacement while old storage is live. Old
per-field gathering remains deliberately available for equivalence tests; it
is no longer a production detector call path and should not be removed casually.

## Optimization opportunities, in recommended order

Item 1 is implemented and measured as a follow-up. The remaining items are
proposals, not correctness findings or implemented speedups.

### 1. Create configured locator views on demand

At the time of the recorded benchmarks, `parser.py:_locate_configured`
constructed a CuPy slice and `DeviceFieldLocators` object for each eager handle
after every parse. The canonical gather consumes combined backing directly and
needs none of these objects. In the original measured workload this meant 192
wrappers per parse, or 96,000 over 500 executions.

The follow-up keeps GPU location eager and budgeted, but makes `locate(handle)`
construct/cache its view from `_configured_indices` and `_configured_backing`
on first access. This removes the N-handle Python loop from the canonical-only
path without changing GPU launches, the public locator result, or storage layout. It preserves
lazy decoding for handles outside the configured set. Validation covers repeated
access, empty inputs, tails, shared readiness, and no additional device allocation.

The [same-allocation follow-up](performance/lazy_locator_wrappers_sdf.md)
measured medians A 22.464 s, eager-wrapper gather 25.288 s, and on-demand-wrapper
gather 24.824 s. The observed 1.84% median reduction has overlapping ranges and
four wins in six rounds; it is not an isolated wrapper-construction measurement.
The earlier 0.578 s location host scope also includes other work.
**Bulk dependency:** `InputWindow` currently discovers eager readiness by
iterating `batch._locators`. With on-demand views, it must receive the explicit
configured-ready event even when that dictionary is empty. The descriptor
`batch.configured_locations().ready` exposes that event independently of views.
This bulk-branch adaptation remains required at integration; bulk commits have
not been changed here. See `validation/lazy-locators-20260920/README.md` for the
follow-up validation, separate from the preserved benchmark installations.

### 2. Batch calibration and fold in missing-row handling

`GPUDetector.process_batch` still submits 20 calibration and 20 cleanup kernels
per 20-event execution: 40 of the remaining 44 kernels. A subbatch calibration
kernel can reuse the single-event constants by indexing the pixel position
within each event and consult the presence mask before writing output.
That would make this stage one launch and the current dense-JF path five
kernels overall: walker, locator init, location, gather, calibration.

Do not pass a flattened multi-event array to the existing `fused_calib_gpu`:
its constants contract is `3 * raw_gpu.size`, which assumes one event.
Preserve gain-bit interpretation, numerical operations, canonical ordering,
BeginStep constant updates, exact zero for absent segments, raw results,
float32 passthrough, and result leases. In one saved host sample calibration
plus cleanup submission totals about 0.459 s, so fewer launches alone do not
promise a large full-pipeline saving. Keep this as a separate measured change;
it need not delay the input-owner merge.

### 3. Reduce gather index arithmetic

The current kernel uses flat 64-bit division/remainder to recover event,
segment, and pixel tile. Existing synthetic probe job 38560573 measured medians
of approximately 4.119 ms (current), 3.070 ms (32-bit indexing prototype), and
2.746 ms (3D grid prototype), with matching synthetic pixels/presence.
This is preliminary isolated GPU evidence, not an end-to-end comparison.

A guarded faster index path is worth testing independently. Retain 64-bit
addresses/offsets, validate integer products, and cover a fallback for CUDA's
y/z grid limits and unusually large event counts. Do not apply the prototype
by string substitution to production or combine it with the owner-layout merge.

### 4. Reduce redundant dependencies and transient host metadata

- `GpuXtcBatchPool.parse` waits on the same immutable config-ready event on
  every parse. Once per consuming stream should suffice if stream identity and
  lifetime are retained, as the gather plan already does. This removes about
  500 waits in the current test, a much smaller target than the removed
  per-field waits.
- The walker event is dependency-only but currently created with timing
  enabled. A timing-disabled event is a small follow-up cleanup; retain it for
  lazy field lookup and bulk input readiness.
- `build_dgram_records` allocates/fills a host table each parse. A reusable,
  pinned, input-owner buffer could reduce allocation/staging work. Its lifetime
  should be designed with bulk input windows rather than another temporary
  execution-slot convention.
- Event/stream row-map preparation can be shared where detector selections
  actually agree. Its entire gather submission scope was only 0.071 s in a
  saved 10,000-event sample, so it is lower priority than examining calibration.

### 5. Consider stream-local locator storage for resident inputs

The current rectangular backing is `all_handles * all_dgrams * 88` bytes;
GPU decoding is stream-grouped, but storage and initialization are not compact.
With 192 handles, five streams and 20 events in every stream, the rectangle is
1,689,600 bytes; storing only each handle's own stream rows would be 337,920
bytes. At 1,000 events those become 84,480,000 and 16,896,000 bytes.
These are derived layout sizes, not measured speedups, and exclude other tables.

Compaction would help metadata footprint and resident-input capacity, but changes
row indexing, public field-view representation, capacity strides, and admission
accounting. Preserve the validated rectangular layout for the first bulk merge;
revisit compaction only after correctness and residency measurements justify it.
Likewise, narrowing the eager handle set requires a policy for arbitrary public
field access and its memory budget; it is not a free removal of unused fields.

## Bulk integration requirements

These are required adaptations, not optional performance polish:

1. Keep parsing/location at **input-window** lifetime and gathering at
   **execution** lifetime. Extend the gather map with owner identity plus each
   owner's raw base, locator base, capacity and local row. A detector can span
   resident and transient owners; the current one-owner check is intentional.
2. Make ready dependencies explicit and deduplicate shared events in
   `InputWindow._ready`, preserving distinct lazy and terminal-consumer events.
   Do not infer readiness solely from instantiated locator wrappers.
3. Update parser `allocation_requirements` for the combined eager backing and
   separate lazy buffers. Update detector `allocation_requirements` and
   `trim_slot_buffers` for device maps and pinned upload sources. Retain fixed
   gather tables for their run lifetime and report their bytes.
4. Preserve the bulk branch's full-replacement growth accounting and
   `parse_window` failure drain/ownership retention. Old B-era metadata/output
   growth on this branch still reserves only the delta, and EventPool submit
   does not publish partially submitted work as an occupied slot on failure.
   Do not overwrite the bulk fixes with the older methods during merge.
5. Test no/mixed/all residency, detectors spanning owners, different owner
   capacities/local indices, resident reuse, tails, slow consumers, BeginStep,
   allocation failure, failure after asynchronous upload, and safe trimming.

The bulk branch was inspected without modification. This review does not claim
that multi-owner gathering already works or that merging text alone is enough.

## Validation and evidence preservation

The reporting cleanup passed **168 CPU unit tests**, including a new test for
two-detector totals, routing, pinned maps, shrinking snapshots, and logged
high-water values. At that review, parser source and gather-map, gather-plan,
and kernel ASTs matched the frozen measured build; those reporting-only edits
needed no new GPU timing or pixel run.

Tests used an isolated copied install in
`validation/pre-bulk-review-20260918/install`. An initial source-tree import
attempt failed collection because native extensions are installed separately;
the isolated prefix resolves that without altering recorded installations.
The earlier benchmark prefixes and scripts remain unchanged and auditable.
The test log and reproduction command are documented in that directory.

The on-demand-view follow-up passed **206 unit/GPU integration tests** (six
slow cases excluded) in job **38676735**, including cache identity, zero unused
wrappers, empty/tail/growing inputs, cross-stream readiness, unchanged fallback,
and canonical gather equivalence. All production GPU modules and modified
integration tests match the SHA256 manifest in
`validation/lazy-locators-20260920/sources.json` at final review.
Job **38676923** passed another 33 locator/gather tests, three CPU-reference
preflights, and the audit of all 18 clean performance samples. Earlier full
pixel, main-suite, MPI, and sanitizer results are retained in the
[gather review](batched_canonical_gather_review.md). No code changed after these
checks; final cleanup updates the review/call-path documentation and marks
historical status snapshots. `git diff --check` passes.

Keep old benchmark harness snapshots intact: their paths and hashes are part
of the evidence. For future experiments, a shared controller plus an explicit
variant manifest would avoid more copies of bench/staging/audit scripts.
Do not rewrite the old scripts merely to reduce duplication.

No new correctness blocker was identified in the final combined review. The
larger optimizations above have not been applied; the explicit bulk integration
requirements remain the next review boundary.
