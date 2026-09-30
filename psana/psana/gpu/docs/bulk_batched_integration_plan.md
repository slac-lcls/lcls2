# Bulk input windows with batched location and canonical gathering

Proposed 2026-09-20; planning only. No integration or policy change has been
implemented by this document.

## Baselines and evidence

Preserve both published histories:

- Batched locators, canonical gathering, and lazy wrappers:
  `4c3cdf5a791f5c3801b03154bf696579d0b8be92`.
- Bulk input windows and admission:
  `8f94e3c7bc4643309d022cb5c1dd7e85805d3ac4`,
  branch `codex/psana2-gpu-xtc-parser`.

The **Design GPU XTC parser** task's completed bulk acceptance report is at
`/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline/psana/psana/gpu/docs/performance/parser_bulk_acceptance_sdf.md`.
It is a local validation artifact, not part of the bulk branch commit; preserve
it and its underlying `validation/perf-acceptance-20260916/` artifacts explicitly.
Its 96 timing samples and 16 diagnostics passed correctness, but **performance
acceptance failed**. Relevant findings:

| Workload | Observation | Integration implication |
| --- | --- | --- |
| Run 387, 10,000 events, batch 20, depth 1, 8 GiB | Legacy A 361.3 events/s warm; bulk D 214.3, about 41% less throughput. Bulk-off C to D cold throughput fell 29.7–38.0% across paired repetitions. | A-to-D includes the original parser/gather overhead; it is not a measurement of bulk alone. Compare integrated bulk off/on directly. |
| Same primary workload, full diagnostics | Reads fell from 50,000 to 2,885 while execution size stayed 20. | Fewer read requests alone did not improve throughput. Investigate drain, trim, allocation, resident-read waits, and overlap. |
| Run 51, JF + epix, batch 1000, depth 2, 8 GiB | Residency changed 39 executions per batch into 500. Warm throughput was 238.6 off versus 87.4 on on sdfampere027, a 63.4% decrease. | Preserve useful execution width when choosing resident streams. |
| Primary full diagnostics | Bulk ledger peak 3,201.8 MiB versus sampled CuPy **used** peak 8,325.5 MiB. | Audit live ownership and accounting before relying on the quota or scaling BD ranks. |
| Mixed batch-1000 full diagnostics | Bulk ledger 7,197.9 MiB versus sampled CuPy used 13,756.1 MiB. | This is not explained solely by unused allocator cache. |

The old primary results span two allocations on the same node/GPU; CPU/NUMA
placement was not fixed. The mixed report separates nodes and uses within-
allocation pairs. These were KvikIO compatibility-mode reads, not true GDS.
Run 51 has same-rate detectors; it does not establish sparse/mixed-rate behavior.

The [latest optimized comparison](performance/lazy_locator_wrappers_sdf.md)
is a different allocation: A 22.464 s, B+gather 25.288 s, and zero-wrapper
B+gather 24.824 s for 10,000 events. It validates the optimized starting point,
not bulk performance; do not combine its absolute times with the old bulk rates.

## Proposed sequence

### 1. Audit ownership and fix accounting on the bulk base

Use an isolated integration branch/worktree starting at the bulk head. First
reproduce the ledger/live-allocation discrepancy with bounded diagnostics.
Trace allocations through reader/parser slots, input windows, stream dgram
views, locator views, detector outputs, and retained event facades.

`InputWindow._try_retire` currently retains `self.batch` after invoking its
release callback. Trimming drops cached references and releases ledger charges
even if another view still owns the allocation. These are concrete candidates,
not proof of the entire measured gap. Distinguish safe reuse after CUDA
completion from physical deallocation and from dropping the budget charge.

Define and test retirement semantics: retired facades must reject storage
access; internal heavy references can be detached only after all uses and CUDA
consumers finish. Retained valid views must keep their allocations accounted
for. Include repeated windows, retained facades, delayed consumers, growth,
trimming, and exceptional submission. Do not hide retention by forcing global
synchronization or freeing the allocator pool in the normal path.

Land any resulting lifetime/accounting fix as a separate integration commit.
This remains a gate even if reproducing it points to a different cause.

### 2. Merge the optimized history and adapt the owner interfaces

Use a normal merge of the batched branch into the integration branch; preserve
both histories. Resolve behavior explicitly, rather than replacing bulk files
with the older parser/detector/EventPool implementations. Preserve full-size
replacement reservations, failure drains, and ownership of partially submitted
work from the bulk branch.

The required call path is:

```text
Configure/binding lifetime:
    build/upload stream-grouped config tables and canonical gather plan once
Input-window lifetime (resident or transient):
    read -> walk XTC -> one locator init + one grouped locate -> shared ready
Execution subbatch:
    build (event, stream) -> (owner index, owner-local row)
    wait on each distinct required owner dependency
    one canonical gather per supported detector across all required owners
    existing per-event calibration/cleanup -> register terminal uses
Retirement:
    last planned/live use + CUDA completion -> reuse/release with accounting
```

Keep the rectangular locator layout initially. Extend the gather input with a
small owner table containing raw/locator bases, byte bounds, locator capacity,
and dgram count. Never assume two owners share row numbering or capacity.
Retain owners and pinned upload buffers for the entire asynchronous use.
Resident data is parsed/located once per input window, then reused by later
executions. A multi-owner event currently has `batch=None`; route through its
stream owners instead of dereferencing that convenience field.

Take shared configured readiness directly from the batch, even when the lazy
wrapper cache is empty. Deduplicate shared events while preserving distinct
fallback-locator and terminal-consumer dependencies. The canonical path should
still create zero per-handle wrappers, with no CPU loop over all configured
handles per execution. Public `locate(handle)` retains its lazy API.

Update parser and detector allocation requirements and trim paths for combined
locator backing, fallback buffers, owner/row maps, and fixed gather tables.
Keep pinned host lifetime accounting distinct from device-budget accounting.
Cache Configure-derived detector stream membership at binding setup; the old
report measured repeated reconstruction in the hot path.

First validate this merged implementation with `gpu_bulk_read=False`, then
exercise residency for correctness. Keep the opt-out available throughout.

### 3. Change residency admission separately

The old planner admits whole streams while reserving only minimum one-event
progress per slot. It can also reduce depth to admit the first resident stream.
This is insufficient to preserve GPU execution efficiency.

A CPU calculation using the report's recorded run-51 sizes, parser estimate,
fixed costs, 8-GiB budget and depth 2 gives:

| Resident streams | Resident bytes | Full execution width | Executions / 1000 events |
| --- | ---: | ---: | ---: |
| None | 0 | 26 | 39 |
| epix only | 1,099,176,000 | 22 | 46 |
| epix + 5-MiB JF stream | 6,360,242,000 | 2 | 500 |

The middle row is a **model prediction**, not a measured performance result.
The other two plans were confirmed by the old diagnostics. Recompute costs
after integration, including actual allocation growth and new tables.

Start conservatively: establish the nonresident execution plan/depth, reserve
its useful working set and growth allowance, and admit whole resident streams
only in the remaining space. This may reject all full-batch residents in the
tight case; that is preferable to silently accepting a 26-to-2 collapse.
Do not assume epix-only preserves width, or that 26 is a permanent target.
Log rejected candidates and their predicted width/depth costs.

If whole-batch residency cannot provide a benefit under this guard, evaluate
bounded resident windows or a measured width/depth tradeoff in a later change.
Do not add another policy knob before measuring the tradeoff. Treat selective
trimming, buffer reuse, and asynchronous resident-read overlap as separate
follow-ups driven by a timeline: the current resident path drains executions,
trims caches, and waits for its read before continuing.

### 4. Validate and benchmark before merging back

Correctness must cover no/mixed/all residency, a canonical detector spanning
owners, unequal capacities and local rows, missing data, tails, resident reuse,
BeginStep, delayed consumers, growth/allocation failures, and failure after
asynchronous upload. Compare canonical pixels against CPU references. Generic
epix field access does not imply a new epix canonical calibration implementation.

In one allocation, compare frozen zero-wrapper B+gather, integrated bulk off,
and integrated bulk on. Include A as a reference, but use off/on to isolate
bulk behavior. Keep the intermediate integration/policy revisions as explicit
variants when measuring the admission change. Use balanced repeated runs,
fixed CPU/NUMA placement, identical GPU/process settings, and verified warm/cold
cache conditions; separate instrumented diagnostics from clean timing samples.

Minimum workloads: run 387 batch 20/depth 1/8 GiB (unchanged-width regression),
run 51 batch 1000/depth 2/8 GiB (admission collapse), plus batch 10 and 1-GiB
controls. Record:

- Actual execution widths/counts, active depth, residents, read requests,
  payload/read amplification, and input-window counts.
- One walk/init/grouped-location sequence per parsed input window; zero eager
  wrappers on canonical paths; one gather per detector execution; dependency
  counts governed by distinct owners, not configured handle count.
- Host submission/read-wait/drain/trim times and CUDA timelines. Do not sum
  overlapping host scopes or compare only `_submit_gpu` after work moves into
  resident admission.
- Ledger and held bytes, CuPy used versus cached bytes, device observations,
  replacement peaks, and retained objects at retirement and loop end.

Acceptance: no repeatable integrated bulk-off regression beyond measurement
noise; correct pixels/lifetimes; explained and bounded live allocations; no
unjustified execution/depth collapse; and repeatable on/off benefit for the
intended workload before recommending bulk enablement. Fewer reads alone is
not acceptance. Repeat multi-BD scaling only after memory ownership is sound.
True GDS, sparse/mixed-rate streams, and further calibration batching remain
separate validation/optimization work.

Merge the reviewed integration branch back into the bulk branch only after
these gates, preserving the two original benchmark heads and their artifacts.
