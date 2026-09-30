# B++ bulk integration: Stage 1

2026-09-23. Branch: `codex/psana2-gpu-bulk-batched-integration`.
Ownership checkpoint: `786cd16ca`. Incoming B++ history: `4c3cdf5a7`.
The normal merge is resolved and staged without a commit; both parents are
preserved. Stage 1 acceptance is complete.

## Scope and implementation

Stage 1 integrates stream-grouped field location, canonical single-owner gather,
and lazy locator wrappers with the D+ ownership implementation. Acceptance uses
Jungfrau with `gpu_bulk_read=False`. Multi-owner canonical gathering remains
Stage 2 work; this branch is not yet accepted for bulk-on use.

- Parser slots allocate one rectangular configured-locator buffer using
  `owned_empty`. Fallback locators retain their separate owned allocations.
  Admission reserves the full rounded replacement, and cache reporting includes
  physical capacity. Escaped aliases keep old generations charged.
- Stream-grouped handle tables and fixed canonical routing tables use the
  existing setup upload helper, including synchronization of their setup stream
  and quarantine on unproven completion.
- Execution row maps use owned device allocations. Pinned upload sources remain
  slot-owned and are reported separately from device memory. Trimming drops map
  caches only after the caller retires the execution/result leases.
- Input windows explicitly retain the configured-location ready event even
  when no individual locator wrappers exist. This is necessary for single-owner
  execution too: retirement must wait for all submitted location work.
- Combined-location descriptors follow the parsed owner's access lifetime.
  Retirement detaches combined backing and scheduling tables; a saved descriptor
  rejects storage access and does not independently retain those allocations.
- Memory snapshots include routing in cached/owned totals and combine pinned
  row-map storage with D2H host storage. Detector high-water totals preserve D+
  aggregation and allocation inventories.

The calibration algorithm, rectangular locator layout, input residency policy,
and unconditional bulk cache-trimming policy retain their existing behavior.
D+'s result leases, failure handling, and core iterator changes are preserved.

## Validation

Artifacts: `validation/bulk-integration-stage1-20260923/`.
The copied install retains the reviewed native binaries; `build.json` records
its runtime hashes. Jobs record script/test hashes and the CPU-reference source.

- CPU unit suite: **322 passed**.
- A100 device suite: **51 passed**, job **38899563**.
- Jungfrau retained-facade, partial-tail/D2H, and tight-budget cases: **all passed**,
  job **38899563**, with **99 allocation-identity/capacity checkpoints**.
- Main psana suite: **389 passed, 61 skipped, 10 deselected**; MPI byhand:
  **4 passed**, job **38899523**. The suites overlap; do not add their counts.

GPU/JF job **38899563** completed on `sdfampere001` in **3m07s**, exit `0:0`.
Core/MPI job **38899523** completed on `sdfmilan002` in **4m05s**, exit `0:0`.
Runtime source matches the frozen install; Python 3.9 AST and diff whitespace
checks pass. No native code changed.

| Bulk-off JF case | Events | Batch/depth | Budget GiB | D2H chunk | Peak live MiB | Loop-end live MiB |
|---|---:|---|---:|---:|---:|---:|
| All facades retained | 1,003 | 20/1 | 8 | 0 | 3,201.804 | 0 |
| Partial tail, third/last facades retained | 1,003 | 13/2 | 8 | 7 | 3,970.328 | 0 |
| Tight budget, third/last facades retained | 1,003 | 20/1 | 4 | 0 | 3,201.804 | 0 |

Each case matched all 1,003 timestamps and 15 sampled raw/calibrated image pairs
against the independent CPU reference. Non-D2H cases additionally checked direct
parsed-field CPU images. Loop-end zero-live checks precede forced GC or diagnostic
synchronization, with facades still saved. The input is run 387 of
`mfx101210926`, JF streams 5–9, one BD/A100 using KvikIO CPU fallback.

The device suite covers grouped/single-field equivalence, canonical gather
equivalence, absent/malformed fields, tails and reuse, same/cross-stream
dependencies, delayed consumers, exact admission growth, and allocation aliases.
New CPU cases verify configured readiness without wrappers and retirement of a
saved combined-location descriptor while an escaped array stays charged.

The initial CPU run exposed two old test assumptions: an incomplete detector
fixture and an assertion that omitted a still-retained generation. GPU job
**38899522** passed 47 tests and exposed two fixture defects: synthetic descriptors
omitted timestamps required by input windows, and restoring a bound method as
an instance attribute created a self-reference cycle. The corrected device
suite passes all 51 cases. No production accounting was weakened for these tests.

## Stage boundary

Stage 2 must extend the gather map and kernel to independent input owners,
including per-owner local rows, capacities, bounds, and readiness. The current
single-owner check is intentional. Performance comparison, allocation reuse,
residency-policy changes, hybrid detectors, multi-BD scaling, and true GDS are
outside this stage's acceptance. No throughput conclusion follows from these
instrumented correctness runs.
