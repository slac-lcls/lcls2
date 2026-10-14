# B++ bulk integration: Stage 3 lifecycle validation

2026-09-23, branch `codex/psana2-gpu-bulk-batched-integration`.
Stage 2 was reviewed and committed as **ac87a93b2**. Its runtime and tests
matched the validated manifest; all pre-commit checks passed. Review added the
explanation of the bulk-on memory peak and retained drain/trim behavior as a
Stage 4 performance measurement item. No additional runtime fix was needed.

## Stage 3 changes

Production code is unchanged. New device tests extend the integrated parser,
canonical gather, and ownership validation:

| Coverage | Checks |
|---|---|
| Resident input with `max_events=203` | Exact delivered timestamp prefix; all input windows retired; no pending reads, execution slots, input holds, or admission holds |
| Generator close during delivery | Close while resident and transient input uses exist; repeated manager close is safe; saved event facades reject input access |
| Read failure after resident input | Real KvikIO I/O completes, then a transient future reports an injected error; production failure handling drains and closes resources |
| Failure after gather submission | Raise after a real canonical gather launch; execution ownership survives until cleanup drains the queued work |
| Automatic D2H | Repeat all four stop/failure cases with chunk size seven, as well as D2H disabled |
| BeginStep and EndRun across batches | Verify old input owners retire before calibration refresh; new GPU constants produce exact calibrated pixels in a 203-event tail; repeated finish emits nothing |
| Multi-owner growth, uint16 and float32 | Execute 1, 4, then 2 events while retaining old gather-map and output aliases; reject a reservation one byte below the required growth; accept the exact reservation; account for old backing after trim until aliases are dropped |
| Pending work at transitions | BeginStep/EndRun must join an external CUDA consumer and release both input owners before computing new constants or dispatching the transition |

The mixed-rate fixture was extracted from `test_gpu_residency_device.py` for
reuse; its original two acceptance cases remain in the focused suite. It uses
real XTC bytes and CPU reference pixels, the production reader/parser/detector,
1,000 fast events and ten slow events, and a budget that admits one resident
fast input plus five transient slow executions. Multi-owner tests use the
existing three-segment synthetic detector with independent input bases,
unequal locator capacities, and missing streams.

## Results

Artifacts: `validation/bulk-integration-stage3-20260923/`.
Runtime: the immutable Stage 2 install, with build hashes verified against
`validation/bulk-integration-stage2-20260923/build.json`. Test manifests and
source snapshots are saved separately. No native rebuild was required.

Job **38902294**, A100 `sdfampere001`, completed in **36 seconds**, exit `0:0`:

- **323 CPU tests passed**.
- **26 focused device tests passed**: eleven new lifecycle/growth cases plus
  fifteen existing residency, admission, multi-owner gather, and bounds cases.
- The **eleven new cases also passed CUDA memcheck**, zero errors, using
  `--show-backtrace device` as established in Stage 2.

Job **38902625**, also on `sdfampere001`, passed the **two additional
pending-transition cases**, both ordinarily and under CUDA memcheck with zero
errors. The initial attempt, job **38902417**, exposed a missing logging flag
in the minimal test fixture; only that fixture was corrected. Both attempts'
logs and source manifests are retained.

Stage 3 therefore passes **323 CPU tests, 28 distinct focused GPU tests, and
13 overlapping sanitizer cases with zero device errors**. No production
change was required. The new tests and fixture extraction are recorded in
the validation checkpoint; the validated runtime remains `ac87a93b2`.

These tests establish lifecycle and accounting behavior, not throughput.
KvikIO uses CPU fallback; true GDS is not covered. The unchanged runtime's five
JF cases and main/MPI acceptance remain the Stage 2 evidence. No additional
full-JF throughput or Nsight result is claimed by these lifecycle tests.
The subsequent [Stage 4 comparison](performance/bulk_batched_integration_sdf.md)
and [optimization checkpoint](bulk_batched_optimization_baseline.md) record
the completed performance baseline and remaining drain/trim/overlap work.
