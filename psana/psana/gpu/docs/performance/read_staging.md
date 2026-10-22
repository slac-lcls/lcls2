# JF and JF+feespec read/staging performance

Latest revision comparisons: October 16–17, 2026 (summarized below).
The topology scaling campaigns are from September 28, runtime `10df4c6e3`.
These measure JF read, GPU XTC parsing and dense raw gathering **without a
GpuTask or calibration**. A benchmark-only input adapter requests the gather.
Mixed runs also perform the existing per-event feespec `raw.hproj` GPU int64
sum. Thus these are pipeline-staging rates, not isolated storage bandwidth.
The production no-task path does not automatically request that dense gather.

## Configuration and timing definition

Run 387, 10,000 events, full 32-panel JF across five streams; mixed runs add
feespec. One EB, A100 GPUs, batch 20, execution depth 1, eight KvikIO workers per
BD, 1 MiB task/group targets, automatic per-GPU peer budgets and **CPU fallback**.
Each topology uses bulk off/on and cold/warm cache preparation on private
node-local staged prefixes. Two fresh-process repetitions reverse ordering.

Rates below are **10,000 / median event-loop seconds**, including lazy work
inside the loop. Explicit DataSource/run setup is separate. Staging, cache
preparation, imports and MPI startup are excluded from loop time. Both sample
loop durations, setup, cache checks, GPU placement and resource counters are
retained in [the evidence](evidence/read_staging.json). Two repetitions support
these descriptive medians, not narrow confidence intervals.

## October revision comparisons

The pre/post review-stack comparison showed **no consistent staging slowdown**.
The September/post-stack comparison showed a warm, bulk-off slowdown, but the
code investigation found no changed processing implementation that explains it.
These are separate experiments; the latter is not evidence that splitting and
merging the PR stack changed performance.

Both campaigns used JF-only dense raw-input staging: one GPU/four BDs, 10,000
events per timed sample, batch 20, execution depth 1, eight KvikIO workers and
1 MiB tasks. Both used official KvikIO 26.10, CuPy 14.0.1, Python 3.11 and CPU
fallback. Within each campaign, A and B ran in adjacent matched pairs on one
exclusive A100 node, with reversed ordering on alternate repetitions. Cache
state, input hashes, timestamps, read counts and correctness preflights gated
acceptance. Setup and staging time are excluded from the event-loop timing.

Tables report rates in events/s and the median paired **B/A loop-time change**:
positive means B was slower. Rates use each revision's median loop duration;
the median paired percentage need not equal the ratio of these rates.

### Pre-stack versus post-stack

Reference: Amanda's revert/review stack, PRs
[#136](https://github.com/slac-lcls/lcls2/pull/136) through
[#151](https://github.com/slac-lcls/lcls2/pull/151), merged in order on
October 14, 2026. The final merge is `410ebba1c`.

A = `9b32dda19b46807f7a4db62fb2351ba87b359e47`; B =
`410ebba1ce4d8b2a9d8d7fab56f6dc1fa26a6ef8`. Their complete repository trees are
identical: `65b8d582a61a96cafd2e91f03c52b5c910e5e455`.
Each revision was built independently in separate source/build/install
directories. Clean A → B → A activation and per-rank import checks verified
the selected Python and native libraries.

Job **41173787**, `sdfampere033`, completed October 17 in **43m00s**, exit 0,
under `lcls:data` (Slurm: `lcls:data@ampere`). Four 200-event correctness
preflights and 16 timed samples passed: **two pairs per condition**, 160,800
events total. All 1,852 frozen artifact hashes passed at completion.

| Cache | Bulk | Pre-stack rate | Post-stack rate | Paired time change | Pair range |
| --- | --- | ---: | ---: | ---: | ---: |
| Cold | Off | 310.15 | 308.50 | +0.56% | −0.91% to +2.04% |
| Cold | On | 306.89 | 299.69 | +2.42% | +1.73% to +3.10% |
| Warm | Off | 490.43 | 496.37 | −1.17% | −3.25% to +0.90% |
| Warm | On | 493.78 | 509.27 | −2.82% | −7.87% to +2.22% |

Direction varies by condition. This supports staging performance preservation
after the stack merge, but two pairs per condition cannot exclude a small
regression or establish a speedup. With identical source, build/runtime
variability must be considered when interpreting the timing differences.

### September versus post-stack

A = `10df4c6e3cad16779797ad017b08690a06c707bb`; B = the same post-stack revision
above. Job **40882509**, `sdfampere012`, completed October 16 in **1h13m22s**,
exit 0. Four correctness preflights and 32 timed samples passed: **four pairs
per condition**, 320,800 events total. All 2,533 frozen-file hashes passed at
completion, with no recurrence of the earlier KvikIO failure. This campaign
used separate frozen Python runtimes with shared native build artifacts.

| Cache | Bulk | September rate | Post-stack rate | Paired time change | Pair range |
| --- | --- | ---: | ---: | ---: | ---: |
| Cold | Off | 391.35 | 391.32 | −0.57% | −7.94% to +5.68% |
| Cold | On | 398.01 | 385.84 | +3.67% | −1.15% to +7.79% |
| Warm | Off | 456.87 | 432.31 | +6.00% | +0.54% to +9.84% |
| Warm | On | 454.76 | 440.31 | +2.55% | −14.12% to +11.90% |

Warm, bulk-off was slower in B in all four pairs, with a median increase of
about 1.30 seconds per 10,000-event loop. Other conditions had mixed-sign
differences. Four pairs do not establish a cause or a precise regression size.

The endpoint code audit found only two changed core Python modules:
`gpu_stream.py` has a docstring edit; `ds_base.py` changes the GPU batch-size
default to 20. Both variants explicitly use `batch_size=20`, and the default
selection executes before timing. Read, gather, parser, MPI scheduling,
ownership and native implementations are unchanged. All 593 core Python files
per frozen runtime match their commits, and all 48 benchmark files match
between A and B. Repository benchmark edits therefore did not change the work
between these two measured variants.

The slower warm, bulk-off samples also accumulated more read-future wait time.
This suggests investigating read-completion/scheduling variability; it does
not identify its cause. No production-code fix is justified by this evidence.
If further investigation is needed, use a same-build A/B-label control, then
more balanced September/post-stack pairs with independent builds. The prepared
independent-build September campaign was canceled before running.

### Scope and saved evidence

Compare A/B changes within each experiment. Different nodes and build layouts
prevent treating the absolute rates across these campaigns as a controlled
comparison. Neither campaign measures calibration, external user kernels or
true GDS, and neither changes the default shared build environment.

[October comparison evidence](evidence/revision_comparisons_202610.json)
preserves exact revisions, paired measurements, summaries and code-audit
checks. Full logs and per-sample records remain under
`/sdf/scratch/users/m/monarin/gpu-validation/`:

- Pre/post stack: `prepost-stack-independent-kvikio2610-20261017-r1/job-41173787/`.
- September/post stack: `postmerge-ab-1gpu4bd-kvikio2610-20261015-r1/job-40882509/`.

### October 19–22 follow-up

The [issue159 historical audit](../../notes/issue159_cold_rate_history_20261022.md)
records later cold 4BD/1GPU rates: ~261 Hz before PR187 on October 19 and
~177 Hz for both Stage 3 and Stage 4 on October 22. Payload/reference hashes and
read counts match the campaigns here. Increased read-future waits dominate the
extra loop time. These different-node observations do not locate a causal
commit; matched incremental success must not be read as preservation of the
September absolute rates. A common-harness old/current cold comparison remains
needed to resolve the gap.

## Full JF

Job **39369005**, `sdfampere035`, completed in **3h07m01s**, exit 0.
22 preflights and 88 timed samples passed; 1,289 frozen file hashes verified.

| GPUs | BDs | Cold bulk off | Cold bulk on | Warm bulk off | Warm bulk on |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 221.58 | 211.13 | 437.29 | 382.33 |
| 1 | 2 | 314.45 | 310.28 | 494.16 | 458.29 |
| 1 | 4 | 341.64 | 351.64 | 484.13 | 489.12 |
| 1 | 6 | 355.81 | 355.22 | 456.77 | 463.21 |
| 1 | 8 | 357.91 | 358.70 | 432.26 | 433.81 |
| 2 | 2 | 320.36 | 313.20 | 729.66 | 606.74 |
| 2 | 4 | 354.06 | 353.18 | 689.48 | 658.26 |
| 2 | 6 | 352.20 | 345.57 | 715.47 | 781.78 |
| 4 | 4 | 353.86 | 356.99 | 1007.14 | 847.35 |
| 4 | 8 | 357.54 | 354.89 | 918.04 | 848.15 |
| 4 | 12 | 360.10 | 359.92 | 914.20 | 882.14 |

Peak warm median is **1007.14 events/s** at four GPUs/four BDs, bulk off.
Cold rates approach **360 events/s** across the largest configurations.
Increasing BDs does not monotonically improve warm throughput; bulk grouping
also does not consistently win for these large JF dgrams.

## Partial JF+feespec: one GPU, 1–4 BDs

Job **39369006**, `sdfampere033`, completed in **57m03s**, exit 0.
6 preflights and 24 timed samples passed; 1,289 frozen file hashes verified.
“Partial” denotes the smaller topology sweep, not a reduced JF panel count.

| GPUs | BDs | Cold bulk off | Cold bulk on | Warm bulk off | Warm bulk on |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 205.93 | 188.87 | 377.41 | 336.26 |
| 1 | 2 | 282.74 | 263.74 | 447.31 | 456.40 |
| 1 | 4 | 305.36 | 305.60 | 500.29 | 489.88 |

Peak warm median is **500.29 events/s** at four BDs, bulk off; the best cold
median is **305.60 events/s** at four BDs, bulk on. The mixed workload includes
feespec reduction and its validation; it is not JF-only I/O with another label.

## Correctness and provenance

Preflights checked three JF raw arrays against the independent CPU reference;
mixed preflights checked all 200 feespec arrays. Each timed mixed run validated
10,000 timestamp-associated sums. Timestamp identities, payload/read counts,
GPU placement, quotas and drained resources gated acceptance. No calibrated-pixel
comparison is claimed for this staging workload.

Frozen roots under `/sdf/scratch/users/m/monarin/gpu-validation/`:

- `jf-user-kernel-full-scale-20260928-r1`
- `jf-feespec-user-kernel-scale-20260928-r1`

The [JSON evidence](evidence/read_staging.json) preserves sample measurements,
accepted summaries, source/log hashes and scheduler results. Historical baseline
metadata inside it is provenance only. Earlier automatic-calibration reports
measure different work; their ratios are not matched regression results.
Maintained measurement harness: [jf_scaling](../../scripts/jf_scaling/README.md).

Use [user-kernel performance](user_kernels.md) for matched scheduling and the
latest calibration/integration scaling. Neither report establishes true-GDS
performance or a universally optimal bulk setting.

The mixed benchmark uses an explicit benchmark-only routing exception for the
shared feespec stream; co-resident CPU detectors are not consumed. Production
applications needing those consumers must use the documented hybrid routing
contract, including its duplicate I/O cost.
