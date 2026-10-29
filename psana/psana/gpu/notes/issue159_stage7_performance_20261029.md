# Issue 159 Stage 7: performance evidence reconciliation

Status: evidence checks complete; independent review pending. This step changes
notes and an offline evidence verifier only. No GPU job, A/A comparison, new
runtime implementation, or performance threshold is introduced.

The [reviewed Stage 5–8 plan](issue159_stage5_revised_plan_20261029.md)
allows reuse when runtime is unchanged and no new concern invalidates the
measurements. That condition holds for the Stage 7 base. Retain the accepted
small cold-screen slowdown and the disclosed field-access costs. This is not
a claim of zero overhead or unchanged performance on every workload.

## Candidate identity and what A/B mean

| Identity | Commit | Meaning |
|---|---|---|
| A | `8cf05e86a7511f84a24f82b9c9720bc2308c471d` | Unfixed target after finalized PR187/208 |
| B | `405f868ed380c78b6009da3001f4535cb9ae647b` | Reviewed cumulative alignment fix, including the empty-field correction |
| Stage 7 base | `9706dfeeb` | Personal-fork fix branch after Stage 5/6 and workflow cleanup |

The complete B-to-base tree diff contains only documentation, tests/fixtures,
and the field-retirement benchmark. Production sources, native code, build
configuration and dependency pins are unchanged. The verifier records every
changed path and checks all 37 production GPU Python modules against B's
preserved measured installation by SHA-256. Thus this step reuses B's measured
runtime; it does not claim to have measured a fresh Stage 7 build or environment.
The preserved A/B builds and external dependencies remain the provenance for
the reused measurements.

A and B are **before/after the alignment fix on the same finalized dependency
stack**, not before/after the PR246 merge. Stage 5/6 added compatibility tests,
usage guidance and test-setup corrections without changing this runtime.
[Stage 6](issue159_stage6_correctness_20261029.md) records correctness coverage,
including the disposition of historical failures; older incomplete-suite
statuses do not replace that reconciliation.

## Accepted cumulative cold screen

[Full report](issue159_pr246_cold_20261028.md) and
[original evidence](evidence/issue159_pr246_cold_20261028.json): job **42255812**,
`sdfampere015`, same exclusive A100 node for both arms. A and B have separate
fresh native builds/installations, with matching external dependencies. Four
BDs share one GPU; batch size 20, depth 1, 6 GiB/BD, eight KvikIO workers,
1 MiB tasks/groups, compatibility mode ON. Workload: full 32-panel Jungfrau,
mfx101210926 run 387, five streams, read/parse/dense gather. It excludes user
GpuTask/calibration and per-event field accessor work.

Private input prefixes were staged once on node-local `/lscratch`. Before
**each** timed process, only those private files were flushed/evicted and each
measured prefix verified at most 1% resident. Maximum observed pre-residency
was **0.000492%**. Compilation warmups and pixel/address preflights were separate
processes. These are cold **file-cache** results, not cold CUDA compilation.
The benchmark-only reader wrapper reserved N+1 bytes identically in both arms
with zero shift, and preflights checked safe actual addresses for A. Timed
runs disabled the address-audit hook.

There were 16 timed processes, each 10,000 events: four balanced adjacent AB/BA
pairs per bulk mode, n=4 per arm/mode, **160,000 total timed events**. Each rate
is 10,000 divided by that run's slowest-BD loop time. Table entries are the
mean and **sample standard deviation (ddof=1)** across those four runs; rates
are averaged per-run rates, not 10,000 divided by the mean loop time.

| Bulk | A loop, s | B loop, s | A events/s | B events/s | Change in mean loop |
|---|---:|---:|---:|---:|---:|
| off | 34.662 ± 1.678 | 35.119 ± 0.907 | 289.00 ± 13.96 | 284.89 ± 7.54 | +1.32% |
| on | 34.698 ± 0.679 | 35.373 ± 0.844 | 288.29 ± 5.63 | 282.82 ± 6.82 | +1.95% |

| Bulk | Median paired loop change | Pair range | B slower |
|---|---:|---:|---:|
| off | +2.17% | -3.55% to +4.94% | 3/4 |
| on | +1.28% | +0.26% to +5.01% | 4/4 |

Mona already accepted these costs. Four pairs describe observed variation;
they do not establish a precise causal cost of the alignment branch or a
confidence bound. Read-wait counters include future completion/bookkeeping,
and averages across BD ranks cannot be subtracted from the slowest-rank
headline loop to infer kernel overhead. Do not pool different nodes, earlier
warm screens or the historical ~391 Hz run into this comparison.

Both arms used Python 3.11.14, OpenMPI 5.0.8, mpi4py 4.1.1, NumPy 2.3.5,
CuPy 14.0.1 and KvikIO/libkvikio 26.10.0. Original evidence records compiler,
CUDA, library hashes, actual imports and the separate native installations.
This CPU-fallback result is not a GDS throughput claim.

## Per-event fields: retain costs, keep sites separate

These are in-memory accessor/retirement microbenchmarks, not batched application
throughput. Scalar and 4 MiB fields use synthetic locators, excluding parser
and locator-transfer costs. Each access case has 20 warmups and 200 measured
calls per process. The tables summarize four process medians with their sample
SD; the percentage column is the **mean of four paired percentage changes**.
It is not the ratio of the two displayed means.

### SDF job 42263681

[Report](issue159_pr246_fields_20261028.md), `sdfampere016`, A100 40 GB,
same A/B revisions above. All four pairs were preserved.

| Aligned host access | A, µs | B, µs | Mean paired change |
|---|---:|---:|---:|
| scalar first | 40.585 ± 1.962 | 41.237 ± 1.936 | +1.87% |
| scalar cached | 12.575 ± 0.275 | 13.327 ± 0.154 | +6.04% |
| 4 MiB first | 40.533 ± 0.625 | 41.706 ± 0.664 | +2.93% |
| 4 MiB cached | 12.860 ± 0.386 | 13.471 ± 0.437 | +4.89% |

### Perlmutter job 60467642

[Report and provenance](issue159_pr246_perlmutter_fields_20261028.md),
`nid001024`, A100 40 GB. This uses the same A/B revisions but a different site,
MPI/compiler/driver environment. Do not pool its absolute timings with SDF.
Stage 7 verifies the retrieved original report's SHA-256; it does not newly
reprocess the remote raw samples.

| Aligned host access | A, µs | B, µs | Mean paired change |
|---|---:|---:|---:|
| scalar first | 38.523 ± 2.038 | 41.850 ± 1.984 | +8.79% |
| scalar cached | 12.199 ± 0.242 | 13.070 ± 0.206 | +7.18% |
| 4 MiB first | 38.756 ± 0.840 | 40.632 ± 0.404 | +4.87% |
| 4 MiB cached | 12.043 ± 0.221 | 13.008 ± 0.365 | +8.03% |

Both sites show a modest absolute host cost for aligned access; no percentage
here should be presented as an application event-rate regression. A's unaligned
typed access is unsafe and is not a performance baseline.

### Unaligned B and retirement

| Site / field | First host access, µs | Cached host access, µs | Isolated byte copy, µs |
|---|---:|---:|---:|
| SDF scalar | 73.811 ± 0.952 | 14.986 ± 0.158 | 2.179 ± 0.064 |
| SDF 4 MiB | 76.888 ± 1.086 | 14.960 ± 0.187 | 8.607 ± 0.091 |
| Perlmutter scalar | 71.338 ± 0.647 | 14.484 ± 0.247 | 2.603 ± 0.065 |
| Perlmutter 4 MiB | 74.266 ± 0.474 | 14.419 ± 0.234 | 8.484 ± 0.059 |

The aligned cache owns no copy; the unaligned cases retain respectively
512 bytes (allocation rounding) and 4,194,304 bytes. First-access host timing
and isolated CUDA copy timing measure different things; do not subtract them
to invent a full breakdown. CUDA stream elapsed intervals can include host
submission gaps.

The delayed-consumer retirement case exposes the accepted limitation: an
unaligned retained copy makes `EventPool.finish_retire_next()` wait roughly **7.10 ms** on SDF
(**7.098 ms** on Perlmutter), whereas aligned deferred retirement returns in
about 11–13 µs and pays the remaining wait during drain. This is the remaining
consumer wait, not a 7 ms image-copy kernel. The copy was materialized before
the timed retirement. The synthetic delay is 10 million clock ticks, not a
portable fixed-duration workload.

SDF's 3,840 retirement samples include 1,920 delayed samples, all pending at
entry. Original harness checks cover data, budget release and no early release;
Stage 7 independently recomputes their recorded timing and pending/completion/
release counts. Slot-wide copy accumulation and retirement waiting remain
tracked by [#256](https://github.com/slac-lcls/lcls2/issues/256). These microbenchmarks
do not establish sustained memory capacity under arbitrary event-loop use.

## Large unaligned Jungfrau coverage already performed

The accepted current cold comparison uses safe aligned data in both arms.
Unaligned dense gathering also has historical performance and device evidence:

- [Optimized Stage 2](issue159_stage2_20261019.md), job **41319573** on
  `sdfampere012`: independent old A `90c05965c`, B `7f29f0dc7`.
  Full 32-panel `uint16`, batch 20, 640 MiB useful payload; 256 warmup launches,
  eight samples of 20 launches per process. Process-order reversal produced
  A medians **4.0193 / 4.0191 ms**, aligned B **4.0850 / 4.0849 ms** (~+1.64%),
  and one-byte-shifted B **4.1252 / 4.1243 ms** (~+0.96–0.98% beyond aligned B).
  Stage 7 recalculates these medians from the committed kernel samples. Batch-1
  samples remain available too. This is kernel-only timing on older revisions.
- [Staged Stage 2 screen](issue159_short_screen_20261019.md), job **41337036**
  on `sdfampere030`: four BDs/one GPU, 10,000 events/sample, **warm** cache,
  two balanced pairs per comparison/bulk mode. A0 and B0 are aligned; B1 reads
  directly into a one-byte-shifted input view. All cases reserve N+1 bytes with
  the same accounting, without input repacking or an extra image copy.
  The two B1 audits each checked 320,000 odd-address rows and 50,000 even-sized/
  even-offset dgrams. Eight GPU-rank memcheck logs reported zero errors.
  Aligned bulk-on loop changes were +0.98%/+1.30%, accepted by Mona. Shifted
  B1/B0 timings were variable and do not establish a speedup.

These records address the original request to exercise a large unaligned
detector. They remain historical evidence, not a newly measured cumulative
cold B1/B0 result. The current cold screen measures cumulative aligned-data
cost after finalized PR187/208. The reviewed reuse policy does not require a
new shifted cold campaign when runtime is unchanged. Correctness reconciliation
and later sanitizer evidence are mapped separately in Stage 6.

## Reproducible reconciliation and limits

[Stage 7 evidence](evidence/issue159_stage7_20261029.json) records the exact
base, changed-path inventory, 37 module hashes, selected artifact hashes,
recomputed cold statistics, field comparisons and historical gather medians.
The [offline verifier](evidence/verify_issue159_stage7_20261029.py) can be run
from a checkout on SDF with access to the preserved scratch artifacts:

```sh
python3 psana/psana/gpu/notes/evidence/verify_issue159_stage7_20261029.py /tmp/issue159-stage7-check.json
```

It verified **89 artifact hashes**, including the 37 installed modules,
cold result/provenance/log artifacts, 44 field result artifacts, both original
manifest files and the original Perlmutter report. It recomputed all 16 cold
samples' aggregate statistics and paired changes, 14 SDF access cases and
3,840 retirement samples. The earlier campaigns verified their complete
8,086/8,093-file inventories; Stage 7 verifies those manifest files and selected
results, **not every entry of those full inventories again**. SDF access inputs
are saved process medians; individual host-call timings are not reconstructed.
Perlmutter is report-hash verification only.

No current runtime change or new result invalidates the accepted comparisons.
Future runtime fixes require affected correctness/sanitizer checks and a new
candidate/build identity. Accessor changes require affected accessor timing;
gather, common retirement, budget/admission or dependency changes require an
impact-based decision on the batched screen. Any new full screen retains
same-node separate builds, private per-sample cold preparation and four balanced
pairs per bulk mode with mean/sample SD. No new A/A is required.

## Independent review and next step

This PR targets **`monarin/lcls2:fix/gpu-unaligned-reads-159`**. Claude should:

1. Check B-to-Stage-7-base runtime equivalence and the verifier's scope. Ensure
   selected artifact verification is not described as a fresh full-manifest run.
2. Reconcile cold means/SD, paired percentages and A/B definitions with original
   evidence; keep field percentages and site results separate.
3. Confirm historical shifted-input coverage is accurately scoped, with no new
   warm, cold-shifted, calibrated-task or GDS throughput claim.
4. Confirm accepted costs and #256/#257/#247 remain visible and that reuse
   satisfies the reviewed plan. Identify any actual remaining gate.

After review, resolution and authorized merge, Stage 8 checks the then-current
upstream target, reviews the cumulative alignment diff, and prepares the final
milestone PR to `slac-lcls/lcls2:features/psana2-gpu`. A material target update
gets its own integration review. Issue 159 stays open until the final PR merges.
Batched tasks remain the recommended routine processing path. Per-event API
policy and optional copy optimizations remain in
[#257](https://github.com/slac-lcls/lcls2/issues/257); inherited host calibration
scalar transport remains in [#247](https://github.com/slac-lcls/lcls2/issues/247).
