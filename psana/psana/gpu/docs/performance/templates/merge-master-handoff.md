# Merge-master PR performance campaign handoff

The reusable template implementation and overhead-recording task is complete.
The next task owns one integrated campaign for the merge-master PR: select exact
builds, prepare its contracts/receipts, run the existing gated phases, and report
results. No additional generic overhead experiment is required by this handoff.
This is a task handoff, not a claim that the shipped placeholder profile is
execution-ready or that every PR component has already been qualified.

## Starting point

- Maintained branch: `slac-lcls/lcls2`, `results/psana2-gpu-scaling`. Start from its
  current reviewed template and record the exact template commit used.
- Runner: [scripts/performance/regression](../../../scripts/performance/regression/README.md).
  Keep campaign-specific frozen profiles, scripts and results in a new campaign
  directory; preserve prior study sources and captures.
- Workload: raw Jungfrau `mfx100848724`, run 51, 10k events, separate 200-event
  warmup, six CPU/GPU cases, with the settings in the maintained profile.
  Do not expand to full-node, calibrated, D2H or multi-stream cases by default.
- Perlmutter input specification:
  `/pscratch/sd/m/monarin/psana2-gpu/pr-regression-accounting-20261118/inputs-01.json`,
  SHA256 `8c8ff73085849969c8f162a7a1f5b13ff03e6419dd6dd7a93f811414daf0fbdd`.
  It points to the prepared private run-51 copy. Revalidate its manifest/stat/hash
  receipts and compute-node access; do not copy the dataset again unless needed.
- The installed `1995752058bb9aaa3f16ccd86c22e84153257538` runtime used for harness
  qualification is an API fixture, **not automatically the PR baseline**.
  Select and pin both actual PR revisions before building.
- Site scope: the maintained native adapter is Perlmutter-specific. SDF execution
  and arbitrary private-API compatibility are not established by these receipts.

## One integrated next task

1. Pin the baseline and candidate commit SHAs for the merge-master PR, state the
   comparison being made, and build isolated installations using [RUNTIME.md](../../../scripts/performance/regression/RUNTIME.md).
   Check both against the adapter's required APIs. Reuse existing relevant
   unit/device correctness tests; do not recreate them in the campaign.
2. Freeze the campaign's [component contract](component-regression.md) and input,
   runtime, adapter-review and qualification receipts before candidate measurement.
   Reuse verified historical evidence only within its actual scope. Derive bounds
   from algorithm/workload semantics or qualified baseline-only observations;
   never invent limits or tune them after candidate results. The existing
   `profile` phase runs **both** revisions, so it cannot serve as baseline-only
   bound calibration before the contract is frozen. Record unavailable metrics;
   a required unavailable bound remains unqualified.
3. Prepare allocation-local placement with [SITE.md](../../../scripts/performance/regression/SITE.md),
   validate the frozen profile's blockers, and use one reviewed orchestration of
   correctness -> A/A -> A/B. The controller enforces those dependencies. Run
   separate diagnostics when needed; never pool diagnostic timings into A/A or
   A/B. Preserve all samples, cache/NIC evidence, exact observer code/settings
   and scheduler outcomes. Use the existing scope's independent review loop.
4. Publish one PR report: timing results, component pass/breach/unavailable,
   correctness receipts, exact revisions/configuration, overhead limitation and
   any unresolved acceptance gates. A controller loop-time label alone is not
   full PR acceptance. Do not report performance acceptance if required evidence
   is missing or the A/A gate remains inconclusive.

The existing commands for a **prepared, qualified** profile, run in the same
approved compute allocation with the exact prepared controller, are:

```bash
set -euo pipefail
python -B /frozen/source/campaign.py execute /scratch/campaign correctness
python -B /frozen/source/campaign.py execute /scratch/campaign aa
python -B /frozen/source/campaign.py execute /scratch/campaign ab
python -B /frozen/source/campaign.py summarize /scratch/campaign ab
```

These placeholders illustrate sequential orchestration; they are not a runnable
campaign supplied by this handoff. No new all-in-one CLI was added. Runtime/site
preparation, honest receipt generation and component-bound selection remain
campaign-specific work. Do not flip the shipped qualification flags to make the
sequence run. Budget the full six-cell A/A and A/B workload separately from the
recent one-cell overhead job; do not reuse its six-hour allocation estimate.

## Carry forward the overhead limitation

The [completed A/B/C observer study](../../../notes/pr_regression_abc_20261119/README.md)
finished job 61610523 with six preflight and 36 main samples; all mechanical,
cache, NIC and scheduler checks passed. B/A, C/A and C/B timing intervals were
all inconclusive at +/-5%. Earlier [two-mode results and retry](../../../notes/pr_regression_overhead_20261118/README.md)
are retained too. No more overhead jobs are planned in the template task.

Use identical observer source/settings on both PR builds and report measured
performance under that instrumentation. Equal observer code does not prove equal
cost across configurations or runtime behavior; do not subtract the uncertain
point estimates or claim the overhead cancels. Do not label the full-detail
observer low-overhead or the native adapter fully qualified solely on this study.
If the campaign's acceptance policy requires demonstrated observer overhead,
record that requirement as unresolved for results review rather than silently
waiving it. Existing gates remain unchanged.

Production MPI request cleanup remains explicitly deferred and is not part of
this handoff.
