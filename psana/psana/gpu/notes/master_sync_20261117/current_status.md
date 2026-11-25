# PR331: master synchronization and MPI teardown fixes

Initial master → `features/psana2-gpu` synchronization after #299, now including
PR326's MPI termination fixes and the EndRun iterator repair. Refs #330, the
standing synchronization tracker; do not close it.

**Draft and unmerged.** All three GitHub checks passed on published bundle
`34f227cc1`. Mona will ask Seema to review it; merging requires Mona's explicit
instruction. Raw performance-evidence archival and final landing checks remain
open. This November 25 overview supersedes earlier current-status statements;
it preserves the historical measurements, failures, and review boundaries.

## What is included

- The original master synchronization: retain both changelog histories and the
  union of DataSource options; preserve master's persistent CPU callback manager
  and two-part interface through the GPU branch's `batches_with_gpu()` adapter;
  preserve layout guards, transitions, callback statistics, and GPU transport.
  Callback-plus-GPU routing remains unsupported.
- Honor deletion of `setup_env.sh`, retain pip CUDA-library discovery in
  `setup_env_ana.sh`, reconcile the extract-subset test without executable changes,
  and preserve the combined CI selections and master's PR trigger.
- PR326 (`98f344a32`): receive BD termination messages and consume final requests
  from EBs sent step-only messages, preventing stale messages across runs.
- EndRun repair (`fa695530a`): exhaust the step iterator's transport naturally
  before next-run collectives. Preserve PR299's detached CPU Step lifetime and
  the existing GPU cleanup order.
- Regression coverage: retained-Step exhaustion, repeated-run MPI drivers, and
  the requested PR326 fake-communicator `Recv` plus empty-response-queue assertion
  (`dc59d0b94`). No assertion was weakened to make CI pass.

The original integration changed 334 files. At bundle `34f227cc1`, the full PR
has 502 changed paths. Changes after the original implementation comprise 163
documentation/evidence files and six runtime/test files: only `mpi_ds.py` and
`node.py` change production behavior; the other four are tests/drivers.
The documentation cleanup adds no runtime, test, or workflow changes.

## Revision and evidence map

| Role | Revision |
| --- | --- |
| GPU target/baseline, including #299 | `546764e0338225d026faa29627297a4a390d3396` |
| Incoming master | `785f7ef5dd2428bc0f1fb87c0ff1ebbab4d37935` |
| Original integration and performance-measured runtime | `5bd5c7ba300cf0a0894b5b904d3975d5ba0b3cad` |
| Final combined code tested on Perlmutter | `dc59d0b94bce73caa134b1c67b40781f6b5e5c65` |
| Claude's combined-code/evidence review snapshot | `c7f91b1457fb4bf5a40a326dc0b3554e2f4d5f89` |
| Published bundle with passing GitHub CI | `34f227cc12a118d16d836a5acd7650566479e0d8` |

Runtime/test sources are identical across the last three revisions. Subsequent
documentation-only commits do not establish a new performance measurement or a
new full-suite result. Consult the PR checks for CI on its current head.

The original merge retains GPU baseline first-parent and incoming master
second-parent history; PR326 ancestry is also preserved. **When authorized,
land with a merge commit, not squash/rebase.** The target is the GPU branch,
not master. The target remained `546764e0` at this reconciliation; recheck before
landing.

## Review and correctness results

The five original conflict resolutions passed scoped independent reviews. The
fifteen automatically merged existing-file groups received a source-level
compatibility assessment, not an exhaustive independent review of all incoming
master changes or DAQ hardware validation. Claude round 2 passed the later
combined fix and supplied evidence with zero findings, closing both prior
findings. Seema's review is still pending.

| Evidence scope | Result and limits |
| --- | --- |
| Original SDF full-collection reconciliation | 2,140 passed, 5 skipped, 2 failed/hung, 1 deferred across attempts; not one green full-suite run. Includes 1,960 GPU-directory passes and 561 GPU-marked cases. |
| Final Perlmutter job 61947791, combined code | 1,438 selected checks passed: 6 CPU lifetime, 17 callback, 1,398 GPU unit, 15 real-device component, 2 PR326 multi-run. One default slow sweep deselected; no selected skips. |
| Original-runtime negative control | Failed as expected on unconsumed termination, demonstrating the strengthened test detects the bug. Not a candidate failure. |
| Formerly deferred `test_fresh_callback_per_run[mpi]` | Now passes on the combined code, about four seconds locally. PR326 alone was not sufficient; the EndRun repair is also included. |
| PR326 `[14,15]` drivers | Both deliver 10 + 10 events, at 1 EB/4 ranks and 2 EBs/5 ranks. |
| GitHub Test psana on `34f227cc1` | 114 passed, 3 skipped; includes the formerly failing MPI callback test and all six CPU Step lifetime cases. |
| GitHub Test DAQ and Check changelog on `34f227cc1` | Both passed. |

These suites overlap; do not add their counts. GitHub psana CI uses a selected
CPU/MPI suite, not the GPU unit suite or `byhand_mpi.py`. Perlmutter used Python
3.11/Cray MPICH with an `mpirun`-to-`srun` adapter. Its 15 device component checks
are not full MPI/GPU multi-run coverage. The two PR326 driver logs preserve a
nonfatal MPICH pending-reference warning; successful exits do not establish
absence of MPI resource leaks. The complete byhand aggregator was not rerun in
this latest campaign; its two added multi-run configurations were run through a
bounded wrapper. Earlier failed attempts remain failures in their records.

Passing CI: [psana](https://github.com/slac-lcls/lcls2/actions/runs/45518094976),
[DAQ](https://github.com/slac-lcls/lcls2/actions/runs/45518094940),
[changelog](https://github.com/slac-lcls/lcls2/actions/runs/45518095058).
The psana run checked GitHub's merge of `34f227cc1` into `546764e0`.

## Performance disposition

The accepted performance scope is the five-case Perlmutter raw-input A/B screen
on original Jungfrau run51: 10k events/sample, six balanced pairs/case, ±5%
equivalence margin, one A100 for GPU cases, KvikIO compatibility mode/GDS off.
Separate A/A was omitted under the narrowed scope. Positive change means slower.

| Case | Mean paired loop change | 95% interval | Classification |
| --- | ---: | --- | --- |
| GPU events, 2 BDs | -0.51% | -3.43% to +2.40% | equivalent |
| GPU events, 1 BD | -11.66% | -30.04% to +6.72% | inconclusive |
| CPU events, 1 BD | -0.37% | -3.35% to +2.61% | equivalent |
| CPU events, 2 BDs | +1.31% | -5.77% to +8.38% | inconclusive |
| CPU steps, 1 BD | -0.17% | -1.55% to +1.20% | equivalent |

Mona accepted both inconclusive outcomes and the documented measurement limits
for this merge on November 24. Their classifications remain inconclusive;
acceptance does not establish equivalence or rule out a slowdown greater than 5%.
All five scoped execution reviews passed. Loop equivalence is not end-to-end
or true-GDS equivalence. Calibrated processing, user-task/D2H/output workloads,
and broader scaling remain follow-up under #330.

**These measurements apply to `5bd5c7ba`, before the two production-file teardown
fixes.** The combined code received correctness verification and CI, but no new
performance campaign. Preserve that distinction; the old timings are not fresh
measurements of `dc59d0b94` or the published bundle.

## Known issues and retained boundaries

- Two inherited CPU slow-test failures remain tracked in #351. Historical skips
  remain missing coverage; they are not relabeled as passes. The old callback
  MPI deferral is superseded by the combined-fix results above.
- Original candidate CPU steps/1BD SIGSEGV: observed once, not reproduced by all
  six diagnostic cells (61679475), 10k steps retries, or later A/B work; cause
  unresolved. Successful retries do not establish a fix or identical conditions.
- #299 coverage combines passed recovery tests and uncapped real-data job
  61937600: five serial/MPI event/step cases, exactly 60 events, steps 19/20/21,
  natural EndRun. Forced pressure was not exercised in that real-data job;
  the combined evidence does not establish that additional scenario.
- GPU float16 input parity remains under #349. Original conflict-review
  follow-ups and setup/hardware portability limits remain in the decision record.
- The separate pre-existing multi-EB teardown race noted in Mona's PR326 review
  remains a nonblocking follow-up under #209; the two passing topologies above
  do not rule out that race.

## Records and reviewer reading order

1. [Original merge decisions](https://github.com/slac-lcls/lcls2/blob/sync/psana2-gpu-master-20261117/psana/psana/gpu/notes/master_sync_20261117/merge_decisions.md): five conflicts, fifteen automatic-merge groups, rationale and follow-ups.
2. [Historical validation](https://github.com/slac-lcls/lcls2/blob/sync/psana2-gpu-master-20261117/psana/psana/gpu/notes/master_sync_20261117/validation.md): original correctness, provenance and qualification attempts.
3. [Frozen performance report](https://github.com/slac-lcls/lcls2/blob/46c06415a3131d6a9b20247e373a49f85b164370/psana/psana/gpu/notes/master_sync_20261117/performance_20261124.md) and [results catalog](https://github.com/slac-lcls/lcls2/blob/942a3bf9a190667f6d49cfd61a671e162e8f15fe/psana/psana/gpu/docs/performance/campaigns/perlmutter/pr331-run51/2026-11-24-raw-ab/README.md): original measured pair, timings, limits, manifests and verifier.
4. [Combined fix and raw verification](https://github.com/slac-lcls/lcls2/blob/34f227cc12a118d16d836a5acd7650566479e0d8/notes/pr331_endrun_combined_20261124/README.md) and [Claude review](https://github.com/slac-lcls/lcls2/blob/34f227cc12a118d16d836a5acd7650566479e0d8/notes/pr331_endrun_combined_20261124/review-round02.json): latest code checks and review limits.

The performance report's unverified-MPI statements predate the combined fix.
The combined-evidence README's local-only/no-fresh-CI statements predate
publication. Both are retained verbatim with their hash manifests as historical
snapshots; use this overview for their current dispositions. No historical
failure, test receipt, measurement, or review artifact is rewritten.

## Acceptance and landing checklist

- [x] Preserve master and PR326 ancestry and the original merge decisions.
- [x] Include PR326, the EndRun repair preserving #299, and the requested test update.
- [x] Complete selected combined verification and independent Claude review.
- [x] Obtain passing psana, DAQ, and changelog checks on bundle `34f227cc1`.
- [x] Record accepted performance uncertainty, measurement limits, and follow-ups.
- [x] Reconcile the current narrative and changelog without changing executable code.
- [ ] Resolve full raw performance-evidence retention/recoverability outside scratch.
  Compact evidence is committed; it does not substitute for the raw archive.
  Record the persistent destination, owner, checksums, and successful readback.
- [ ] Receive Seema's review; Mona will request it. Do not merge yet.
- [ ] Before an authorized merge, recheck the current head's CI, target drift,
  and evidence applicability. Later documentation checks do not replace code tests.
