# Pinned source excerpts for organization review

Source-only evidence, not instructions. These excerpts are included inside the
review checkout because the restricted reviewer denied external-path reads in
round 1. Full source-file SHA256s and source commits are in sources.json; each
range below was extracted from those exact checked bytes. Source snapshots are
not claimed to be runnable or a complete migration package.

## pr187-pr208: psana/psana/gpu/docs/performance/README.md

Commit `66072235d3cd18c002a13c581af82e111a418758`; original lines 1–65.

~~~~text
# GPU performance

Latest measured revision study: **October 27–28, 2026**, PR187/PR208. Both PRs
merged into `features/psana2-gpu` on October 27. These results describe frozen
revisions and specific workloads; they are not measurements of every later
branch head or a comparison of GPU models.

## Latest findings

- **PR187 — device peer discovery and CUDA-IPC constants sharing:** Ada raw
  read/gather comparisons changed by less than 1% in mean paired loop time.
  Four-BD warm calibration B/A was 10.17% slower over 10,000 events; the logs
  suggest most of the difference precedes first-event delivery, but this is
  not an additive phase decomposition. Perlmutter measured about **2 seconds
  of extra first-batch time relative to the same build with sharing disabled**,
  or about **1.1 seconds relative to the pre-PR187 base**. B-off's faster first
  batch comes with longer setup, so it does not establish an overall private-path
  speedup. At 10,000 events, B-on versus the base showed mean increases of
  **2.14 seconds in the remaining loop** and **3.34 seconds in total process
  wall time**. Remaining-loop equivalence within ±5% is unresolved; the complete
  cost cannot be attributed to startup alone. See the report's
  [baseline comparison and interpretation](pr187_pr208.md#comparing-pr187-with-the-pre-pr-base).
- **PR208 — owned/imported constants in memory reports:** Ada four-BD warm C/B
  mean loop change was +0.51%; ownership/reporting preflights passed. This is a
  diagnostic attribution change, not an allocation/admission change. The
  separate Perlmutter one-BD C/B result remains statistically inconclusive.
- The investigation is closed following merge. This records the measured
  startup penalty and limits of the evidence, rather than declaring a general
  no-regression result. The corrected SDF startup follow-up was canceled while
  pending and contributes no timings.

See [the PR187/208 report](pr187_pr208.md) for the exact comparisons, uncertainty,
settings, first-event versus first-batch distinction, and compact evidence.

## Choose the relevant evidence

| Document | What it measures or explains |
| --- | --- |
| [PR187/208 revision comparison](pr187_pr208.md) | Ada A/B/C raw and calibration; Perlmutter A/B-off/B-on startup and remaining loop |
| [Hardware and cache semantics](hardware.md) | Dated Ampere SSD/PCIe classes, actual Ada/Perlmutter hardware, and cold/warm definitions |
| [Read/staging performance](read_staging.md) | Historical raw read, GPU parsing and dense gather topology scaling; no calibration task |
| [User-kernel performance](user_kernels.md) | Historical calibration plus radial integration, matched batching and topology scaling |
| [Performance history](history.md) | Major changes/PRs, measured revisions and evidence links, newest first |

The older scaling tables are still useful. They measure different work from
calibration-only PR validation and should not be replaced by its rates.

## Reading the numbers

Compare revisions within one campaign, on the same physical node, dataset,
GPU mapping, cache condition and workload. Distinguish setup, first-use work,
remaining loop, cleanup and total process wall time. Positive paired duration
changes mean slower. Ratios of separately averaged rates are not paired
percentage changes; independently maximized rank durations are not additive.

Cold/warm refer to the **host file cache**, not GPU-resident inputs or warmed
kernels. Local NVMe cold and Lustre client-cold are different I/O paths. All
campaigns discussed here use KvikIO CPU fallback unless a report explicitly
states otherwise; none establishes true-GDS performance.

For new studies, add a dated report and compact evidence, update this overview,
and add a history entry with a short change description and PR links where
available. Retain prior measurements and their revisions. Keep job chronology,
failed attempts and qualification work in `notes/`; large logs, traces, builds
and datasets remain at the recorded artifact locations.
~~~~

## pr187-pr208: psana/psana/gpu/scripts/pr187_pr208/README.md

Commit `66072235d3cd18c002a13c581af82e111a418758`; original lines 1–108.

~~~~text
# PR187/208 benchmark harness archive

These are byte-for-byte snapshots used for the October 27, 2026 campaigns in
the [performance report](../../docs/performance/pr187_pr208.md). They preserve
the launch, cache controls, timing instrumentation and aggregation code behind
the reported measurements. They are historical campaign artifacts with fixed
site paths, rather than a portable benchmark entry point.

## Contents and provenance

| Campaign | Job | Archived files | Layout |
| --- | --- | ---: | --- |
| Ada A/B/C | 42114353, sdfada016 | 23 | `ada/screen/` driver and calibration measurement; `ada/harness/` raw benchmark and support modules; `ada/metadata/` launch checks; `ada/run.sbatch`. |
| Perlmutter A/B-off/B-on | 60415698 | 24 | `perlmutter/scripts/` driver, timing, analysis and existing CPU tests; `perlmutter/harness/` support modules; prepare/pilot/sweep Slurm launchers. |

The [manifest](manifest.json) records SHA-256, size and original absolute paths
for every snapshot, plus the path and SHA-256 of each campaign's full frozen
manifest. All 47 archived files (184,368 bytes) matched their run-frozen hashes.
The Ada support files were identical across A, B and C: each is stored once
under `ada/harness/`, with all three original paths recorded in the manifest.
Perlmutter helpers retain their own campaign versions.

The manifest covers the archived subset, not every file in the original frozen
runtime. README and manifest are newly authored archive documentation. The
[compact evidence](../../docs/performance/evidence/pr187_pr208_202610.json)
records source versions, result provenance and sample mappings; the
[sample-mapping audit](../../notes/pr187_sample_mapping_20261030.md) traces
Perlmutter logs through these scripts into the published arrays.

## Inspect and verify without GPUs

From this directory, verify the archived bytes:

```bash
python -B - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path('.')
manifest = json.loads((root / 'manifest.json').read_text())
for entry in manifest['files']:
    data = (root / entry['path']).read_bytes()
    assert len(data) == entry['bytes'], entry['path']
    assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['path']
print('Verified', len(manifest['files']), 'snapshots')
PY
python -B -m unittest discover -s perlmutter/scripts -p test_startup.py -v
```

To replay Perlmutter analysis, supply the original `results.json` and
`provenance.json` from `sweep-60415698` in a separate directory. With Python
3.11.14, this reproduces the original `summary.json` byte for byte:

```bash
PYTHONPATH=perlmutter/scripts python -B - /path/to/copy-of-results <<'PY'
import sys
from analysis import summarize
summarize(sys.argv[1])  # Writes summary.json in the supplied directory.
PY
```

Expected summary SHA-256:
`b954804bfd7da107c65b127c877dd044abaeb8897a25a1aa84f066363c87cfe1`.
Python 3.6 introduces tiny standard-deviation/interval rounding differences;
use the original Python version for byte comparison.

Ada aggregation is inline in `ada/screen/campaign.py`, from `summary=[]`
through the save of `summary.json`. Archive validation executed just that
block and `conditions()` against saved results: all 12 summary records matched
exactly. It did not invoke the campaign's `main()` or launch workloads.
Validation also checked all Python/shell syntax and passed the six existing
Perlmutter CPU tests. No GPU benchmark was rerun.

## Reconstructing a future campaign

The launchers retain their original accounts, node constraints, output paths
and environment setup. A new campaign needs a new scratch root and deliberate
path updates, rebuilt source/install variants, the original input selection
and reference fixtures, and a fresh frozen manifest. Retain these snapshots
unchanged when preparing such a campaign.

For Ada, copy the archived `ada/harness/` contents into each new
`screen/{A,B,C}/scripts/` tree; the manifest gives the exact original mapping.
The `screen/` files, `metadata/` files and launcher otherwise retain their
original relative layout. The driver also checks the prior integration
campaign's frozen installations and external dependencies. Its stage cleanup
and host/GPU assertions are specific to the original allocation.

Perlmutter preserves the original `scripts/` and `harness/` layout. Its
bootstrap expects source/install A and B, the dependency environment, and
provenance files including `states.json` and `HANDOFF.md`. Preparation depends
on the October 26 campaign's staged inputs, references and calibration
constants; the sweep checks inherited frozen runtime hashes. The unused
`stage1_regression` directory on its original `PYTHONPATH` is not archived.

Full logs/results, installed environments, source builds, raw XTC inputs,
reference/pixel fixtures and calibration payloads remain external at the
source paths recorded in the manifest and performance report. These snapshots
preserve the measurement code; they do not package those runtime prerequisites.

## Historical analysis caveats

The snapshot intentionally retains the original `analysis.py` report generator.
Its generated narrative is superseded by the corrected performance report:
missing B-off refresh timers must not be read as measured zero, and the
`within_5pct` flag is meaningful only for percentage differences, not absolute
timings. Use the saved samples and tagged comparisons for interpretation.
The archive adds no psana runtime behavior or new performance claim.
~~~~

## pr187-pr208: psana/psana/gpu/scripts/pr187_pr208/manifest.json

Commit `66072235d3cd18c002a13c581af82e111a418758`; original lines 1–19.

~~~~text
{
  "schema_version": 1,
  "archived_date": "2026-10-30",
  "campaigns": {
    "ada": {
      "job_id": "42114353",
      "root": "/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2",
      "frozen_manifest": "/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-performance-ada016-20261027-r2/screen/hashes.json",
      "frozen_manifest_sha256": "c9fb068211928f954aee28205391202ada27f265f6316830fe649cc8b397885c"
    },
    "perlmutter": {
      "job_id": "60415698",
      "root": "/pscratch/sd/m/monarin/psana2-gpu/pr187-ipc-startup-perlmutter-20261027/attempt01",
      "frozen_manifest": "/pscratch/sd/m/monarin/psana2-gpu/pr187-ipc-startup-perlmutter-20261027/attempt01/sweep-60415698/frozen-hashes.json",
      "frozen_manifest_sha256": "d397c38891c9d06f011a6ecd272ef357857508f8677b8b2f19ee6bcc4496ac8a"
    }
  },
  "files": [
    {
~~~~

## pr299: psana/psana/gpu/notes/pr299_perlmutter_20261115.md

Commit `e8d69a7e6604e1d459ad04c633de50a56b46ac9a`; original lines 1–177.

~~~~text
# PR299: completed Perlmutter Run51 performance screen

## Verdict and scope

**Accept the agreed Run51 performance follow-up; do not approve the PR for
merge on this evidence.** No consistent regression was demonstrated across
the six measured cases. Small PR effects remain inconclusive because of
paired spread, execution-order effects, and uncontrolled storage/cache state.
This is not a statistical equivalence test or a universal regression bound.

The measured comparison is exactly:

- Baseline: `11e4a13b541e2aefbfcb7168ab4d2040e2bcd4ba`.
- PR head: `acfd029a8d27898b5f165e02facd3c3bbc9c4ad2`.
- PR: <https://github.com/slac-lcls/lcls2/pull/299>.

The commit adding this archive is **not** the measured head. Neither these
results nor later evidence-only commits measure a subsequent C1-fixed revision.

## Execution and configuration

Job 61375111 on Perlmutter nid003260 completed with exit `0:0` in 02:44:14,
within the requested 04:03:00. Slurm allocated/billed a full four-GPU node;
each workload step used **one physical A100-SXM4-40GB**. BD1/BD2 mean one/two
BD workers sharing that GPU, plus one SMD0 and one EB, with eight disjoint
physical cores per rank. This was not a multi-node scaling test.

The 12 preflights used 120 events each. All 48 timed samples processed the
same first 10,000 of 17,872 available events from experiment mfx100848724,
Run51. Each of six cases has four adjacent pairs in B/H, H/B, B/H, H/B order.
The selected prefix contains one step and does not exercise EndRun.

| Setting | Value |
| --- | --- |
| GPU workload | Raw Jungfrau minimal task; calibration skipped; no full-image D2H |
| GPU streams | One per BD; two BDs share the physical GPU in BD2 |
| Batch / GPU budget | 20 events / 8 GB per BD |
| Reads | Bulk enabled, 1 MiB target; KvikIO compatibility ON (CPU fallback), no GDS |
| KvikIO | Four threads, 1 MiB tasks |
| CPU controls | Existing events and nested-steps iterator workloads, not equal-byte CPU/GPU speedup tests |
| Builds | Fresh local scratch release builds; optimization 3, debug false, LTO, ndebug true, C++20, build_daq false |
| Runtime | Python 3.9.20, GCC 14.3.0, CuPy 13.6.0, KvikIO 24.08.02, CUDA runtime 12.9 |
| MPI | Actually mapped Cray MPICH 9.0.1; module default was 9.1.0 |

## Paired results

Positive change means faster head: `100 * (head_events_per_second /
base_events_per_second - 1)`. The reported median is the median of four
paired ratios, not the ratio of median times. Ranges are observed pair
extrema, **not confidence intervals or equivalence bounds**.

| Case | Median change | Pair range | Median loop seconds, base / head |
| --- | ---: | ---: | ---: |
| GPU events BD1 | +0.00% | -3.28% to +2.75% | 149.89 / 150.89 |
| GPU events BD2 | +0.98% | -0.50% to +3.16% | 107.90 / 105.53 |
| CPU events BD1 | +0.40% | -2.12% to +0.64% | 223.92 / 222.72 |
| CPU events BD2 | +0.74% | -0.41% to +6.82% | 194.46 / 190.39 |
| CPU nested steps BD1 | -0.48% | -0.66% to +0.67% | 227.18 / 227.94 |
| CPU nested steps BD2 | +0.80% | -0.75% to +2.17% | 200.20 / 199.26 |

GPU BD1's head-second median is +2.52%, versus -2.79% head-first: a 5.31
percentage-point order gap hidden by its near-zero overall paired median.
Both GPU BD2 variants slow in the last pair. CPU events drift as well; CPU
BD2's first +6.82% pair is much larger than the other pairs.

Startup and MPI-process wall times are separate scopes. GPU BD1 max-rank
setup medians are 14.02 / 17.43 seconds; MPI-wall medians are 185.68 / 185.78.
Its *paired* MPI-wall throughput change has median -1.73%, ranging from
-10.47% to +4.81%. Event-loop acceptance is not startup/end-to-end equivalence.
Do not add medians of different timing scopes.

![Individual event-loop times in actual pair order](evidence/pr299_perlmutter_20261115/plots/event-loop-order.png)

## Sustained execution and storage

All 16 timed GPU samples completed. BD1 last-half rates range from 66.58 to
69.17 events/s; last-three-quarter min/max rate ratios are 0.923 to 0.977.
BD2 per-worker last-half rates range from 43.36 to 51.13 events/s, with
quarter ratios 0.767 to 0.941. These are rank-local windows with different
endpoints, not rates that can simply be added. BD1 logs reach 10,000 events;
BD2 unlogged final tails are 0 to 80 events, covered by final sample counts.

Each timed GPU sample records 335,571,760,000 completed read bytes; the 16
samples total 5,369,148,160,000. Accounting sums disjoint reset windows and
final tails. It is not unique data volume, physical traffic, or isolated
I/O bandwidth. Sampled GPU peaks are 1,831 MiB (BD1) and 3,657 MiB (BD2),
not exact allocation high-water marks.

The 48 derived storage summaries contain 4,157 observations. Peak observed
job memory is 94.006 GB against a recorded 240.965 GB limit. Job
`memory.events` deltas are zero, but substantial scan/steal/refault counters
are present: **no OOM does not mean no reclaim**. Actual monitor intervals
span 2.262 to 2.366 seconds; mean of sample means is 2.310 seconds.
Lustre llite and memory PSI were unavailable. Network counters are node/client
observations, not process-attributed or necessarily Lustre traffic. There
was no staging, cache eviction, or cold-cache claim. Storage versus reclaim
cannot be separated causally from this evidence.

Completed sustained measurements replace the earlier SDF short-prefix
extrapolation. They do not diagnose or demonstrate resolution of the prior
SDF/Weka slowdown, and do not establish performance for other datasets,
multi-stream concurrency, calibration-heavy work, or GDS.

## Provenance, auditability, and qualifications

The reviewed manifest is
`1433a2107fd8dd6ec589ce4fc8863b78df39e45d1e933e2fbca8814f6ce1eabd`.
All 1,171 entries in the transferred source bundle passed checksum verification
before packaging. The archive mapping retains original paths and hashes for
every copied file; the package checksum list covers the relocated package.
Paths in the preserved `FINAL_REPORT.txt` refer to the original tree: its
review is relocated to `reviews/campaign-r1.json`, its snapshot to
`campaign-job-61375111/`, and the PDF plot is omitted (the PNG is retained).
These locations are relative to this note's evidence directory.

The compute guards checked the intended initial psana import/native prefixes,
hashes of loaded project libraries, rank placement, and GPU isolation.
MPI/CUDA/KvikIO runtime observations were repeated after work. The preserved
`FINAL_REPORT.txt` overstates this as psana import validation both before and
after: `verify_install()` is called **before** the workload only. This note
qualifies that historical wording without rewriting the original artifact.

All timed samples record the timestamp digest
`2034208363befd6a797f889cbbff997a4c28d7297cf136de15e6b775a0597f17`,
which matches the independent SMD reference. The harness checked event lists,
per-rank order, and step assignments before discarding those lists. Offline
verification checks the recorded digest/counts, not every discarded event.
Ten SMD hashes and bigdata metadata matched at execution; full payloads
were not hashed. Fresh-build and post-campaign frozen-check claims rely on
the retained execution/build records; build trees and inputs are not included.

The campaign-r1 review approved readiness, not the statistical result or
merge. Its optional end-time cross-check and mocked campaign-loop test remain
nonblocking follow-ups. The original missing QoS/balance evidence was later
archived; only the non-sensitive QoS output is included here, not account
balances or reviewer authentication/invocation records.

The compact archive preserves sample summaries, per-rank progress/runtime/I/O,
sampled GPU measurements, commands, and status. Raw `storage.jsonl`, full
worker/event arrays, source/build/install trees, data, and complete node
inventory are omitted. Storage rollups can be checked against per-sample
derived summaries, but cannot be regenerated from raw storage logs here.
The sizing approval digest is reconstructed using its original-tree hash
index and estimate, not by rehashing the omitted sizing payloads.

## Functional review remains separate

- **C1 remains blocking and unfixed at the measured head:** a returned CPU
  Step loses its events when its temporary `run.steps()` parent is released.
  Nested-loop timings retain the parent and do not exercise that regression.
- **Run77 remains deferred:** no local dataset was available for the GPU
  events-versus-steps multi-step/EndRun comparison. This is a coverage gap,
  not another newly confirmed defect.
- **R2 remains nonblocking:** document GPU parent-generator lifetime.
- **R1 remains nonblocking:** remove the identified trailing whitespace.
- Parser-cost optimization remains separate in issue #310.

No C1 fix, production-code change, new job, or external PR review is part of
this evidence package.

## Evidence and scripts

- [Frozen scripts and execution-layout guide](../scripts/pr299_perlmutter/README.md).
- [Paired CSV](evidence/pr299_perlmutter_20261115/plots/paired-timings.csv),
  [campaign summary](evidence/pr299_perlmutter_20261115/campaign-summary.json),
  [validation and sustained windows](evidence/pr299_perlmutter_20261115/campaign-validation.json).
- [Storage rollup](evidence/pr299_perlmutter_20261115/campaign-storage-rollup.json)
  and [per-sample storage/progress](evidence/pr299_perlmutter_20261115/campaign-storage-progress.json).
- [Manifest](evidence/pr299_perlmutter_20261115/manifest.json),
  [readiness review](evidence/pr299_perlmutter_20261115/reviews/campaign-r1.json),
  [compact samples](evidence/pr299_perlmutter_20261115/campaign-job-61375111/).
- [Original source checksums](evidence/pr299_perlmutter_20261115/SOURCE_SHA256SUMS),
  [archive mapping](evidence/pr299_perlmutter_20261115/archive-map.json), and
  [package checksums](evidence/pr299_perlmutter_20261115/SHA256SUMS).

Package checksums use repository-relative paths and exclude their own file.
`SOURCE_SHA256SUMS` refers to the larger original transfer, not to this subset.
~~~~

## pr299: psana/psana/gpu/scripts/pr299_perlmutter/README.md

Commit `e8d69a7e6604e1d459ad04c633de50a56b46ac9a`; original lines 1–117.

~~~~text
# PR299 Perlmutter execution archive

This is the **historical execution harness**, not a new portable benchmark or
part of the production psana runtime. The files in `harness/` and `analysis/`
are byte-preserved selections from the completed Perlmutter campaign. They
were not rerun to prepare this evidence commit.

- [Reviewed results and limitations](../../notes/pr299_perlmutter_20261115.md)
- [Compact evidence](../../notes/evidence/pr299_perlmutter_20261115/)
- [Source-to-archive mapping](../../notes/evidence/pr299_perlmutter_20261115/archive-map.json)

## What was run

Baseline `11e4a13b541e2aefbfcb7168ab4d2040e2bcd4ba` and head
`acfd029a8d27898b5f165e02facd3c3bbc9c4ad2` were built locally from separate
scratch source/build/install trees. Job 61375111 completed 12 preflights and
48 timed samples of the first 10,000 events in mfx100848724 Run51. The six
cases are GPU events, CPU events, and CPU nested steps, each with one/two BD
workers. Each case has four adjacent baseline/head pairs with alternating
execution order. Run77 and the C1 fix were not included.

The original task root was:

```text
/pscratch/sd/m/monarin/agent-work/lcls2/pr299-perlmutter-screen-20261115TKJ0du6
    source-base/       source-head/
    build-base/        build-head/
    install-base/      install-head/
    harness/
    manifest.json      campaign-plan.json
    pilot-estimate.json
    review-approval-campaign.json
    results/reference-51.json
    results/pilot-job-61373624/
    results/campaign-job-61375111/
    evidence/          logs/          review/
```

`harness/campaign-job.sh` contains the submitted Slurm configuration and
invokes `python -B "$TASK_ROOT/harness/perlmutter_screen.py" campaign`.
The orchestrator runs `worker.py` through `srun --mpi=cray_shasta`, with
three/four ranks, eight physical cores per rank, one GPU per step, and
`--gpu-bind=none`. It sets separate install prefixes for each variant and
checks the source, build options, install hashes, input reference, completed
sizing evidence, and campaign review approval before launching. Per-sample
`command.json` files retain actual argv, timeout, and selected environment.

| Script | Purpose |
| --- | --- |
| `harness/campaign-job.sh` | Campaign Slurm wrapper and environment activation |
| `harness/perlmutter_screen.py` | Frozen-input checks, review gate, preflights, pairs, and process status |
| `harness/make_campaign_plan.py` | Bounds derived from completed local sizing samples |
| `harness/worker.py`, `harness/common.py` | Workload, independent SMD reference, event and step validation |
| `harness/check_runtime_provenance.py`, `harness/perlmutter_support.py` | Install/library, MPI, and runtime checks |
| `harness/gpu_identity.py`, `harness/io_accounting.py`, `harness/storage.py` | Placement, completed-read accounting, and monitoring |
| `analysis/summarize_campaign.py`, `analysis/analyze_storage_progress.py` | Paired and storage/progress summaries |
| `analysis/analyze_campaign.py` | Historical driver producing campaign-summary.json and campaign-storage-progress.json from original sample.json files |
| `analysis/campaign_validation.py`, `analysis/plot_campaign.py` | Contract checks, sustained windows, CSV, and order plot |
| `analysis/write_final_report.py`, `analysis/preserve_evidence.py` | Historical report generation and compact snapshot preservation |

The other harness helpers and tests are retained to preserve dependencies.
`screen.py`, `nic.py`, and `test_nic.py` are SDF-era helpers. `job.sh` is the
earlier Perlmutter pilot/generic wrapper with a 30-minute allocation; it accepts
a campaign argument but was not this campaign's entry point. The entry point
was **`campaign-job.sh`**. Neither archived wrapper should be submitted as-is.

## Offline verification of this package

From the repository root, using Python 3.9 or newer:

```bash
python3 -B psana/psana/gpu/scripts/pr299_perlmutter/verify_archive.py
python3 -B -m unittest discover -s psana/psana/gpu/scripts/pr299_perlmutter -p test_verify_archive.py
```

`verify_archive.py` and its tests were added during packaging; they are not
part of the measured harness. Verification reads files only and requires no
GPU, MPI, psana installation, dataset access, scheduler, or network. It
checks package/source checksums, approval binding, sample contracts, paired
timings and order, provenance, sustained progress, and storage aggregation.
It cannot reproduce physical storage measurements or discarded event arrays.

The [validation receipt](../../notes/evidence/pr299_perlmutter_20261115/packaging-validation.json)
describes the pre-receipt 630-file payload at
`5183e2d07cfbc77fe68438ee5396d136b7e787c3`, not subsequent review records.
The [packaging review records](../../notes/evidence/pr299_perlmutter_20261115/reviews/packaging-summary.json)
identify `f5b3a260938a1a9b56cdcfd87039033898cf4970` as the independently reviewed
payload; the later commit adds review records and documentation only.
Use the documented `python -B` commands: stray `__pycache__` or `.pyc` files
under the package cause its strict checksum-coverage check to fail.

Three source-format exceptions are intentionally retained: the original
`plots/paired-timings.csv` uses CRLF, `logs/slurm-61375111.err` ends with a
blank line, and `provenance/build-environment.txt` contains a space-only line.
These produce `git diff --check` warnings. Their source hashes are verified;
normalizing them would break byte-preserved provenance. Newly authored files
must pass the whitespace check without these exceptions.
The original build-options JSON files also omit a final newline. The
`trailing-whitespace` and `end-of-file-fixer` pre-commit hooks must not rewrite
these archived inputs; those two hooks are skipped for archive commits after
the scoped whitespace and source-hash checks. Other hooks remain enabled.

## Reuse is not automatic

Do **not** submit the archived batch scripts directly from this checkout.
They contain original absolute paths, account/environment choices, and
historical approval hashes. `common.ROOT` expects the original task layout.
Analysis scripts expect original `sample.json` and sometimes `storage.jsonl`,
not this archive's `sample-summary.json`; several write reports to fixed
locations. The frozen harness tests also include site/layout assumptions.
Their archived 94-test result is historical, not a claim that those tests
were rerun here.

A future run needs a new scratch task root, accessible inputs, clean builds,
reviewed site configuration, and a new manifest/approval. Preserve this
archive; adapt a separate copy rather than changing historical provenance.
No such adaptation or new performance job is part of this commit.
~~~~

## pr299: psana/psana/gpu/scripts/pr299_perlmutter/verify_archive.py

Commit `e8d69a7e6604e1d459ad04c633de50a56b46ac9a`; original lines 1–85.

~~~~text
"""Read-only, standard-library audit of the compact PR299 evidence package."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


SCRIPTS = Path('psana/psana/gpu/scripts/pr299_perlmutter')
EVIDENCE = Path('psana/psana/gpu/notes/evidence/pr299_perlmutter_20261115')
REPORT = Path('psana/psana/gpu/notes/pr299_perlmutter_20261115.md')
BASE = '11e4a13b541e2aefbfcb7168ab4d2040e2bcd4ba'
HEAD = 'acfd029a8d27898b5f165e02facd3c3bbc9c4ad2'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_checksums(path):
    entries = {}
    for line in path.read_text().splitlines():
        checksum, name = line.split('  ', 1)
        relative = Path(name)
        require(re.fullmatch('[0-9a-f]{64}', checksum), 'Invalid SHA256')
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe checksum path')
        require(name not in entries, 'Duplicate checksum path')
        entries[name] = checksum
    return entries


def near(actual, expected, context):
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), context)


def package_files(repository):
    return {path for directory in (SCRIPTS, EVIDENCE)
            for path in (repository / directory).rglob('*') if path.is_file()} | {repository / REPORT}


def verify(repository):
    repository = repository.resolve()
    evidence = repository / EVIDENCE
    checksums = read_checksums(evidence / 'SHA256SUMS')
    files = package_files(repository) - {evidence / 'SHA256SUMS'}
    require(set(checksums) == {str(path.relative_to(repository)) for path in files}, 'Checksum coverage differs')
    for name, expected in checksums.items():
        path = repository / name
        require(path.resolve().is_relative_to(repository) and not path.is_symlink(), 'Unsafe package path')
        require(sha256(path) == expected, 'Checksum mismatch: ' + name)
    original = read_checksums(evidence / 'SOURCE_SHA256SUMS')
    mapping = load(evidence / 'archive-map.json')['files']
    for name, entry in mapping.items():
        require(name in checksums and checksums[name] == entry['sha256'], 'Mapped hash mismatch: ' + name)
        require((repository / name).stat().st_size == entry['bytes'], 'Mapped size mismatch: ' + name)
        if entry['source'] != 'SHA256SUMS':
            require(original[entry['source']] == entry['sha256'], 'Source hash mismatch: ' + name)
    manifest = load(evidence / 'manifest.json')
    approval = load(evidence / 'review-approval-campaign.json')
    review = load(evidence / 'reviews/campaign-r1.json')
    require((manifest['base'], manifest['head']) == (BASE, HEAD), 'Wrong measured refs')
    require(sha256(evidence / 'manifest.json') == approval['manifest_sha256'] == review['reviewed_manifest_sha256'], 'Approval binding differs')
    require(review['reviewed_commit'] == HEAD and review['verdict'] == 'pass', 'Historical readiness review differs')
    require(not any(finding['blocking'] for finding in review['findings']), 'Historical readiness blocker')
    harness = repository / SCRIPTS / 'harness'
    require({str(path.relative_to(harness)): sha256(path) for path in harness.rglob('*') if path.is_file()} == manifest['harness'], 'Frozen harness differs')
    require(sha256(evidence / 'reference-51.json') == manifest['reference_sha256'], 'Reference hash differs')
    sizing_index = load(evidence / 'provenance/sizing-source-index.json')['files']
    sizing_payload = dict(estimate=sha256(evidence / 'pilot-estimate.json'), results={name: entry['sha256'] for name, entry in sizing_index.items()})
    sizing_digest = hashlib.sha256(json.dumps(sizing_payload, sort_keys=True).encode()).hexdigest()
    require(sizing_digest == approval['pilot_evidence_sha256'] == review['pilot_evidence_sha256'], 'Sizing index/approval differ')
    build_options = []
~~~~

## old-results: benchmarks/scaling/README.md

Commit `9674e3e803354b65c86e42d616af50b23d2c6e3a`; original lines 1–43.

~~~~text
# psana single-node scaling results

This branch records measured SDF and Perlmutter scaling results, their test scripts and acceptance evidence, and the pending Sim10/XPP-cube campaign. Start with the campaign reports below; each record pins the sources and preserves the measurement scope.

| Campaign | Main finding | Record |
|---|---|---|
| SDF Sim10 parameter screen and heatmaps | Four BDs/one GPU is the practical staging profile; 48 configurations across seven allocations | [Report](campaigns/sdf/sim10-staging/2026-11-14-parameter-screen/README.md) |
| Perlmutter C++ stripe screen | Count 2: 12.92 GiB/s at 40 threads; count 1: 11.42 GiB/s | [Report](campaigns/perlmutter/cpp-read/2026-11-11-stripe-screen/README.md) |
| Perlmutter psana stripe screen | Count 1 led: 5.443 GiB/s with one GPU/four BDs, 100k events | [Report](campaigns/perlmutter/sim10-staging/2026-11-16-stripe-screen/README.md) |
| Perlmutter psana GPU scaling, count 1 | 5.075 / 7.558 / 9.341 / 9.842 GiB/s for 1/2/3/4 GPUs | [Report](campaigns/perlmutter/sim10-staging/2026-11-16-gpu-scaling/README.md) |
| Sim10/XPP-cube application baseline | Full S/E/EC/ECC results pending; prepared scripts preserved | [Task snapshot](campaigns/perlmutter/sim10-xpp-cube/2026-11-15-baseline/README.md) |

C++ full-file reads, psana staging, and full cube processing measure different work. SDF fixes four BDs total in its GPU comparison; Perlmutter scales four BDs per GPU. The newer Perlmutter scaling policy uses fewer CPU cores per BD than the earlier stripe screen. Compare paired cases within a campaign, and do not treat reader bandwidth as a single-OST ceiling.

## Layout

```text
benchmarks/scaling/
  README.md                    campaign index and interpretation
  FORMAT.md                    recording and reproduction conventions
  catalog.json                 machine-readable campaign index
  campaigns/<site>/<workload>/<date>-<experiment>/
    README.md                  result, configuration, tables and limitations
    manifest.json              pinned scripts, evidence, jobs and review state
    runs.csv                   recommended for new campaigns
    summary.json               recommended for new campaigns
    figures/                   optional compact PDF/SVG figures
    evidence/                  optional compact receipts and raw-file index
  sources/                     imported snapshots and per-file provenance
  tools/                       read-only archive validation and plot replay
  templates/                   starting point for the next campaign
notes/<original-experiment>/   existing reviewed scripts and native evidence
```

Existing experiments keep their reviewed paths: several bundle builders encode them. Their campaign manifests link to the authoritative tables and scripts instead of copying or relocating them. Imported SDF sources preserve their relative paths under `sources/`; the Perlmutter cube subtree is an exact committed snapshot under `notes/`.

The branch starts from `2de8179b8` of `work/perlmutter-sim10-stream-length`. The source checkout and active cube task remain separate. New source changes belong on their implementation branch; copy reviewed evidence into this results branch with source hashes and a new campaign record. Result status and review status are separate.

See [format and update workflow](FORMAT.md), [script and reproduction guide](tools/README.md), [import provenance and availability](sources/README.md), and [campaign template](templates/manifest.json). This branch does not submit jobs or authorize new campaigns.

[Archive validation](validation.json) records file-integrity checks, table recomputation and SDF heatmap replay.

[Claude catalog review](review/FINAL.md) passed in round 2 on implementation `c55407e1d`; original campaign review statuses remain separate.
~~~~

## old-results: benchmarks/scaling/FORMAT.md

Commit `9674e3e803354b65c86e42d616af50b23d2c6e3a`; original lines 1–41.

~~~~text
# Recording a scaling campaign

Use `campaigns/<site>/<workload>/<UTC-date>-<experiment>/`. Use distinct workloads for `cpp-read`, `sim10-staging`, and `sim10-xpp-cube`. A campaign is one controlled comparison with explicitly identified allocation blocks; a synthesis can reference several campaigns but must retain their cohort boundaries.

## Required record

Each campaign has a readable `README.md` and a `manifest.json` indexed in `catalog.json`. Manifest paths are repository-root-relative. Pin the framework commit separately from the harness commit, and preserve per-file hashes for packaged or untracked sources. Record the runtime/install receipt, environment versions, dataset identities and raw location. Never substitute the results-branch HEAD for the code that ran.

Record site, compute node, CPU/GPU models, MPI roles, active and allocated GPUs, BDs/GPU, workers, physical/logical CPU affinity, batch/stream/bulk/task/bounce parameters, input budget, calibration/cube options, D2H/export policy and GDS/compatibility state. Record stripe size, count and placement for each file; count 1 per file does not imply one OST for the node.

Define the timing bracket, event and byte denominators, warmup, repetitions, randomized ordering and cache checks. Separate full cases, correctness pilots, anchors and diagnostics. For new campaigns, retain one row per measured case in `runs.csv` with at least:

```text
case_id,job_id,block,repeat,phase,stage,site,node,storage,
gpus_allocated,gpus_active,bds,workers_per_bd,cpu_policy,stripe_count,
events_global,requested_bytes,loop_seconds,whole_case_seconds,payload_gib_s,
cache_state,nic_rx_bytes,nic_tx_bytes,raw_path,raw_sha256
```

Use explicit null/blank values for unavailable metrics; zero means measured zero. Keep unknown/failed cases out of summaries rather than plotting zero throughput. Store any per-rank/pixel arrays outside Git and retain a raw index with locations, sizes and SHA256s. Scratch paths may expire; hashes alone cannot recreate lost data.

For psana Sim10, payload GiB/s is globally requested input bytes / 2**30 / loop seconds; the five-stream fixture has 3,358,200 bytes/event. C++ whole-file byte counts include other file bytes and have their own denominator. NIC traffic, expanded image bytes, calibration bytes and cube exports are separate metrics.

## Acceptance and comparison

Record terminal job/step accounting, correctness and resource gates, raw-evidence audit and independent review separately. A completed scheduler job is not enough to accept throughput. An implementation review does not constitute a results review. Preserve failed attempts and their causes without promoting them into full-run tables.

Describe median versus mean, sample count, observed range and paired speedup formula explicitly. Keep nodes/allocations and workload stages separate; do not pool cohorts or divide unmatched absolute rates to claim an optimization. Three repeats support descriptive spread, not a storage ceiling or an isolated bottleneck claim.

For full Sim10/XPP-cube, record S/E/EC/ECC separately, expansion shape and dtype, calibration definition, dense-bank bins/dtype, nonempty/empty-bin counts, the exact validation policy, GPU/host/pinned peaks and end-to-end total. Record numerical-reference checks and export/reduction only when performed; the current prepared bootstrap deliberately performs neither, retaining distributed GPU cubes with shape/content/bin-count checks. Current staging batch and sharing choices are not established application optima.

## Updating this branch

1. Keep execution scripts on the implementation branch; finish its intended review and checks there. New job submission follows the task's authorization, not this catalog.
2. Preserve an exact script snapshot or immutable source commit with its dependency/bundle manifest. Keep existing `notes/` paths because bundle builders may embed them.
3. Copy only compact reports, CSV/JSON receipts, review records and PDF/SVG figures. Keep XTC data, native installs, profiler traces and full cubes in site storage with hashes/paths.
4. Add a new dated record from the template. Link existing authoritative evidence instead of introducing edited copies. If correcting a report, state the correction and retain the native receipt.
5. Run `python3 benchmarks/scaling/tools/check_catalog.py`. Recompute or replay analysis with the documented tools where available. A catalog pass verifies archived records; it does not replace the original raw audit.
6. Commit the record to `results/psana-scaling`. Publish/push only as requested. Do not merge changing framework code into this branch just to update a result.

Historical SDF imports contain the full normalized report tables and plotting code, but not every original cohort source CSV or linked report. Their origin and availability are explicit in the campaign record. Archive replay is a derivative calculation, not a claim that the original raw data have been re-audited.
~~~~

## old-results: benchmarks/scaling/tools/check_catalog.py

Commit `9674e3e803354b65c86e42d616af50b23d2c6e3a`; original lines 1–125.

~~~~text
"""Read-only integrity and arithmetic checks for the scaling results archive."""
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
BYTES_PER_EVENT = 3358200


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify_file(record):
    path = (REPO / record["path"]).resolve()
    require(path.is_relative_to(REPO), "Path outside repository: " + str(path))
    require(path.is_file(), "Missing artifact: " + str(path))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    require(actual == record["sha256"], "Hash mismatch: " + str(path))


def near(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12),
            "Arithmetic mismatch: " + label)


def expect_cells(cells, fields, expected, label):
    keys = [tuple(cell[field] for field in fields) for cell in cells]
    require(len(keys) == len(expected) and set(keys) == set(expected), label + " cell population")


def validate_location(record, manifest_path):
    parts = manifest_path.relative_to(ROOT).parts
    require(len(parts) == 5 and parts[0] == "campaigns" and parts[-1] == "manifest.json",
            "Campaign path structure")
    require(record["site"] == parts[1] and record["workload"] == parts[2],
            "Campaign site/workload taxonomy mismatch")


def check_readme_table(text, expected, label):
    table = [line for line in text.splitlines() if line.startswith("|")]
    actual = [[cell.strip() for cell in line.strip("|").split("|")] for line in table[2:]]
    require(actual == expected, label + " README table differs from source summaries")


def main():
    catalog = read_json(ROOT / "catalog.json")
    require(catalog["schema_version"] == 1, "Catalog schema")
    ids = set()
    records = {}
    reports = {}
    checked = 0
    for entry in catalog["campaigns"]:
        require(entry["id"] not in ids, "Duplicate campaign")
        ids.add(entry["id"])
        manifest_path = (ROOT / entry["manifest"]).resolve()
        require(manifest_path.is_relative_to(ROOT), "Manifest outside catalog")
        record = read_json(manifest_path)
        validate_location(record, manifest_path)
        require(record["id"] == entry["id"], "Campaign identity mismatch")
        require(record["status"] in ("planned", "pending", "complete", "failed"), "Campaign status")
        require(record["review"] and record["metric"]["timing"], "Missing review or timing scope")
        report = ROOT / entry["report"]
        require(report.is_file(), "Missing campaign report")
        if record["status"] == "complete":
            require(record["full_cases"] > 0 and record["jobs"], "Completed campaign lacks measurements")
        for item in record["artifacts"]:
            verify_file(item)
            checked += 1
        records[record["id"]] = record
        reports[record["id"]] = report
    imports = read_json(ROOT / "sources/IMPORTS.json")
    for item in imports["files"]:
        verify_file(item)
        checked += 1

    # Validate authored catalog links; historical snapshots retain original links.
    docs = [ROOT / "README.md", ROOT / "FORMAT.md", ROOT / "tools/README.md",
            ROOT / "sources/README.md", ROOT / "templates/README.md"]
    docs += [ROOT / item["report"] for item in catalog["campaigns"]]
    for doc in docs:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", doc.read_text()):
            if "://" not in target and not target.startswith("#"):
                require((doc.parent / target.split("#")[0]).exists(), "Broken link: " + str(doc) + " -> " + target)

    # Recompute source summaries without pooling allocations or phases.
    sdf = ROOT / "sources/sdf-20261115/reference/notes/sim10_parameter_report_20261114/artifacts"
    sdf_rows = rows(sdf / "runs.csv")
    require(Counter(r["phase"] for r in sdf_rows) == {"full": 102, "pilot": 48, "anchor": 6}, "SDF case inventory")
    require(len({(r["cohort"], r["case"]) for r in sdf_rows}) == 156, "SDF duplicate case")
    fields = ("cohort", "media", "gpus", "bds", "workers", "streams", "batch", "task_mib", "bounce_mib")
    groups = defaultdict(list)
    for row in sdf_rows:
        near(row["gib_s"], int(row["events"]) * BYTES_PER_EVENT / 2**30 / float(row["seconds"]), "SDF rate")
        if row["phase"] == "full":
            groups[tuple(row[k] for k in fields)].append(row)
    summaries = rows(sdf / "summary.csv")
    require(len(groups) == 48, "SDF cells")
    expect_cells(summaries, fields, set(groups), "SDF")
    for summary in summaries:
        group = groups[tuple(summary[k] for k in fields)]
        require(len(group) == int(summary["n"]), "SDF repeat count")
        near(summary["mean_gib_s"], statistics.mean(float(r["gib_s"]) for r in group), "SDF mean")

    cpp = REPO / "notes/sim10_striping_20261110/reader/sweep"
    cpp_rows = rows(cpp / "trials.csv")
    require(len(cpp_rows) == 54, "C++ case count")
    for row in cpp_rows:
        near(row["gib_per_second"], 335820951264 / 2**30 / float(row["seconds"]), "C++ rate")
~~~~

## old-results: benchmarks/scaling/sources/README.md

Commit `9674e3e803354b65c86e42d616af50b23d2c6e3a`; original lines 1–7.

~~~~text
# Imported source and evidence snapshots

[IMPORTS.json](IMPORTS.json) records every imported file, its original path and SHA256. The SDF archive retains the exact verified transfer bytes, including report figures and explicitly packaged untracked scripts. Its coordinator Git commit alone does not identify those untracked sources. The Perlmutter cube files under `notes/sim10_perlmutter_baseline_20261115` come from recorded commit 746a883f79089684ee2378fbb4fedca0a5459a2f; all 44 execution/test source files also match reviewed implementation af6ebc03b86033accc6c0af34bb6e3eb2b85f35c.

[sdf-input-availability.json](sdf-input-availability.json) identifies the original inputs cited by the SDF report: nine are present with matching hashes, eight were not transferred. The complete normalized 156-row measurement table, 48-cell summary, comparisons, four-page PDF and four SVG heatmaps are preserved. The archive replay uses those retained measurements without claiming access to missing original raw inputs.

These snapshots preserve historical paths, statuses and approval records. They are evidence for their original tasks. Use a new implementation commit and campaign entry for changes; do not edit an imported file or treat an old approval as permission to launch a new job. Original whitespace, including generated SVG formatting and patches, is intentionally preserved so source hashes remain valid.
~~~~

## old-results: benchmarks/scaling/sources/IMPORTS.json

Commit `9674e3e803354b65c86e42d616af50b23d2c6e3a`; original lines 1–28.

~~~~text
{
  "schema_version": 1,
  "sdf_coordinator_commit": "004b30e01668ad1269db41f5455a27159e32d36a",
  "sdf_source_policy": "Includes explicitly packaged untracked sources; coordinator commit alone is insufficient.",
  "package_sha256": "0e37dd75e1b9fc5bb53cf09811eee898d7d4e2f2ce934a9662537684efde5690",
  "perlmutter_cube_source_commit": "746a883f79089684ee2378fbb4fedca0a5459a2f",
  "files": [
    {
      "path": "benchmarks/scaling/sources/sdf-20261115/reference/notes/jf_cold_ffb_20261109/run.py",
      "sha256": "a946811519d497629684169d8ca081abcae4e6ca136ec9fff06ab35c7a34fd36",
      "origin": "/sdf/scratch/users/m/monarin/gpu-validation/sim10-compute-cube-20261115/attempt03/source/notes/jf_cold_ffb_20261109/run.py",
      "transfer_path": "/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-transfer-20261115/package/reference/notes/jf_cold_ffb_20261109/run.py"
    },
    {
      "path": "benchmarks/scaling/sources/sdf-20261115/reference/notes/jf_raw_cache_20261109/common.py",
      "sha256": "d4ffdde96c09c6895e350bbd6deb6860543bcbaffa927f3a3be03bb3d41f5c63",
      "origin": "/sdf/scratch/users/m/monarin/gpu-validation/sim10-compute-cube-20261115/attempt03/source/notes/jf_raw_cache_20261109/common.py",
      "transfer_path": "/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-transfer-20261115/package/reference/notes/jf_raw_cache_20261109/common.py"
    },
    {
      "path": "benchmarks/scaling/sources/sdf-20261115/reference/notes/jf_raw_cache_20261109/reference.json",
      "sha256": "94625e5b7efb92ba38e3a60050d688a86f912c946a219f61315b15f890f0d9ce",
      "origin": "/sdf/scratch/users/m/monarin/gpu-validation/sim10-compute-cube-20261115/attempt03/source/notes/jf_raw_cache_20261109/reference.json",
      "transfer_path": "/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-transfer-20261115/package/reference/notes/jf_raw_cache_20261109/reference.json"
    },
    {
      "path": "benchmarks/scaling/sources/sdf-20261115/reference/notes/jf_raw_cache_20261109/runtime.sh",
      "sha256": "4fe181ef41dfb56565956aae8ec0718658c331292c1fc0340f770b2e56cfe524",
~~~~
