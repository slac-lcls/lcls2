# GPU performance

This index brings together dated studies on the results/psana2-gpu-scaling
branch. Each report identifies its measured revisions, workload and evidence
limits. A report is not a measurement of the current branch head.

## PR299: Perlmutter Run51, November 15, 2026

The [PR299 report](https://github.com/slac-lcls/lcls2/tree/e8d69a7e6604e1d459ad04c633de50a56b46ac9a/psana/psana/gpu/notes/pr299_perlmutter_20261115.md) covers 48 timed
samples in six baseline/head cases, with one or two BDs sharing one GPU.
No consistent regression was demonstrated in this scope; storage/cache state
was uncontrolled and small effects remain inconclusive. The measured head
still had the C1 CPU Step-parent lifetime defect, and Run77 was deferred.
Performance acceptance is not functional merge approval.

The [frozen scripts](https://github.com/slac-lcls/lcls2/tree/e8d69a7e6604e1d459ad04c633de50a56b46ac9a/psana/psana/gpu/scripts/pr299_perlmutter/README.md) and
[verified archive](https://github.com/slac-lcls/lcls2/tree/e8d69a7e6604e1d459ad04c633de50a56b46ac9a/psana/psana/gpu/notes/evidence/pr299_perlmutter_20261115) preserve
the actual execution package. These archives are source-only: GPU notes and
frozen campaign scripts are excluded from installed packages. In an installed
copy, use the corresponding source revision on
[GitHub](https://github.com/slac-lcls/lcls2/tree/e8d69a7e6604e1d459ad04c633de50a56b46ac9a/psana/psana/gpu).

## PR187/208: October 27–28, 2026

Both PRs merged into features/psana2-gpu on October 27. Their frozen revision
comparisons are distinct from the PR299 workload and cache conditions.

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
| [PR299 Run51 regression study](https://github.com/slac-lcls/lcls2/tree/e8d69a7e6604e1d459ad04c633de50a56b46ac9a/psana/psana/gpu/notes/pr299_perlmutter_20261115.md) | Paired GPU raw-input and CPU iterator controls on Perlmutter; scope and functional limitations |
| [Input-group bookkeeping](input_group_bookkeeping.md) | Historical optimization and extracted PR298 staging acceptance |
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
changes mean slower; PR299 reports positive throughput changes as faster. Ratios of separately averaged rates are not paired
percentage changes; independently maximized rank durations are not additive.

Where a campaign controls cache state, cold/warm refer to the **host file cache**, not GPU-resident inputs or warmed
kernels. Local NVMe cold and Lustre client-cold are different I/O paths. All
campaigns discussed here use KvikIO CPU fallback unless a report explicitly
states otherwise; none establishes true-GDS performance.

For new studies, add a dated report and compact evidence, update this overview,
and add a history entry with a short change description and PR links where
available. Retain prior measurements and their revisions. Keep job chronology,
failed attempts and qualification work in `notes/`; large logs, traces, builds
and datasets remain at the recorded artifact locations.

## Scaling and application campaigns

The [campaign catalog](https://github.com/slac-lcls/lcls2/tree/results/psana2-gpu-scaling/psana/psana/gpu/docs/performance/catalog.json) indexes the historical PR studies and these
five scaling/application records. Catalog and campaign records are source-only;
an installed copy can use the
[results branch](https://github.com/slac-lcls/lcls2/tree/results/psana2-gpu-scaling/psana/psana/gpu/docs/performance).
The branch was published to slac-lcls/lcls2 on November 16, 2026.
In a source checkout, open `docs/performance/campaigns/README.md` for working local links.

| Campaign | Main finding | Record |
|---|---|---|
| SDF Sim10 parameter screen and heatmaps | Four BDs/one GPU is the practical staging profile; 48 configurations across seven allocations | [Report](https://github.com/slac-lcls/lcls2/tree/results/psana2-gpu-scaling/psana/psana/gpu/docs/performance/campaigns/sdf/sim10-staging/2026-11-14-parameter-screen/README.md) |
| Perlmutter C++ stripe screen | Count 2: 12.92 GiB/s at 40 threads; count 1: 11.42 GiB/s | [Report](https://github.com/slac-lcls/lcls2/tree/results/psana2-gpu-scaling/psana/psana/gpu/docs/performance/campaigns/perlmutter/cpp-read/2026-11-11-stripe-screen/README.md) |
| Perlmutter psana stripe screen | Count 1 led: 5.443 GiB/s with one GPU/four BDs, 100k events | [Report](https://github.com/slac-lcls/lcls2/tree/results/psana2-gpu-scaling/psana/psana/gpu/docs/performance/campaigns/perlmutter/sim10-staging/2026-11-16-stripe-screen/README.md) |
| Perlmutter psana GPU scaling, count 1 | 5.075 / 7.558 / 9.341 / 9.842 GiB/s for 1/2/3/4 GPUs | [Report](https://github.com/slac-lcls/lcls2/tree/results/psana2-gpu-scaling/psana/psana/gpu/docs/performance/campaigns/perlmutter/sim10-staging/2026-11-16-gpu-scaling/README.md) |
| Sim10/XPP-cube application baseline | Full S/E/EC/ECC results pending; prepared scripts preserved | [Task snapshot](https://github.com/slac-lcls/lcls2/tree/results/psana2-gpu-scaling/psana/psana/gpu/docs/performance/campaigns/perlmutter/sim10-xpp-cube/2026-11-15-baseline/README.md) |

C++ full-file reads, psana staging, and full cube processing measure different work. SDF fixes four BDs total in its GPU comparison; Perlmutter scales four BDs per GPU. The newer Perlmutter scaling policy uses fewer CPU cores per BD than the earlier stripe screen. Compare paired cases within a campaign, and do not treat reader bandwidth as a single-OST ceiling.


[Record format](https://github.com/slac-lcls/lcls2/tree/results/psana2-gpu-scaling/psana/psana/gpu/docs/performance/FORMAT.md) · [Offline tools](https://github.com/slac-lcls/lcls2/tree/results/psana2-gpu-scaling/psana/psana/gpu/scripts/performance/README.md) · [Migration verification](https://github.com/slac-lcls/lcls2/tree/results/psana2-gpu-scaling/psana/psana/gpu/notes/scaling_migration_20261116/README.md).

The [maintained PR regression template](https://github.com/slac-lcls/lcls2/blob/results/psana2-gpu-scaling/psana/psana/gpu/docs/performance/templates/pr-regression.md)
provides the controller, run-51 candidate profile and agent workflow for new
comparisons. Run51 passes the pixel-status check. A qualified native adapter,
durable dataset identity/retention and native correctness/A/A remain required
before measuring a new PR; template validation supplies no performance result.

See the [campaign evidence assessment](https://github.com/slac-lcls/lcls2/blob/results/psana2-gpu-scaling/psana/psana/gpu/notes/campaign_evidence_assessment_20261116/README.md)
for the distinction between execution completion, archive verification and
independent results review, and which historical gaps warrant follow-up.
