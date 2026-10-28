# GPU performance

Latest measured revision study: **October 27, 2026**, PR187/PR208. Both PRs
merged into `features/psana2-gpu` that day. These results describe frozen
revisions and specific workloads; they are not measurements of every later
branch head or a comparison of GPU models.

## Latest findings

- **PR187 — device peer discovery and CUDA-IPC constants sharing:** Ada raw
  read/gather comparisons changed by less than 1% in mean paired loop time.
  Four-BD warm calibration B/A was 10.17% slower over 10,000 events, with most
  of the difference before first-event delivery. Perlmutter's sharing on/off
  study measured about **2 seconds of extra first-batch time**, nearly constant
  between 1,000 and 10,000 events. Remaining-loop equivalence within ±5% is
  unresolved in the cold-Lustre measurements.
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
