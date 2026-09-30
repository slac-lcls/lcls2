# Bulk-on slot-selection index

2026-09-25. Implementation and correctness gates complete. Primary performance
job **39070635** completed on **sdfampere012**, exit `0:0`, in **15m05s**.
All **14 samples**, final source/placement checks, and exact paired pipeline
diagnostic comparison passed. Warm follow-up job **39071037** completed on
the same node in a separate allocation, exit `0:0`, in **8m05s**; all eight
controls and final source/placement checks passed.

## Change

`InputGroupPool.plan_slots` now snapshots free slots and cached capacities
once per call, sorts `(capacity, slot_id)` pairs, and uses binary search for
best-fit selection. It removes selected entries and uses an ascending cursor
for the lowest-ID fallback when no cached allocation fits. An undersized
fallback is also removed from the capacity index, preventing duplicate choices.

The index is local to the call. Growth, trimming, and completed ownership are
reflected by the next snapshot; no persistent capacity index requires updates
on error or retirement. Zero-byte cached buffers remain distinct from empty
slots. Busy slots, small-stream credits, group order, allocation reservations,
and completion dependencies retain their existing rules.

Baseline and candidate both include the earlier transition-drain fix and the
public bulk-target parameter. Frozen manifests verify that their **only
runtime difference is `psana/gpu/gpu_input_group.py`**. File-handle scanning
and legacy read planning were not changed in this step.

## Correctness validation

- **408 CPU unit tests passed**. Eight new cases include six explicit policy
  cases, 2,000 deterministic generated comparisons with the original policy,
  and successive calls after growth, trim, and out-of-order completion.
  Tests compare exact slot tuples and verify planning does not consume credits
  or mutate allocated buffers.
- **15 harness tests passed**, including the bulk-on-only and balanced
  warm-repeat matrices, separation of controls from diagnostic timings, and detection of changed pipeline
  counters/charged peak.
- **10 A100 tests passed** before the benchmark in job 39070635: group pixels
  and batched launches, independent delayed-consumer reclamation, partial
  parser setup/failure, deferred transition consumers, and multi-owner
  calibration transitions.
- Python syntax checks and `git diff --check` passed.

## Measurement method

One A100 allocation, one BD, three MPI ranks. Both builds use bulk on with
1,000 JF+feespec events, batch 100, depth 1, an 8 GiB device budget, eight
KvikIO workers, and 4 MiB bulk target/task size. Compatibility mode is ON;
data are on the same private Weka FFB SSD files and CPU references/calibration
are frozen. The shared native build is unchanged.

The 14-sample campaign has eight controls (cold/warm, both builds, two rounds
with reversed ordering), four separate loop profiles, and two separate warm
pipeline diagnostics. Only control timings determine throughput. All samples
retain warmup array checks, timestamp/feespec checksums, expected 5,019 API
requests and 33,566,911,424 payload bytes, cache residency and cold NIC checks.
Code/reference hashes and storage placement are checked before and after.

The separate CPU sanity measurement on the login node compared 500 selections
from 606 slots, 20 iterations per case. Exact choices matched. Median times
were 19.96 vs 0.26 ms for empty caches, 55.36 vs 0.51 ms for populated caches,
and 29.57 vs 0.43 ms for mixed caches with 100 busy slots. These are isolated
Python costs, not end-to-end GPU throughput or an acceptance criterion.

## Profile attribution

Primary unprofiled controls:

| Cache | Baseline R1 / R2 events/s | Candidate R1 / R2 events/s | Baseline median loop s | Candidate median loop s | Candidate time change |
|---|---:|---:|---:|---:|---:|
| Cold | 120.16 / 124.31 | 130.03 / 129.26 | 8.18329 | 7.71360 | -5.7% |
| Warm | 180.68 / 179.66 | 169.66 / 176.03 | 5.55035 | 5.78750 | +4.3% |

The paired pipeline samples have identical 29 execution subbatches, 29 each
walk/init/locate/gather launches, 1,000 calibration launches, 750 allocation
reservations, and 7,675,234,816 peak charged bytes. This supports unchanged
allocation and execution behavior for the tested workload; it does not prove
all possible retained-view or tight-budget schedules equivalent.

Both real-workload profiles make 31 `plan_slots` calls:

| Cache | Baseline cumulative seconds | Candidate cumulative seconds | Reduction |
|---|---:|---:|---:|
| Cold | 0.58561 | 0.03928 | 93.3% |
| Warm | 0.59676 | 0.03758 | 93.7% |

This confirms a reduction in the selected CPU cost. Profiled loop times are
excluded from throughput decisions because instrumentation can change overlap
and waiting; the primary unprofiled warm controls regressed despite the
profile improvement. The additional warm controls address that discrepancy.

## Reproduction and evidence

Shared scratch campaign:
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-slot-index-20260925`.

- `baseline/python`, `candidate/python`, `builds.json`: frozen variants and
  complete hashes; `baseline.patch`, `candidate.patch` record working-tree state.
- `run.sbatch`: full environment and exact launch. It runs `device.py` first,
  then `acceptance.py --study slots`; device failure stops the campaign.
- `device-39070635.log`, `test-hashes.json`, `tests/`: device test evidence.
- `job-39070635.log`, `job-39070635/results.json`, `summary.md`, `summary.json`,
  `provenance.json`, and `*.pstats`: samples, audits, and call graphs.

The [step plan](../stream_read_slot_selection_plan.md) describes the decision
gate. Pending-file scanning remains the next isolated optimization candidate;
broader Stage 4 and long-run acceptance gates are separate.

## Warm follow-up

The first two warm control pairs showed a 4.3% median loop-time regression,
while cold candidate rates were consistently higher. Job **39071037** ran
four additional alternating warm pairs, all with bulk on, using the exact
frozen builds and shared references. Its separate allocation is analyzed as
its own paired study.

| Round | Order | Baseline events/s | Candidate events/s |
|---|---|---:|---:|
| 1 | Baseline, candidate | 176.47 | 185.82 |
| 2 | Candidate, baseline | 181.81 | 186.85 |
| 3 | Baseline, candidate | 181.07 | 199.88 |
| 4 | Candidate, baseline | 176.27 | 193.29 |

Median loop time was **5.59472 s baseline vs 5.26283 s candidate**, a **5.9%
reduction**. All four pairs favor the candidate. The equivalent rates from
these median times are 178.74 vs 190.01 events/s. This follow-up contains
controls only; pipeline equivalence was checked separately in the primary
campaign.

The selected CPU overhead is substantially reduced, exact slot choices and
tested ownership behavior are preserved, and cold controls plus the repeated
warm study support a throughput benefit. The initial warm regression remains
part of the evidence: these runs do not establish a consistent warm speedup
across allocations. Keep the isolated implementation for review, with broader
Stage 4/10k acceptance and any general throughput claim still outstanding.

Follow-up root:
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-slot-index-warm-repeat-20260925`.
Its `run.sbatch`, `job-39071037.log`, and `job-39071037/{results.json,summary.json,provenance.json}`
record the full settings, samples, and final audits. The current implementation
matches the frozen candidate SHA-256
`354152ece8df1571b9ea5510ccbf2f0c1f06ff5952258a854b21faea99529504`.
