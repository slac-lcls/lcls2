# Direct bulk-on group submission

2026-09-25. Implementation, CPU validation, and A100 correctness complete.
Initial job **39077041** on **sdfampere018** stopped at the baseline warm
profile's premeasurement cache gate (97.53% residency for s005, below 99%).
It has eight validated controls and two cold profiles, but no final audits
or paired pipeline result. Full retry **39078469** completed on
**sdfampere026** in **16m51s**, exit `0:0`, with all **14 samples**, final
hash/placement audits, and exact pipeline equivalence passing. It reused the
same frozen runtimes and kept every validation gate unchanged. The candidate
reduces group-submission cost but is not accepted as a throughput improvement.

## Change

The baseline group adapter converts resolved dgrams to `GpuReadDesc` objects,
builds a temporary view and file-epoch map, then invokes `issue_batch`, which
constructs resolved dgrams again and sorts/coalesces them into a legacy plan.
Every admitted group already contains one contiguous range from one stream
and file inside one transition fence.

The candidate builds the descriptor table, one physical range (none for an
empty row), and group-relative offsets directly. It retains lightweight
`ReadPlan`/`LogicalDgram` metadata for diagnostics without invoking the generic
planner. Rows keep their original order. Each group is submitted separately;
adjacent groups across file or transition boundaries are not combined.

`validate_read_descriptors` is extracted from the generic planner and shared
with the direct path. It checks record types, duplicate event/stream rows,
timestamp consistency, size limits, and total bytes. Immutable `ResolvedDgram`
construction continues to validate field types, uint64 ranges, the stream mask,
and file-end overflow. The group path additionally validates normalized group
size/offset, exact contiguity, matching file/stream, singleton zero rows, and
the declared total. Validation precedes allocation and submission.

Both paths use `_check_slot_available` and `_submit_read` for slot/input-owner
guards, allocation, generation assignment, handle accounting, pread submission,
and partial-failure draining. Bulk-off planning and behavior remain unchanged.
The EB stream planner still uses generic planning to validate overlap once
per batch; this change removes repeated per-group legacy planning only.

Maintained native-trace, read-count, phase-timing, and request-summary tools
now observe shared submission. Older frozen builds retain an `issue_batch`
fallback. Submission diagnostic regions exclude preparation in the new build,
so those regions alone are not comparable to historical timings. Controls use
unchanged native counters and complete measured-loop timings.

## Validation

- **435 CPU unit tests passed**, including 102 deterministic direct-versus-
  legacy comparisons of descriptor rows, plan metadata, requests, and bytes;
  malformed-group rejection; empty rows; file/transition separation; busy/input-
  held slots; full replacement cost; bulk-off rejection; and native trace
  coordination through the new submission hook.
- **20 harness tests passed**, including profile attribution, cold/warm repeat
  matrices, and control/diagnostic separation for this study.
- CPU imports explicitly use the frozen candidate prefix, preventing an
  installed reader from masking source changes.
- **20 A100 tests passed**: existing group pixel/launch, delayed consumer, parser
  failure, transition, multi-owner calibration, KvikIO byte-parity, and eight
  real parser/gather lifecycle cases (early stop, generator close, read failure,
  gather failure, with and without D2H), plus BeginStep/EndRun drain ordering.

## Isolated measurement

Baseline commit **`c3357e622`** includes the previously committed slot index
and pending-file accounting, plus the existing uncommitted transition-drain
and target-parameter fixes in both snapshots. Frozen manifests differ only
in `psana/gpu/gpu_kvikio_read.py` and `psana/gpu/gpu_read_plan.py`.

One A100, one BD/three MPI ranks, 1,000 JF+feespec events, batch 100, depth 1,
8 GiB, eight KvikIO workers, compatibility mode ON, and 4 MiB target/task size.
Eight controls cover both builds and cold/warm caches in two reversed-order
rounds. Four separate profiles and two warm pipeline diagnostics follow.
Only controls determine throughput. Preserve array/checksum, 5,019 request,
33,566,911,424 payload-byte, residency, cold-NIC, code-hash, and Weka FFB
placement gates, with exact paired pipeline/charged-memory comparison.

The first candidate cold control passed audits but ran at 63.99 events/s,
versus 122.05 baseline. The first warm pair was 177.83 baseline vs 180.20
candidate. GPU monitoring shows clock variation in both builds and does not
establish the cause of the slow cold sample. The sample remains in the record.
Four alternating cold pairs were queued as **39077775**. That job was canceled
without running when its successful-primary dependency became impossible;
the full retry replaces it. The cache-gate failure does not establish the
cause of the earlier cold slowdown.

## Initial controls (incomplete campaign)

| Cache | Baseline R1 / R2 events/s | Candidate R1 / R2 events/s | Baseline median loop s | Candidate median loop s | Candidate time change |
|---|---:|---:|---:|---:|---:|
| Cold | 122.05 / 122.19 | 63.99 / 127.54 | 8.18863 | 11.73379 | +43.29% |
| Warm | 177.83 / 180.39 | 180.20 / 177.57 | 5.58340 | 5.59054 | +0.13% |

All eight controls passed their audits. The second candidate cold run did not
repeat the first run's slowdown, but the first run remains in the median.
The warm controls are effectively flat. Neither cache establishes a repeatable
throughput improvement from these two pairs.

Cold profiles from this attempt show `_coalesced_plan` calls falling from
5,019 to zero. Total `issue_group` cumulative time fell from 0.810751 s to
0.638624 s (21.2%); generic `build_read_plan` calls fell from 5,029 to 10,
retaining only EB-level validation. Profile timings are diagnostic, not
throughput controls.

## Retry controls

Job **39078469**, **sdfampere026**, same frozen runtimes and all original gates:

| Cache | Baseline R1 / R2 events/s | Candidate R1 / R2 events/s | Baseline median loop s | Candidate median loop s | Candidate time change |
|---|---:|---:|---:|---:|---:|
| Cold | 109.29 / 115.87 | 115.59 / 114.25 | 8.89014 | 8.70194 | -2.12% |
| Warm | 172.97 / 168.11 | 151.55 / 166.91 | 5.86499 | 6.29481 | +7.33% |

All eight controls passed. Cold pairs disagree; both warm pairs favor the
baseline, with the larger difference in round one. The first attempt's cold
slowdown did not recur in these controls. Neither discarding that earlier
sample nor pooling allocations would establish a general throughput win.

Paired pipeline diagnostics match exactly: all 29 subbatch sizes, 29 each
walk/init/locate/gather launches, 1,000 calibration launches, 750 allocation
reservations, and **7,675,234,816 peak charged bytes**. Every sample retains
5,019 read requests and 33,566,911,424 payload bytes. These checks establish
equivalence for the measured workload, not all possible lifetime schedules.

## Retry profile attribution and decision

Both builds make 5,019 `issue_group` calls. Baseline makes 5,019
`_coalesced_plan` calls; candidate makes zero. The remaining ten generic
`build_read_plan` calls in the candidate perform EB-level validation.

| Cache | Baseline issue_group cumulative s | Candidate cumulative s | Reduction | Baseline legacy replanning s |
|---|---:|---:|---:|---:|
| Cold | 0.894147 | 0.661904 | 26.0% | 0.253145 |
| Warm | 0.804458 | 0.536666 | 33.3% | 0.233689 |

These times include actual submission and native calls, not only Python
planning. Cumulative times overlap and are not additive. The warm profiled
loop was 6.345 s baseline vs 7.226 s candidate; `cufile.get` self time rose
from 1.476 s to 2.053 s with the same 5,019 calls. Some calibration/preparation
CPU self times also increased. This locates costs outside the removed planner
but does not identify a causal explanation for the throughput regression.

The implementation removes the targeted per-group work and passes correctness.
It is **not accepted as a throughput improvement**: warm controls regressed
7.33% in the completed control matrix, and cold pairs disagree despite a
2.12% lower median. The candidate is retained as a validated reduction in
per-group planning work;
do not treat it as a completed broader Stage 4/10k performance gate. The
previous pending-file cleanup is committed independently as `c3357e622`.

## Evidence

Scratch campaign:
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-direct-group-20260925`.

- `builds.json`, `baseline/python`, `candidate/python`: frozen variants/hashes.
- `baseline.patch`, `candidate.patch`: worktree state at freeze time.
- `run.sbatch`, `device.py`, `tests/`, `test-hashes.json`: launch and device gate.
- `device-39077041.log`, `job-39077041.log`: test and campaign logs.
- `job-39077041/{results.json,summary.md,summary.json,provenance.json}` and
  `*.pstats`: samples, full call graphs, and audits.

Launch uses `acceptance.py --study groups`. CPU tests use the same activated
environment and frozen candidate import procedure documented in the
[pending-file report](stream_read_file_refs.md), with the additional direct
submission tests and updated phase-timing test.

Canceled cold-repeat preparation (no samples):
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-direct-group-cold-repeat-20260925`.
The repeat uses `--study groups-cold`; its harness is frozen separately from
the immutable primary campaign. Controls have the same correctness, cache,
NIC, and provenance gates.

Full retry root:
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-direct-group-retry-20260925`.
It runs `--study groups`, references the original frozen baseline/candidate
prefixes, and skips repeating the already-passed A100 suite. Its logs/results
use job **39078469**. The first attempt and its failure evidence are retained.

Current runtime SHA-256 hashes match the tested candidate:

- `gpu_kvikio_read.py`: `9a43193f6720b896e6f3b1c3522891054a4dceac61c2d6c46fafb21f40efa13c`
- `gpu_read_plan.py`: `72de7118c3c3bf3dbfaaaacb4bcc1dd51e0845d1cbf951c91e2aaae79017a441`
