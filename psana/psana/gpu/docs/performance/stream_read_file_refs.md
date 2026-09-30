# Bulk-on pending-file ownership accounting

2026-09-25. Implementation, CPU correctness, A100 correctness, and the primary
isolated bulk-on comparison are complete. Job **39072327** on **sdfampere004**
completed in **14m24s**, exit `0:0`, with all **14 samples**, final hash/placement
checks, and exact pipeline equivalence passing. Warm follow-up job **39072683**
completed all eight controls and final audits in **7m57s**, exit `0:0`, in a
new allocation on the same node. Cleanup CPU cost fell **87.5–91.7%**;
neither campaign demonstrates an end-to-end throughput improvement.

## Change and scope

`KvikioGpuReader` maintains a count for each file identity held by pending
bulk reads. Each acquired range handle contributes one reference before
`pread` submission. A batch releases all its references only after every
submitted future has been drained, even when a future fails or reports a
short read. Partial submission failures release acquired handles through the
same completion path. The existing `completed` guard prevents repeated waits
from releasing twice.

`_prune_files` retains the latest file for every stream plus identities with
outstanding references. It no longer rebuilds pending-file ownership by
walking every pending batch and its handles after each completion. It still
checks the open-file cache for obsolete handles. The pending-batch list,
slot ownership, allocation charges, read plans, and bulk-off scan are unchanged.

The baseline includes slot-selection commit `23600736f` plus the existing
working-tree transition-drain and bulk-target changes. Frozen manifests
verify the only runtime difference is `psana/gpu/gpu_kvikio_read.py`.

## Correctness

- **417 CPU unit tests passed**, including nine new cases covering shared
  old handles, duplicate ranges, out-of-order completion, repeated waits,
  partial submission, failed futures, short reads, file-open failure,
  zero-byte descriptors, shared latest-file identities, and close failure.
  Existing short-read/drain tests verify every started future is waited once
  and failed storage cannot be reused.
- **17 harness tests passed**, including cleanup profile attribution,
  diagnostic/control separation, bulk-on-only matrices, and pipeline mismatch
  rejection.
- **11 A100 tests passed** before the campaign: group scheduling and pixels,
  delayed/out-of-order consumers, partial parser failures, transition drains,
  multi-owner calibration, and actual KvikIO/parser byte parity. Bulk-off
  appears only in correctness parity checks; all throughput samples use bulk on.
- Tests ran with an explicit assertion that the reader imports from the
  frozen candidate prefix. An earlier attempt using a source-path override
  loaded the prior reader during psana initialization; that attempt is not
  candidate validation.

## Measurement

Job 39072327 ran the targeted device suite first, stopping on failure, then
14 samples on one A100 allocation: eight controls (cold/warm, baseline/candidate,
two reversed-order rounds), four separate profiles, and two warm pipeline
diagnostics. All performance samples use bulk on, 1,000 JF+feespec events,
batch 100, depth 1, 8 GiB, eight KvikIO workers, compatibility mode ON, and
4 MiB target/task size. One BD uses three MPI ranks.

The existing array/checksum, 5,019 API request, 33,566,911,424 byte, cache
residency, cold NIC, source hash, and Weka FFB placement checks remain active.
Performance claims use only controls. Reducing the historical ~0.40 s
profiled cleanup cost does not guarantee the same end-to-end time reduction.

## Primary controls

| Cache | Baseline R1 / R2 events/s | Candidate R1 / R2 events/s | Baseline median loop s | Candidate median loop s | Candidate time change |
|---|---:|---:|---:|---:|---:|
| Cold | 123.68 / 122.69 | 120.91 / 125.15 | 8.11779 | 8.13035 | +0.15% |
| Warm | 191.63 / 192.06 | 181.37 / 199.54 | 5.21259 | 5.26252 | +0.96% |

Each cache has one pair favoring each build. These controls do not demonstrate
a throughput improvement. Four additional alternating warm pairs ran as job
**39072683** after successful completion of the primary job. The repeat uses
the same frozen runtimes and is analyzed separately below.

The paired pipeline diagnostics have identical subbatch sizes (29 execution
subbatches), 29 each walk/init/locate/gather launches, 1,000 calibration
launches, 750 allocation reservations, and **7,675,234,816 peak charged bytes**.
Every sample has 5,019 API requests and 33,566,911,424 payload bytes. This
demonstrates equivalence for the tested workload; the broader Stage 4 and
long-run acceptance gates remain separate.

## Profile attribution

Both builds make 5,019 `_prune_files` calls in each measured profile.

| Cache | Baseline cumulative seconds | Candidate cumulative seconds | Reduction |
|---|---:|---:|---:|
| Cold | 0.518354 | 0.043073 | 91.7% |
| Warm | 0.397261 | 0.049526 | 87.5% |

These figures cover cleanup itself. Reference acquisition and release now
occur in `issue_batch` and `wait_batch`; their costs are included in the full
profiles and controls. Cumulative times overlap and must not be summed.
The profiled total loop times are nearly unchanged, and profiling overhead
can affect overlap. The primary unprofiled controls likewise do not establish
an end-to-end improvement despite the reduced selected CPU cost.

## Warm repeat and decision

| Round | Order | Baseline events/s | Candidate events/s |
|---|---|---:|---:|
| 1 | Baseline, candidate | 182.90 | 191.38 |
| 2 | Candidate, baseline | 196.50 | 196.40 |
| 3 | Baseline, candidate | 190.52 | 191.67 |
| 4 | Candidate, baseline | 196.89 | 193.71 |

Median loop times were **5.168864 s baseline vs 5.189810 s candidate**,
or **0.41% more time** for the candidate. Equivalent rates from these medians
are 193.47 vs 192.69 events/s. Two pairs favor each build, with round two
essentially tied. This repeat contains controls only; its pipeline diagnostic
status is unavailable, not a separate equivalence pass.

Correctness and the selected CPU-overhead reduction are established for the
tested cases. Throughput is effectively flat in these short campaigns: the
primary medians are +0.15% cold/+0.96% warm time, and the additional warm
median is +0.41%. These samples do not establish a statistically significant
regression or improvement. Retain the isolated implementation
for review as an ownership-accounting simplification; do not describe it as
an accepted throughput win or a completed broader Stage 4/10k gate.
Repeated legacy read-plan construction remains the third, separate item.

## Evidence and reproduction

Scratch root:
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-file-refs-20260925`.

- `builds.json`, `baseline/python`, `candidate/python`: frozen runtime hashes.
- `baseline.patch`, `candidate.patch`: worktree changes at freeze time.
- `tests/`, `test-hashes.json`, `device.py`: frozen device checks and hashes.
- `run.sbatch`: complete environment, device gate, and `--study files` launch.
- `job-39072327.log`, `device-39072327.log`: scheduler/device logs.
- `job-39072327/{results.json,summary.md,summary.json,provenance.json}` and
  `*.pstats`: sample results, final audits, and full profile call graphs.

Warm-repeat artifacts use
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-file-refs-warm-repeat-20260925`.
Its `job-39072683.log`, `run.sbatch`, and
`job-39072683/{results.json,summary.json,provenance.json}` record the repeat
and its final audits. Current reader SHA-256 matches the frozen candidate:
`02ecc35a25a609fd5233c7c13d3be4cdf5e60eb55d666e31dfb55bf75d369431`.

CPU validation uses `setup_env.sh`, the frozen candidate `PYTHONPATH`, shared
Integrated native libraries, `PS_PARALLEL=mpi`, `OMPI_MCA_btl=^smcuda`, and
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Preimport `psana`, assert the reader module
path, then invoke `pytest.main` on `psana/psana/tests/gpu/unit` and
`psana/psana/gpu/scripts/feespec_bulk_benchmark/test_harness.py`, with
`-q -p no:cacheprovider --tb=short`.
