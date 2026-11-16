# Ampere 013 frozen-reference qualification

Node qualification preceding the [PR187/208 performance study](../docs/performance/pr187_pr208.md).
**Completed successfully.** Job **42001989** ran on 013 from 08:46:06 to
09:50:57 PDT on October 26 (1:04:51), exit 0. Follow-up job **42004160** completed
on 042 in five seconds, exit 0, and reported PASS. All 16 timed samples, both
pixel preflights, cache gates, dependency checks and 2,175 frozen hashes passed.
The job recorded removal of its private staged data.

## Results and decision

Four samples per condition. SD is the sample standard deviation (ddof=1).

| Cache | Bulk | Individual events/s | Mean events/s | SD | 10,000 / median loop |
| --- | --- | --- | ---: | ---: | ---: |
| Cold | Off | 315.09, 322.74, 322.45, 321.21 | 320.37 | 3.58 | 321.83 |
| Cold | On | 317.15, 318.59, 318.90, 322.21 | 319.22 | 2.14 | 318.75 |
| Warm | Off | 400.32, 428.53, 419.74, 407.98 | 414.14 | 12.48 | 413.77 |
| Warm | On | 403.63, 408.30, 416.52, 406.17 | 408.66 | 5.58 | 407.23 |

**013 is suitable for the planned same-node PR comparisons.** It delivers the
desired roughly 300-event/s cold profile with valid cold/warm separation. Warm
bulk-off spans 7.04% from minimum to maximum; paired, interleaved comparisons
and additional repetitions if needed remain necessary for a 5% regression
decision. This qualification establishes neither PR performance nor a merge pass.
The backup node was not requalified in this job.

Final evidence review:

- Cold prefix residency before each sample was at most 0.000352%, below the
  1% gate. All warm prefixes were 100% resident before and after each sample.
- Disk snapshots after cache preparation and around each sample show
  **335.601–335.608 GB** read from md0 for every cold sample, split evenly
  across nvme1n1/nvme2n1. All eight warm samples show **zero bytes read** from
  md0 and both member SSDs during these intervals.
- Every timed sample validated 10,000 unique events, 50,000 requests and
  335,571,760,000 payload bytes. Each BD reported four GPU peers and a
  6 GiB budget. Runtime versions were CuPy 14.0.1 / KvikIO 26.10.00.
- Fresh inventory confirms the two KCM6XRUL7T68 SSDs, firmware 0106, RAID0,
  512 KiB md request limit and 4 MiB read-ahead. GPU 0 is A100-SXM4-40GB at
  PCI 01:00.0, Gen4 x16, matching the selected hardware profile.

The automatic follow-up saved `followup/result.md` and `followup/result.json`
under the campaign root. The manual disk-counter and variability review above
was completed at approximately 10:00 PDT.

## Frozen workload

| Setting | Value |
| --- | --- |
| Node | sdfampere013, exclusive |
| Reservation | 112 CPUs, 700 GiB RAM, all four A100 GPUs, 90 minutes |
| Active topology | 1 Smd0 + 1 EB + 4 BDs; all BDs use physical GPU 0 |
| Reference commit | `410ebba1ce4d8b2a9d8d7fab56f6dc1fa26a6ef8` |
| Reference source tree | `65b8d582a61a96cafd2e91f03c52b5c910e5e455` |
| Data | mfx101210926 run 387; 32 Jungfrau panels |
| Input | Five hash-verified private prefixes; 335,656,487,644 bytes staged |
| Timed sample | 10,000 unique events; 335,571,760,000 payload bytes; 50,000 requests |
| Batch / depth | 20 / 1 |
| KvikIO | 26.10; CPU compatibility mode; eight threads per BD; 1 MiB target |
| Device budget | Explicit 6 GiB per BD, 24 GiB aggregate |
| Workload | Dense raw gathering; no calibration or user kernel |
| Conditions | Cold/warm × bulk off/on; four repetitions each, 16 timed samples |
| Preflights | Two 200-event pixel checks, bulk off and on |

The native installation and benchmark worker/adapters are reused unchanged from
the frozen October 22 post-stack campaign. This includes the zero-shift N+1
reader-allocation wrapper. Only the driver schedule, cache preparation and
provenance collection were adapted. This job does not test PR187 or PR208.
The old artifacts remain unchanged; writable caches and temporary files belong
to the new campaign.

Odd repetitions run cold then warm, bulk off then on; even repetitions reverse
both orders. Each sample runs in fresh MPI processes. The pixel preflights
exercise the kernels before timed samples, using the same campaign CUDA cache.
Timings retain setup separately and include lazy work and pipeline cleanup in
the event loop, as in the reference benchmark.

## Gates and evidence

- Verify 2,175 frozen file hashes before and after the campaign, including native
  installations, dependency files, original harness and new qualification driver.
- Verify Python/native import origins on all six ranks before staging, and in
  each sample. Check package and loaded-library fingerprints before/after.
- Stage fresh copies under `/lscratch/monarin/jf-qualification-42001989`; validate
  prefix sizes and SHA-256 hashes. Check free space before copying. Normal driver
  completion or failure removes this job's stage in `finally`.
- Cold: fsync and DONTNEED only job-owned private files; every measured prefix
  must be <=1% resident before each timed sample.
- Warm: reset the private copies, then load measured prefixes with NUMA
  interleaving. Allow at most three missing-page repair passes before timing.
  Require every prefix >=99% resident before and after timing; never repair
  after timing. Failed gates invalidate the run.
- Validate event/timestamp identity, payload/request counts, all four active BDs,
  GPU identity, budgets and drained resources. Pixel preflights compare against
  the existing CPU reference.
- Record hardware inventory, GPU memory/utilization/clocks, CPU affinity,
  memory/cgroup state, NUMA information, disk counters and network traffic.
  Disk snapshots bracket each sample, after cache preparation.
- Summarize individual rates, loop times and payload bandwidth with means,
  medians and sample standard deviations (ddof=1). Review disk-counter deltas
  and variability before declaring this node suitable for comparisons.

Historical context: the same frozen post-stack reference produced cold/off
369.64 events/s and cold/on 367.40 events/s on 012 (job 41631223), versus
177.17/176.96 on 027 (job 41614567). These are 10,000 divided by median loop
seconds. Node 013 shares 012's SSD model/firmware but has different block-request
limits, so matching those absolute rates is not assumed or an acceptance gate.

## Artifacts and validation

Campaign root:
`/sdf/scratch/users/m/monarin/gpu-validation/ampere013-qualification-20261026-r1`

| Artifact | Location under campaign root |
| --- | --- |
| Submission record | `submission.json` |
| Batch script | `run.sbatch` |
| Driver and frozen plan | `screen/qualify.py`, `screen/plan.json` |
| Frozen hash manifest | `screen/hashes.json` |
| Preparation recipe | `metadata/prepare.py` |
| Driver / benchmark logs | `logs/driver-42001989.log`, `logs/screen-42001989.log` |
| Fresh hardware inventory | `logs/inventory-42001989.json` |
| Results after execution | `screen/job-42001989/{provenance,results,summary}.json` |

Before submission: shell syntax, Slurm request validation, all 2,175 hashes and
the 16-sample schedule passed. Runtime validation subsequently passed as recorded
above.

Completion review confirmed `provenance.json` has `complete: true`, the two
pixel preflights and all 16 timed samples passed, and all cache gates and
before/after hash and dependency checks succeeded. The measured summary is
preserved above. Subsequent PR integration comparisons completed on Ada and
Perlmutter; see the [completed report](../docs/performance/pr187_pr208.md).
