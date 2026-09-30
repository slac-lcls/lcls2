# A / B / B+ / B+gather in one allocation

Requested matched rerun after separate B+/gather measurements. Job 38564718.
A = f52e90cc66; B = 803a70011d; O = B+ b0c9c3c02; G = the validated,
uncommitted canonical-gather implementation based on B+. No production edits.
`builds.json` records the original installation paths. `installs/{A,B,O,G}`
contains independent frozen copies, checked against previous recorded hashes.
Installed source and native-extension hashes are recorded again by this run.

One A100 allocation, 48 requested CPUs and 450 GiB host memory. Each variant
uses identical 10,000-event mfx101210926/run-387 data, 32 Jungfrau segments,
batch 20, depth 1, 8 GiB GPU budget, three MPI ranks with one BD, eight KvikIO
threads, 1 MiB tasks and compatibility ON (CPU fallback). No user D2H or bulk
integration. Calibration is the same frozen CPU-reference snapshot.

Four clean repetitions use orders ABGO, BOAG, OGBA, GAOB. Every variant appears
once in each position, and all twelve ordered adjacent pairs occur once.
Four fresh CPU-reference preflights must pass. No timing hooks or new profiler
captures are enabled. Prior launch-count traces remain separate evidence.

The event-loop timer is unchanged from the earlier comparisons: detector setup
and final GPU synchronization are included. Each sample follows a 100-event
warmup; input staging and cache preparation are outside the timer. Each sample
must have >=99% file-cache residency both before and after measurement.
Every rank receives the same allocation CPU mask, not exclusive per-rank or
GPU-local NUMA placement. GPU identity, runtime versions, NUMA page samples,
CPU affinity, clocks/power, device memory, and host counters are recorded.

```bash
sbatch validation/four-way-20260918/run.sbatch
python validation/four-way-20260918/audit.py validation/four-way-20260918/job-JOBID-warm-ab
python validation/four-way-20260918/summarize.py validation/four-way-20260918/job-JOBID-warm-ab
```

The audit checks all 16 samples, four preflights, workload/order/cache guards,
placement, import paths and installed hashes. Report every repetition and
median/range; do not substitute samples from other allocations. Ten harness
AST/timing checks pass. A copied test initially referred to the old baseline
path; correcting it to the new frozen prefixes resolved its two missing-file
failures. No benchmark or production behavior changed in that correction.

Completed: job **38564718**, sdfampere004, COMPLETED / 0:0, elapsed 57:46.
All 16 samples and four preflights passed audit; all 32 cache checks were 100%.
Medians (seconds): A 28.338250, B 39.063461, B+ 32.387702, B+gather 29.806849.
Gathering reduces B+ median elapsed by 8.0%; its median is 5.2% above A.
Ranges overlap and round 2 does not favor gathering; see all repetitions in
`psana/psana/gpu/docs/performance/four_way_sdf.md`. No timing samples from earlier
allocations are combined here. No production changes, commits or pushes.
