# Issue 159 stage 3: isolated-install repair and screen retry

Status: isolated-install repair and focused validation passed. Job **41397249**
on `sdfampere032` stopped at the warm-cache gate before timing. Screen-only retry
**41397802** also stopped at the cache gate on `sdfampere013`; performance
acceptance is pending. See the [cache diagnosis](issue159_cache_diagnosis_20261020.md).
The correction is applied in [retry 41400127](issue159_stage3_cache_reset_retry_20261020.md),
running on `sdfampere030`.
The runtime source is still stage-3 commit `f425e601e450a292cda440fe9b67820b5dc7b4eb`.

## Failure and repair

Job **41349323** failed on `sdfampere040` after 5m12s, before any timed samples.
Its 89 CPU tests, 21 CUDA tests and three memcheck tests passed; the staged A
pixel preflight passed. B failed the native-library provenance assertion.

The copied B install retained Stage 2's absolute `DT_RPATH` in 35 of its 36 ELF
files. Its Python modules came from B, but its extensions loaded `libxtc.so`
and `libshmemcli.so` from Stage 2. Fresh A/B/A processes reproduced the mismatch.
The copied binaries were byte-identical to Stage 2's; only `gpu_input.py` differed
between those installs. This explains why correctness checks passed, but those
checks did not establish a self-contained B runtime. The earlier statement that
copying this install needed no native rebuild was incorrect for this benchmark's
isolation contract.

The repair builds the exact Stage 3 commit into a new scratch prefix using the
same compiler configuration, dependency environment and release settings as
Stage 2. The clean Meson build took **96.5 seconds**. No runtime logic, PR #187
code, global linker settings or frozen baseline was changed.

The shared [runtime provenance check](../scripts/check_runtime_provenance.py)
now verifies both imported psana modules and actually mapped project libraries.
The CPU test runner, MPI preflight and per-sample benchmark use that same check.
The six-rank MPI preflight runs for both variants **before data staging**.
It preserves the existing per-sample provenance assertions and adds native
library hashes to the evidence.

## Checks before submission

- Fresh A/B/A processes pass module and native-library isolation checks.
- The same checker rejects the original copied B install with the expected
  outside-prefix native-library error.
- All 36 candidate ELF files were audited; none references either earlier
  validation install. The 35 rebuilt files have the new install/approved
  dependency paths. The remaining file is the unchanged, shipped Python 3.7
  momentum helper `CalcPzArr`, which is not loaded by this Python 3.11 workload.
- Meson's recorded compiler configuration matches the frozen Stage 2 build.
  Installed `gpu_input.py` matches the exact Stage 3 source commit.
- The A/B benchmark harnesses are identical, syntax checks pass, and the schedule
  verifies four balanced pairs/eight warm samples.
- **1,973 runtime/test/harness/reference hashes** are frozen for verification
  before and after validation and the staged screen.

## Retry protocol

Artifact root:
`/sdf/scratch/users/m/monarin/gpu-validation/issue159/stage3-20261020-r2/`.

`A/install` resolves to the unchanged Stage 2 build at `7f29f0dc7`; `B/install`
is the clean build of `f425e601e`. Build commands, compiler audit, import checks,
hashes and failed-install negative control are retained under `metadata/`.
The earlier failed attempt remains under `stage3-20261019-r1/`.

Job 41397249 repeated the focused CPU/CUDA/memcheck groups and host bookkeeping
timings, then attempted the eight-sample warm screen on one exclusive node:

- Four BDs, one physical A100, batch 20, depth 1, KvikIO CPU fallback,
  eight workers, 1 MiB tasks and 6 GiB/BD.
- Four 200-event pixel/address preflights followed by 10,000 events per timed run.
- Bulk off/on, two balanced Stage 2/Stage 3 pairs each, reversing order in pair 2.
- Same staged Jungfrau prefixes, hashes, residency, timestamp/read/byte,
  memory-budget/drain and GPU identity checks as the original screen.

## Validation and cache-gated attempt

Job **41397249** passed **89 CPU tests, 21 CUDA tests and three Compute Sanitizer
tests**, with no failures or skips and zero sanitizer errors. Both six-rank
native-library preflights passed, followed by all four 200-event pixel/address
preflights (Stage 2/Stage 3, bulk off/on). Thus the rebuilt candidate fixes the
original native-library isolation failure in the actual MPI workload.

The first timed sample never started. After the initial warm pass and three
bounded repair passes, the s005 measured prefix was **98.5046%** resident, below
the unchanged **99% per-file** gate. Warming later files repeatedly evicted pages
from earlier files. The captured job cgroup had a 700 GiB limit, used about
367 GiB at the first inspection, and reported no limit or OOM events; reclaim
counters were substantial despite ample node-wide reclaimable cache. This
identifies a cache-preparation failure; it does not establish the underlying
kernel/cache-placement cause or measure a Stage 3 performance regression.

The job ended `FAILED`, exit `1:0`, after **16m14s**. Its private stage
`/lscratch/monarin/jf-issue159-41397249` was removed. All **1,973** frozen-input
hashes were checked again after failure and still match. Logs, memory/NUMA
snapshots, cgroup counters and NIC traces remain in the scratch artifact root.
The requested live time-limit extension was declined by Slurm; the job actually
ended at the residency gate well before its original 45-minute limit.

Host-only bookkeeping microbenchmarks also completed (four balanced process
pairs, five 20,000-event repetitions per mode/process, 2,000-event warmup):

| Mode | Stage 2 median µs/event | Stage 3 median µs/event | Median paired difference |
| --- | ---: | ---: | ---: |
| No child view | 11.395 | 11.358 | +0.128 µs (+1.13%) |
| Child view, no retained copy | 17.788 | 18.287 | +0.610 µs (+3.48%) |

The paired percentage ranges were −8.07% to +1.39% and +0.83% to +6.92%,
respectively. Independent medians and paired medians need not have the same
sign. These noisy host-only measurements do not replace the staged screen.
There are **zero accepted timed samples**, so no throughput or loop-time
comparison is reported for this attempt.

The [compact evidence](evidence/issue159_stage3_rebuild_20261020.json) preserves
build/import audits, validation results, host samples and the cache-gated outcome.

## Screen-only continuation

Job **41397802** uses `screen-retry.sbatch` in the same artifact root, with a
90-minute limit and `sdfampere032` excluded. It reuses the exact frozen installs,
identical A/B harnesses, staged-data hashes, and eight-sample schedule. Each job
has its own output directory and private node-local stage. Native and pixel
preflights still run; the already-passing CPU/CUDA/sanitizer groups are not
repeated. NIC monitoring now starts with the job.

After briefly queuing for resources, the retry ran on `sdfampere013`. Both
six-rank native preflights and all four pixel/address preflights passed.
It failed before timing after three cache-repair passes: s006 was 98.972597%
resident, below 99%. Its private stage was removed. The
[cache-only probe](issue159_cache_diagnosis_20261020.md) demonstrated that resetting
the private staged-file cache once, then using the existing warmer, passes
the residency gate. That preparation correction is applied in retry 41400127;
the frozen installs and residency threshold are unchanged.
Pending: eight timed warm samples, paired events/s and loop seconds, post-run
hashes and stage cleanup. This incremental comparison does not replace the
final cumulative comparison with frozen integrated A. The inherited broad-suite
failures remain tracked; Stage 4 has not started.
