# Issue 159: staged warm-cache preparation diagnosis

Status: cache-only diagnostic job **41398320** completed on `sdfampere013`
in **13m17s**, exit `0:0`. A one-time private-file cache reset followed by the
existing warmer passed the residency checks. The remedy is now integrated in
[screen retry 41400127](issue159_stage3_cache_reset_retry_20261020.md), running
on `sdfampere030`. Stage 3 performance acceptance remains pending.

## Observed failure

Both clean-build screen attempts passed native provenance and all four
200-event pixel/address preflights, but failed before the first timed sample:

| Job | Node | First reported failing prefix | Residency after three repairs |
| --- | --- | --- | ---: |
| 41397249 | sdfampere032 | s005 | 98.504599% |
| 41397802 | sdfampere013 | s006 | 98.972597% |

The latter means 184,126 missing 4 KiB pages, approximately 719 MiB, in a
73.4 GB measured prefix. It is not a percentage-rounding problem. Other files
also lost substantial residency between repair passes. The 99% threshold
applies separately to every file, before and after each timed sample.

Both jobs removed their private stages. Neither produced timed performance
results. Raw logs remain under
`/sdf/scratch/users/m/monarin/gpu-validation/issue159/stage3-20261020-r2/`.
The [diagnostic evidence](evidence/issue159_cache_diagnosis_20261020.json)
retains residency history and job memory/reclaim/NUMA snapshots.

## Evidence and working explanation

- Both allocations had a 700 GiB effective job memory limit. The recorded
  warming snapshots peaked near 367 GiB and 422 GiB, respectively. All job
  memory-high, memory-max and OOM event counters remained zero.
- During warming/repair, the jobs reclaimed approximately 388 GiB and 132 GiB
  of pages respectively. These counters cover the whole job, including cached
  source data; they are not a direct count of evicted destination-file bytes.
- File-cache growth was concentrated on particular NUMA nodes despite the
  warmer reporting `interleave` over nodes 0–7. On the first node, its job
  file-cache charge on NUMA node 2 grew from 68.7 GiB to 103.1 GiB while several
  other nodes barely changed.
- Staging uses ordinary parallel buffered copies, which allocate source and
  destination cache pages before the separate interleaved reader runs.
  The later warm reads preserve already-resident pages in their original
  locations. Linux task NUMA policy governs new allocations; it does not
  relocate existing pages merely because the task policy changes.
  See [Linux NUMA memory policy](https://www.kernel.org/doc/html/latest/admin-guide/mm/numa_memory_policy.html).
- Low node-wide free memory alone is insufficient to explain failure: the
  successful Stage 2 screen also had low free memory and abundant reclaimable
  cache. NUMA placement and reclamation behavior need to be tested directly.

The working explanation is unstable cache placement under reclaim, rather
than a Stage 3 device allocation or a job memory-limit failure. The counters
do not by themselves prove the exact kernel allocation/reclaim decision.

## Controlled probe

Artifact root:
`/sdf/scratch/users/m/monarin/gpu-validation/issue159/cache-diagnosis-20261020-r1/`.

Job 41398320 reserves an exclusive node with the same 700 GiB memory request.
It runs no timed GPU workload and leaves the frozen A/B installs and harness
unchanged. Its private files use the same five staged extents and SHA256 checks.

1. Measure residency after staging, then after the existing interleaved warmer.
2. Flush and discard cached pages for only the probe's private staged files;
   verify less than 1% residency. Run the existing interleaved warmer again.
3. If that reaches 99% per file, verify after an idle interval and another
   complete warm read.
4. If it fails, reset the private files again and assign disjoint 64 MiB
   file ranges round-robin to workers bound to individual allowed NUMA nodes.
   Disable read-ahead in those workers to limit allocation across range
   boundaries. Check all files after completion, after idle, and after another
   pass of the existing warmer.
5. Preserve snapshots/results and remove the probe's private stage.

The initial probe reproduced the problem: immediately after staging, only
39.66–56.02% of each measured file prefix is resident. Job memory is about
272 GiB, with roughly 116 GiB of file cache on one NUMA node and effectively
none on another.

## Measured result

All stages below used the same files, node, allocation, and existing sequential
interleaved warmer. The reset used `fsync` plus `POSIX_FADV_DONTNEED` on only
the five private files, followed by a measured cold-residency check.

| Preparation/check | s005 | s006 | s007 | s008 | s009 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Immediately after staging | 47.11178% | 56.01664% | 39.65642% | 54.06540% | 54.60018% |
| Existing warmer, no reset | 81.51936% | 92.70403% | 94.00593% | 95.55701% | 100% |
| Private-file cache reset | 0% | 0% | 0% | 0% | 0% |
| Existing warmer after reset | **99.70317%** | **100%** | **100%** | **100%** | **100%** |
| Ten-second idle check | **99.70317%** | **100%** | **100%** | **100%** | **100%** |
| Another warm pass, no reset | **99.98632%** | **100%** | **100%** | **100%** | **100%** |

Warm-reader times were 172.8 seconds before reset, 202.7 seconds for the
clean rewarm, and 70.4 seconds for the repeat pass. These are preparation
times, not detector loop times or A/B throughput results. The explicit
NUMA-worker alternative was not run because the simpler remedy passed.

The private reset reduced the job's file cache by approximately 291 GiB.
Aggregate NUMA skew persisted after rewarming: the successful state still had
about 116 GiB of job file cache on the most loaded node. Thus the probe
supports **resetting the staging-created cache state**, but does not prove
that successful NUMA rebalancing caused the recovery. Existing cache placement,
source/destination cache competition, and page reclaim history remain plausible
parts of the mechanism; the exact kernel decision is not established here.

The demonstrated failure is in cache preparation, before timed Stage 3 work:
ordinary staging followed by read-only warming did not reliably establish the
required working set. A private reset followed by the same warmer did, on the
same node where the screen failed.

The diagnostic removed `/lscratch/monarin/jf-cache-probe-41398320`.
Its script/helper/reference hashes verified unchanged. All 1,973 frozen screen
input hashes also verified after failed screen 41397802.

## Recommended harness correction

1. After staging/hash checks and pixel preflights have finished, reset only the
   job's five private staged files once, before its first timed warm-up.
   Keep the operation explicitly scoped to the private stage; do not change the
   generic warmer to discard arbitrary input files.
2. Verify cold residency, then use the existing interleaved warmer. Preserve
   the per-file 99% gates before and after each timed sample. Later samples
   use ordinary warm preparation without another reset.
3. Record the reset and memory snapshots as harness provenance, freeze the
   updated harness, and rerun the same eight-sample screen with identical
   preparation for A and B.

This adds untimed setup work and leaves event count, measured data, runtime
builds, GPU/CPU bindings of timed workers, and the benchmark loop unchanged.
The real screen is still needed to validate residency through GPU processing.

No system-wide cache drop, shared-source eviction, mlock, threshold relaxation,
or reduction in event count is part of this diagnostic. A successful
preparation method still requires validation in the real staged screen with
unchanged pre/post residency gates and identical preparation for A and B.
