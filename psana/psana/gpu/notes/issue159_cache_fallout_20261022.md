# Issue 159: cache-reset remedy fallout

The private-file reset executed correctly in Stage 4 job **41494246**, but the
subsequent warmer did not retain the whole working set. The remedy is useful,
not reliable across the nodes/states observed so far. Keep the 99% per-file gate;
there are no valid 10k-event A/B samples from this failed screen.

## Established facts

The job failed on **sdfampere004** after **12m28s**, exit `1:0`. Its 399 CPU,
323 CUDA and 305 sanitizer checks passed, as did all four 200-event pixel
preflights and the native provenance checks. The first timed sample stopped
in its untimed warm preparation. This failure does not identify a Stage 4
runtime regression.

The reset, warmer and memory-snapshot helpers are byte-identical to the
successful Stage 3 screen **41400127** on **sdfampere030**. Input manifests,
335,656,487,644 staged bytes, XFS mount options and allocated CPU affinity also
match. Thus an omitted reset or changed warmer is ruled out.

| Observation | Successful 41400127 | Failed 41494246 |
| --- | ---: | ---: |
| Aggregate private-prefix residency after reset | 0.00006835% | 0.00000244% |
| Job file-cache reduction during reset | 249.576 GiB | 161.046 GiB |
| Job memory after first rewarm | 542.240 GiB | 426.159 GiB |
| Effective job limit | 700 GiB | 700 GiB |
| Job pages reclaimed during first rewarm | 8.989 GiB | 52.536 GiB |
| First rewarm plus repairs: pages reclaimed | 8.989 GiB (no repairs) | 69.685 GiB |
| Memory-high/max/OOM events | All zero | All zero |
| First prepared working set | All files 100% | s005 78.37%, s007 96.75% |

The counters cover all memory charged to the job, including cached source data;
they are not a direct attribution of reclaimed bytes to private destination
files. The failed run's largest recorded memory usage during warm preparation
was **426.810 GiB**, and anonymous memory remained below 0.236 GiB in these
snapshots. Neither an exhausted job memory allowance nor accumulated anonymous
memory explains this failure.

Residency history in the failed run:

| Check | s005 | s006 | s007 | s008 | s009 |
| --- | ---: | ---: | ---: | ---: | ---: |
| First full rewarm | 78.36585% | 99.88252% | 96.74811% | 100% | 100% |
| After repair 1 | 97.39713% | 99.88252% | 99.29476% | 100% | 100% |
| After repair 2 | 100% | 99.88252% | 96.51449% | 100% | 100% |
| After repair 3, first failing file | 97.72437% | — | — | — | — |

The final exception records only the first failing file; remaining final
fractions were not persisted. `s005` was missing **349,567 pages**, about
**1.3335 GiB**. This is real cache churn, not percentage rounding. Raising the
retry count alone would continue the observed cycle without addressing it.

## What the reset did not solve

The reset removed resident pages from the private files, as intended. It did
not guarantee a stable allocation pattern when those pages were read back.
In the failed run, job file-cache charges after the initial rewarm ranged from
**1.38 GiB on NUMA node 7 to 117.20 GiB on node 3**. Most new cache growth was
concentrated on nodes 1, 2, 3 and 5 despite the reader reporting interleave over
all eight nodes. The successful run was also uneven, but had cache spread over
more nodes (8.23–96.68 GiB) and required much less reclaim during preparation.

**Inference:** NUMA placement/reclaim behavior remains the strongest explanation
for unstable residency. These snapshots do not prove the exact kernel allocation
or reclaim decision, nor demonstrate that the reset caused the failure. The
baseline was not run without reset in job 41494246. Node-wide `MemAvailable`
remained about 931 GiB, so neither that number nor low `MemFree` alone predicts
whether the working set will stay resident.

Linux documents task policy for shared file-cache allocation and allocation-order
interleaving; specifying interleave is not evidence that the desired file set
ended up evenly placed or stayed resident. See the
[Linux NUMA memory-policy documentation](https://www.kernel.org/doc/html/latest/admin-guide/mm/numa_memory_policy.html).
Per-file `mincore` checks remain the acceptance evidence.

## Controlled follow-up

Cache-only diagnostic **41596298**, `sdfampere036`, established that the existing
reset-plus-interleave method can still succeed: all five files were **100%**
resident immediately and after ten seconds idle, with zero repairs. Preparation
including its checks took **226.15 s**. Job memory reached **526.71 GiB** under
the 700 GiB cap; reclaim did not increase during this rewarm. Thus this node's
baseline does not reproduce the failed screen's cache churn.

The alternative did not run to completion: its CPU-locality precondition failed.
The allocation permits memory nodes 0–7 but has no CPU on node 2. The probe
terminated its launched workers and removed its private stage; it provides no
valid result for NUMA-bound warming. This is a diagnostic portability defect,
not a cache-gate failure or GPU-runtime failure.

Corrected diagnostic **41596700** is running on `sdfampere042`. It preflights all
worker assignments before staging. Each worker keeps its target memory binding;
when that node has no allocated CPU, it uses the nearest unused allowed CPU and
records the remote assignment. CPU resources, memory cap and acceptance gates
are unchanged. Artifact root:
`/sdf/scratch/users/m/monarin/gpu-validation/issue159/cache-fallout-20261022-r3/`.
The original node-specific submission 41596096 was cancelled while pending:
Slurm estimated October 25 availability for `sdfampere004`. Both methods run
within one allocation on the same private files; this does not reproduce the
exact original node conditions. No GPU timing or runtime edits are part of the
probe.

1. Stage and hash the same five fresh private copies. Reset them with the existing
   guarded helper; verify cold residency.
2. Run the existing interleaved warmer, including its three bounded repairs.
   Check residency immediately and after a ten-second idle interval.
3. Reset only those same private files again. Assign disjoint **64 MiB file
   ranges round-robin across eight workers**, each bound to an allowed CPU and
   its target NUMA memory node (using a nearby CPU when necessary). Disable readahead on each worker's file descriptors to
   keep allocations within the assigned ranges.
4. Verify complete byte coverage, every file's residency, and memory/reclaim/NUMA
   counters. Recheck after idle, after a full pass of the ordinary warmer, and
   after another idle interval. Run this alternative even if the original
   warmer happens to succeed in the new allocation.
5. Preserve results and remove only the probe's private stage. Frozen Stage 4
   A/B builds and all previous evidence remain unchanged.

Syntax checks and the range-partition check passed: every one of the
**335,572,601,564 measured prefix bytes** is assigned exactly once, without gaps
or overlaps. Worker policy and actual read-byte totals will be recorded.

This tests the alternative already outlined in the October 20 diagnosis, which
was not exercised because the simpler reset passed then. If it establishes
stable residency, apply it identically to both screen variants in untimed setup
and verify the real eight-sample screen. It is not yet a validated remedy. A
fresh allocation may not reproduce the original node state, so baseline success
alone would not establish causality.

No threshold relaxation, global cache drop, eviction of shared source files,
mlock, reduced event count or repair after timing is proposed.

[Compact comparison evidence](evidence/issue159_cache_fallout_20261022.json)
retains memory/residency history, helper and input hashes, and probe provenance.
