# A/B/B+ in one allocation

Rerun requested after observing different B baselines on sdfampere003 and
sdfampere004. A and B use their original frozen installations; O denotes the
validated batched-locator implementation in this worktree. No production code
or installation changes are part of this rerun.

```bash
sbatch validation/batched-locators-abo-20260917/run.sbatch
```

One allocation, one A100, 48 requested CPUs, 450 GiB host memory; identical
10,000-event stage, calibration snapshot, batch 20, depth 1, 8 GiB device budget,
eight KvikIO threads, 1 MiB tasks, CPU-fallback reads, no user D2H. Each timed
sample follows a 100-event warmup and requires at least 99% file-cache residency
before and after. Lazy detector setup and final GPU synchronization are inside
the unchanged event-loop timer.

Six clean repetitions use all permutations: ABO, BOA, OAB, OBA, BAO, AOB.
Each variant occupies each position twice. The first three repetitions also
include separate CPU/NVTX measurements, alternating the mode order. All six
pixel-reference preflights must pass first. This job skips new Nsight captures;
the earlier traces remain independent evidence of kernel-count changes.

The controller records its allowed CPU mask and explicitly applies the same
mask to every MPI rank. This is a shared allocation mask, not per-core or
GPU-local NUMA pinning. Per-rank affinity, allowed memory nodes, and aggregate
NUMA page counts are logged outside timing, before and after every loop. Node
CPU/NUMA topology, GPU identity/topology, runtime versions, installed source
hashes, GPU clocks/power, host CPU/load/memory, and network counters are retained.
The scripts do not change clocks or host memory policy. Process NUMA page
counts do not establish locality of the entire staged file cache.

The old worker is retained with only A label support and placement logging
outside timing (including a barrier before the timer). Timing hooks support all
three implementations; AST-preservation checks cover A/B/B+.

```bash
python validation/batched-locators-abo-20260917/summarize.py \
  validation/batched-locators-abo-20260917/job-JOBID-warm-ab/results.json \
  --output validation/batched-locators-abo-20260917/job-JOBID-warm-ab/summary.json
python validation/batched-locators-abo-20260917/audit.py \
  validation/batched-locators-abo-20260917/job-JOBID-warm-ab
```

The audit verifies the 27 regular samples, six correctness preflights, order,
cache guards, dataset identity, uniform CPU affinity, runtime/import paths,
and unchanged installed hashes. Throughput is measured, never asserted.

Completed job: **38513845**, sdfampere033, exit 0:0, elapsed 1h33m47s.
All 27 measurements and six preflights passed the final audit.
Report: `psana/psana/gpu/docs/performance/batched_locators_abo_sdf.md`.
