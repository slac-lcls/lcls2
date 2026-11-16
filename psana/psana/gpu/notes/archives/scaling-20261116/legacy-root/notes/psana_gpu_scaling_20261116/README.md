# One-node GPU scaling plan

Prepared for Mona's request to scale the best one-GPU staging configuration to
2, 3, and 4 GPUs. Preparation does not submit a job. This measures psana staging
throughput, including parsing and synchronization; no calibration, GDS, or cube
processing is enabled.

The accepted stripe screen (job 61372107) measured stripe count 1 at a median
5.4428 GiB/s, range 5.4325–5.4443, across three repetitions. Source evidence:
`../psana_striping_pr298_20261111/screen-20261115/RESULTS.md` and its
`evidence/screen01/audit.json`. Five physical files each have one 1-MiB stripe on
a different OST (five OSTs total). This is not a one-OST ceiling.

## Fixed configuration and CPU adjustment

Retain four BDs per GPU, four KvikIO workers per BD, batch 200, one CUDA stream
per BD, 512-KiB bulk target, 1-MiB KvikIO task, 16-MiB bounce buffer,
compatibility ON and auto-direct-I/O OFF. The budget remains 6 GiB per BD,
24 GiB per active GPU. One SMD0 and one EB feed the entire node; SMD batches
remain 1,000 events. Use the same PR298 build, input manifests, and count-1 view.

Perlmutter has 64 physical CPU cores, each with two SMT threads. Replicating the
old seven physical cores per BD would need 118 physical cores at four GPUs.
The controlled scaling series therefore binds each BD to three exclusive
physical cores with both SMT siblings (six logical CPUs); the two helpers each
retain three physical cores with one thread per core. No physical core is shared
between ranks. All cases use interleaved NUMA allocation, as before. This is a
repeatable capacity policy, not a claim of optimal GPU-local CPU placement.

| Active GPUs | BDs | MPI ranks | KvikIO workers | Bound physical cores | Bound logical CPUs | Reserved GPU budget |
|---:|---:|---:|---:|---:|---:|---:|
| **1, new baseline** | **4** | **6** | **16** | **18** | **30** | **24 GiB** |
| 2 | 8 | 10 | 32 | 30 | 54 | 48 GiB |
| 3 | 12 | 14 | 48 | 42 | 78 | 72 GiB |
| 4 | 16 | 18 | 64 | 54 | 102 | 96 GiB |

The baseline is remeasured under the same CPU policy. A separate one-GPU pilot
control uses the old 34-physical-core policy; that single control checks for a
large CPU-policy regression but is not a statistically repeated comparison.
The single SMD0 and single EB remain fixed, each on three cores, and may become
serial bottlenecks. Sub-linear scaling alone cannot distinguish helper CPU,
storage, parser, or GPU limits; per-rank CPU utilization is not instrumented here.
Scaling increases GPU, BD, CPU, and reader concurrency together. It is node
throughput scaling, not an isolated GPU-only experiment. Four GPUs are reserved
exclusively for every case, including the one-GPU baseline; active GPUs are
nested subsets in PCI order. Unused GPUs do not run BDs.

Slurm uses CR_CORE_MEMORY. MPI steps request even logical CPU counts per rank
(20/12/8/6 for 6/10/14/18 ranks), so whole-core rounding never requests more
than 64 physical cores. Because step masks may differ, all masks are probed first;
their intersection must fit the 42 physical cores used by all shared ranks
(up to three GPUs). These shared ranks keep identical CPU assignments across
all GPU counts. The four-GPU step must itself provide all 54 needed cores;
its four additional BDs can use cores absent from smaller steps.
The campaign fails before GPU tests if that placement is unavailable.

## Execution sequence

1. Prepare a fresh frozen bundle from the reviewed full commit SHA. Check Iris
   personal/project `lcls_g` balances before any later requested submission.
2. Pilot (45-minute allocation limit): require readable Cassini hardware NIC
   counters, probe actual CPU masks for all four rank counts, run the 31 pinned
   native GPU tests separately on each of four GPUs, validate all 30 input file
   layouts/identities and all event headers/ranges, and build count-1 CPU pixel
   witnesses. Run 6,403-event pixel-exact tail cases at each GPU count, requiring
   every BD to process events, then two 100,000-event one-GPU cases: legacy CPU
   policy and compact CPU policy. No automatic promotion.
3. Collect terminal Slurm accounting and run the raw-evidence audit. A screen
   cannot be prepared or started without a passing same-code pilot audit.
4. Separately requested screen (2-hour allocation limit): repeat preflight and
   run three randomized complete blocks of 1/2/3/4 GPUs, seed 20261116. Exactly
   100,000 events (335,820,000,000 requested bytes, about 312.76 GiB) per case
   across the whole node, never per GPU. Twelve full cases total.
5. Audit all raw receipts and terminal accounting, then use `report.py` to give
   node GiB/s median/range, paired speedup against the same block's one-GPU run,
   scaling efficiency, and per-GPU throughput. Inspect per-GPU memory, event
   balance, whole-case wall time, and NIC deltas alongside throughput. With only
   three repeats, report spread rather than an unjustified significance claim.

Each case is a fresh MPI process. Parser kernels compile before the timing
window. Before every case, evict all 30 physical files using DONTNEED and require
two zero-residency mincore scans plus a selected-five-file check; fail after three
attempts. This proves local page-cache coldness, not remote/OST cache coldness.
The event loop is barrier-bracketed and includes read/parse/drain synchronization.
Actual file descriptors, exact requested/completed bytes and read ranges, global
event uniqueness, tail sizes, physical GPU PCI identity, four peers per GPU,
closed resources, budget peaks and sampled whole-device memory are checked.

NIC monitoring reads each physical Cassini device's `hni_sts_rx_ok_octets` and
`hni_sts_tx_ok_octets` from `/sys/class/cxi/cxi*/device/telemetry`, immediately
around the timed window. It saves raw values, kernel timestamps, paths, device
identities and byte deltas. These readable counters were verified on login34;
compute-node access is a mandatory pilot check. Missing counters, resets/wraps,
identity changes or invalid timing stop the campaign. The counters cover all
traffic and protocol overhead; they do not measure Lustre-only bytes or backend
disk reads. No IP-only fallback, counter reset, or system-wide cache purge.

Proposed allocation limits are 45 + 120 = 165 node-minutes; these are limits,
not a runtime forecast or submission authorization. Each case has a 600-second
bound, and a complete four-case block needs that much remaining budget per case
before it starts. Failed attempts remain failed; no retry/resubmission loop.

## Scripts and commands

Run with the pinned Python 3.11 environment, not system Python 3.6:

```bash
PY=/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python
S=/global/u2/m/monarin/lcls2_worktree/benchmarks/sim10-stream-length-20261109/notes/psana_gpu_scaling_20261116
ROOT=/pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111
# Set COMMIT to the full passing reviewed code SHA recorded in status.json.
"$PY" "$S/prepare.py" "$COMMIT" scale-pilot01 pilot
```

`prepare.py` creates a new attempt, read-only bundle, `plan.json`, and
`submit-command.json`, and prints the exact sbatch command. It never submits.
`run.sbatch` reserves one exclusive four-GPU node with lcls_g/premium. After a
later authorized run completes, save accounting (UTC, one row per job/step):

```bash
TZ=UTC sacct -j "$JOB" --parsable2 --format=JobID,State,ExitCode,Account,QOS,NNodes,ElapsedRaw > "$ATTEMPT/accounting-utc.txt"
"$PY" "$ATTEMPT/bundle/audit.py" "$ATTEMPT" "$COMMIT"
# Only after the pilot audit passes:
"$PY" "$S/prepare.py" "$COMMIT" scale-screen01 screen --pilot "$ROOT/scale-pilot01"
# After the screen completes and its accounting is saved:
"$PY" "$ROOT/scale-screen01/bundle/report.py" "$ROOT/scale-screen01" "$COMMIT"
```

Offline verification: `test_scaling.py` checks synthetic 1–4 GPU receipts and
rejects incorrect placement, duplicated/missing events, bad read accounting,
wrong peer counts, warm cache, missing/reset NIC counters, memory violations,
overlapping CPUs, insufficient SMT/core capacity and missing pilot acceptance.
GPU execution, native device tests, pixel witnesses and speedups remain pending
until the separately requested compute pilot. No performance result is implied
by an offline test or review pass.

## Accepted short-pilot coverage risk

The 6,403-event tail cases contain 33 callbacks. At 16 BDs this is about 2.06
callbacks per BD on average, not a guaranteed minimum: dynamic EB scheduling
can leave a BD idle. An idle BD fails qualification safely; the run does not
retry or relax the coverage requirement. This is an accepted bounded-pilot
abort risk. A failure of this kind calls for a separately reviewed larger
witness case, not a claim that the GPUs or storage failed.

## First compute preflight and correction

Job 61384230 used 15 node-seconds and stopped before native GPU tests or timed
reads. Its step masks contained 60/60/56/54 physical cores, but only 53 cores
were common to all shapes. The original requirement for 54 common cores was
too strict. The corrected mapping reserves a common 42-core prefix for all
ranks shared by 1/2/3/4-GPU cases, then fills only the additional four-GPU BD
ranks from that step's remaining cores. All masks and topology are still
checked; neither per-BD CPU resources nor coverage gates are reduced.
The legacy one-GPU control keeps its original per-step selection policy.
