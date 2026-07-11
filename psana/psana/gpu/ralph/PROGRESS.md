# Ralph Loop Journal — psana2 GPU MVP

Append-only. Newest entry LAST. Each iteration appends: date, what was done,
numbers with log paths, keep/revert decision, recommended next step.

---

## 2026-07-10 — Iteration 0 (seed, written by the setup session, not the loop)

**State at loop start:**

- Branch `features/psana2-gpu-mvp` at the end of the 2026-07-07/08 measurement
  campaign. All campaign numbers and verdicts are in `TASK.md`; sweep logs in
  `bench_mpi_sweep/` at repo root. Correctness is bit-exact on two datasets.
- `notes/cpu_push_prototype.md` restored from git history (it was removed in
  the slim commit 958878d6e) so the loop can read the original design.
- Baseline status per PROMPT.md §1:
  - B-CPU (CPU-only psana at scale on FFB): **not measured** — biggest gap.
    `bench_calib.py --compare-cpu` is serial-only; it needs an MPI-safe CPU
    path (same collective-DataSource pattern the GPU path uses) so CPU and GPU
    are measured at identical rank layouts.
  - B-MVP: measured. 210 Hz (1 node / 32 BD), ~295 Hz (2 nodes / 64 BD),
    ~306 Hz (4 nodes / 64 BD). Logs: bench_mpi_sweep/sweep_*.log,
    slurm-31047910 / 31062160 / 31048352.
  - B-FULL (original features/psana2-gpu branch on FFB/r387): **not measured**
    on comparable footing. Old numbers (100 Hz no-D2H, 60 Hz EventPool) are
    from a different dataset/filesystem — context only.
- Open question left by the campaign: the multi-node ~300 Hz plateau is real
  and un-attributed (smd0, EB count, batch size all ruled out). No BD-rank
  profile (read vs H2D vs kernel vs MPI-wait split) exists yet.
- Tooling probed on sdfampere029 (job 31244832,
  `bench_mpi_sweep/tooling_probe.log`): nvidia-smi dmon, CuPy CUDA events,
  pinned alloc, IB counters all available; nsys/ncu NOT installed; DCGM dead.
  See PROMPT.md §8 for how to use them and §6 for the sync-timing trap.

**Recommended next step (iteration 1):** establish B-CPU — add an MPI-capable
CPU benchmark mode to `bench_calib.py` and run it at 1 and 32 BD ranks on FFB,
same layout as the GPU rows. This anchors every future speedup claim.

**Standing user instructions:**
- Commit each iteration; push only when the human asks.
- Never cancel Slurm jobs the loop did not submit.

---

## 2026-07-10 — Iteration 1 (B-CPU tool built; FFB measurement blocked by data purge)

**Task:** establish B-CPU — the CPU-only end-to-end rate at scale on FFB, to
anchor every GPU speedup claim (the seed's recommended iter-1 task).

**What I did:**
- Added `--cpu` mode to `bench_calib.py`: BD ranks run `det.raw.calib()` on the
  SHARED collective DataSource (new `run_cpu_bench_mpi()`), so CPU and GPU are
  measured at IDENTICAL rank layouts. The pre-existing `run_cpu_bench()` was
  serial-only (built its own second DataSource — unsafe with server ranks).
  Skips `init_gpu_rank()`/`prep_calib_constants()` in CPU mode; `--cpu` is one
  new flag, GPU path untouched. `_report_aggregate` made robust to CPU results.
- Verified the code works at 1 and 32 BD ranks (mpirun --bind-to none on the
  ralph-gpu node, job 31260656 / sdfampere042).

**The blocker (why the headline number is missing):**
- r387 — the reference run behind EVERY B-MVP FFB number — has been PURGED from
  the FFB rolling buffer. The only FFB-resident run for mfx101210926 is r0215,
  which is INCOMPLETE (in-progress stream s003, no fetchable Configure/BeginRun
  dgrams → DataSource fails). r387 exists only on /sdf/data Lustre now.
- So B-CPU-on-FFB (the apples-to-apples comparison to B-MVP's 210 Hz / 32 BD)
  could not be measured. This needs r387 (or another complete Jungfrau run)
  restaged to FFB — a facility action on a live production buffer, not a psana
  code lever, so I did NOT do it autonomously.

**What I did measure (Lustre r387, verification only — logs cpu_lustre_bd*.log):**
- 1 BD rank: 10.1 Hz; 32 BD ranks: 106.5 Hz aggregate (3.33 Hz/rank).
- CPU calib inflates 29.7 → 54.5 ms/event from 1 → 32 ranks (core/mem-bandwidth
  contention). CPU compute ceiling at 32 ranks ≈ 587 Hz vs GPU kernel ~3100 Hz.
- These are Lustre + cache-warm, NOT comparable to the FFB B-MVP table nor to
  the old cold-Lustre GPU column. Recorded only to prove the `--cpu` path works.

**Keep/revert:** KEEP the `--cpu` code — it's the tool the loop needs; it will
produce the real baseline the moment a complete run is on FFB. Correctness gate
not triggered (CPU mode calls `det.raw.calib()`, the reference itself; GPU
numeric path unchanged).

**Recommended next step:** (a) get a complete run onto FFB — either ask the
human to restage r387, or check whether a currently-active FFB experiment has a
complete 32-seg Jungfrau run usable as the new reference; then rerun
`--cpu` and the GPU path at 1/32 BD to fill B-CPU-on-FFB. (b) Failing that, the
next unblocked code experiment is the BD-rank profile (read vs H2D vs kernel vs
MPI-wait) that the multi-node plateau investigation still lacks — but that also
needs FFB data to be meaningful.

BLOCKED: reference run r387 purged from FFB (only incomplete r0215 remains); B-CPU-on-FFB needs a complete Jungfrau run restaged to FFB before it can be measured.

---

## 2026-07-10 — Operator note (human-directed, between iterations)

Reference dataset set to **mfx101572426 r47** on FFB
(`/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc`). Verified 2026-07-10: det
`jungfrau`, shape (32, 512, 1024) = 16.78M px, `det.raw.calib()` returns float32
with constants deployed; ~37k events → full headroom for `-n 500` at 32 BD ranks.
PROMPT.md, TASK.md, and the code docstrings now point here.

The iteration-1 blocker is **resolved** — do not re-flag it. Next task:
re-measure **B-MVP (GPU)** and measure **B-CPU** on r47 at 1 and 32 BD ranks on
FFB. These become the anchor pair (identical run, identical rank layout). The old
r387 FFB numbers (B-MVP 210 Hz, etc.) are historical — same detector and FFB, so
expect them to reproduce, but treat as unverified on r47 until measured.

`MAX_ITERS` raised to 25.

---

## 2026-07-10 — Iteration 1 (B-MVP established on r47)

**Task:** re-establish B-MVP — the GPU two-function calib rate, end-to-end
through the psana event loop on r47/FFB, at 1 and 32 BD ranks.

**What I did:**
- Ran `bench_calib.py -e mfx101572426 -r 47 -n 500 --warmup 10 --dir $FFB` at
  1 BD rank and at 32 BD ranks (34 procs: smd0 + EB + 32 BD), GPU path.
- Confirmed geometry: 32-seg Jungfrau, (32, 512, 1024) = 16.78M px, single-event
  occupancy 7585% (one event saturates the A100).

**Numbers (r47, FFB, A100, `mpirun --bind-to none`):**
- 1 BD rank: 36.8 Hz (550 events); H->D 4.12 ms, kernel 0.352 ms/event.
- 32 BD ranks: **175.3 Hz aggregate** (17,600 events); per-rank 5.48 Hz;
  H->D 11.85 ms, kernel 0.614 ms/event.
- Kernel time consistent with history (~0.32–0.6 ms/event). Historical r387 was
  210 Hz @ 32BD; r47 lands in the same ballpark, modestly lower.

**Keep/revert:** KEEP — this is the r47 B-MVP anchor.

**Recommended next step:** B-CPU on r47. The `--cpu` mode already exists in
`bench_calib.py` (commit edef959b3). Run it at 1 and 32 BD ranks on FFB (same
command with `--cpu` added, `-n 200` is plenty), journal the aggregate Hz + CPU
calib ms/event, and commit. That fills the last missing baseline so the GPU
speedup on r47 can finally be stated.

---

## 2026-07-10 — Iteration 2 (B-CPU established on r47 — anchor pair complete)

**Task:** the last missing baseline — B-CPU (CPU-only `det.raw.calib()` end-to-end
through the psana loop) on r47/FFB at 1 and 32 BD ranks, same layout as B-MVP.

**What I did:**
- Ran `bench_calib.py --cpu -e mfx101572426 -r 47 -n 200 --warmup 10 --dir $FFB`
  on the `ralph-gpu` node (job 31267701, sdfampere029) at 1 BD rank (3 procs) and
  32 BD ranks (34 procs), `mpirun --bind-to none --oversubscribe`.
- Note: `bench_mpi_sweep/ralph_tmp/` had been wiped at iteration start; recreated
  it + `bench.sh` (harness only, not tracked). 34-proc MPI startup takes ~2 min,
  so the 32-BD run must be launched in the background, not a foreground 2-min tool
  call.

**Numbers (r47, FFB, A100 node, `mpirun --bind-to none`):**

| BD ranks | CPU aggregate Hz | per-rank Hz | CPU calib ms/event | log |
|---:|---:|---:|---:|---|
| 1  | 22.5 | 22.53 | 31.6  | `bench_mpi_sweep/ralph_tmp/cpu_1bd_r47_164605.log` |
| 32 | 44.9 | 1.40  | 433.7 | `bench_mpi_sweep/ralph_tmp/cpu_32bd_r47_164919.log` |

**The finding:** CPU calib on the 16.78M-px (32-seg) detector inflates
**31.6 → 433.7 ms/event (13.7×)** from 1 → 32 ranks — far worse than the Lustre
r387 verification run (29.7 → 54.5 ms, smaller/warm). 32 cores buy only ~2× the
aggregate of one rank: `det.raw.calib()` on a 33.5 MB array is memory-bandwidth
bound and collapses under 32-way contention. CPU compute ceiling at 32 ranks ≈
32/0.4337 = **~74 Hz**, i.e. the CPU path is compute-bound, not I/O-bound, at
scale here.

**The anchor pair on r47/FFB (identical run, identical rank layout):**

| | 1 BD | 32 BD |
|---|---:|---:|
| B-CPU (aggregate Hz) | 22.5 | 44.9 |
| B-MVP / GPU (aggregate Hz) | 36.8 | 175.3 |
| **GPU speedup** | **1.6×** | **3.9×** |

The GPU advantage *widens* with scale (1.6× → 3.9×) precisely because the CPU
path stops scaling while the GPU kernel (0.32–0.6 ms) has 76× headroom. The GPU
path's own 32-BD bottleneck is H2D (11.85 ms/event under 32-rank PCIe
contention), not compute — the standing open question.

**Keep/revert:** KEEP — records the last baseline. Correctness gate not triggered
(CPU mode *is* the `det.raw.calib()` reference; GPU numeric path untouched).

**Recommended next step:** all three §1 baselines that can be measured on r47 are
now established (B-CPU, B-MVP; B-FULL still needs a worktree rerun). The next
unblocked experiment is the one the multi-node plateau investigation still lacks:
a **BD-rank profile of the GPU path** (read vs H2D vs kernel vs MPI-wait split)
at 32 BD, using CUDA events + `nvidia-smi dmon` PCIe counters (§6/§8), to confirm
whether the 11.85 ms H2D-under-contention is the true 32-rank ceiling. If it is,
the first lever to try is pinned-memory + async H2D on the BD rank (overlap-safe
timing per §6).

---

## 2026-07-10 — Iteration 3 (BD-rank profile — H2D is NOT the 32-BD ceiling; psana data delivery is)

**Task:** the recommended experiment — attribute the 32-BD per-event wall time
(read vs H2D vs kernel vs generator/MPI wait), to test the standing hypothesis
that per-rank H2D-under-contention is the multi-node ceiling and that pinned+async
H2D is the unlock.

**What I did:**
- Added a `--profile` mode to `bench_calib.py` (`run_gpu_bench_profile`) that
  buckets each event's WALL time into **wait** (time to advance `run.events()` =
  bigdata read + EB/MPI serving), **read** (`det.raw.raw(evt)`), **H2D**, and
  **kernel**. Kept separate from the established `run_gpu_bench` loop so the
  B-MVP timing path stays byte-identical; the buckets sum to wall (the sanity
  check). GPU numeric path untouched → correctness gate not triggered.
- Ran at 1 BD (`-n 500`) and 32 BD (`-n 200`) on r47/FFB, `ralph-gpu` node
  (job 31267701, sdfampere029), `mpirun --bind-to none`. Ran `nvidia-smi dmon
  -s put` alongside the 32-BD run to observe PCIe rx (§8).

**Numbers (r47, FFB, A100, ms/event):**

| bucket | 1 BD | 32 BD | share @32BD |
|---|---:|---:|---:|
| wait (gen advance = read+EB/MPI) | 12.0 | 231.8 | 60% |
| read (`det.raw.raw`) | 7.9 | 119.3 | 31% |
| H->D | 3.9 | 43.6 | 11% |
| kernel | 0.32 | 2.3 | 0.6% |
| **wall/event** | 24.2 | 388.7 | |
| aggregate Hz | 41.4 | 82.3 | |

Logs: `bench_mpi_sweep/prof_ffb_r47_bd1.log`,
`bench_mpi_sweep/prof_ffb_r47_bd32.log`,
`bench_mpi_sweep/prof_ffb_r47_bd32_dmon.log` (sum ≈ wall in every rank block →
attribution trustworthy). dmon active window: **rxpci sustained ~3.7 GB/s (~15%
of the 25 GB/s gen4 wire)**, SM ~16% — the GPU and its PCIe link are both largely
idle.

**The finding (overturns the standing hypothesis):**
- At 32 BD, GPU-side work (H2D + kernel) is only **~12%** of per-event wall.
  psana bigdata delivery (wait + read) is **~91%**. So the per-rank H2D — long
  named in TASK.md as "the leading suspect" and the reason pinned+async H2D was
  "promoted to the one multi-node unlock" — is NOT the ceiling. Even driving H2D
  to zero would cut per-event wall 389 → ~345 ms = **at most +13%**. dmon
  independently confirms it: PCIe rx sits at ~15% of the wire, not saturated.
- The real ceiling is the **psana data-delivery path**: `det.raw.raw(evt)`
  itself costs 119 ms at 32 ranks (15× its 1-BD cost), and the generator advance
  (EB batch serving + bigdata read + MPI) costs 232 ms. Both scale far worse
  with rank count than H2D does.
- Caveat on the absolute rate: this run measured 82.3 Hz @ 32 BD vs the anchor's
  175.3 Hz — the FFB "minute-to-minute variance on a live production FS" already
  documented in TASK.md, and/or the shorter `-n 200`. The **relative breakdown**
  is the robust result and is speed-independent: at the faster 175 Hz moment H2D
  was 11.85 ms of 182 ms/event = 6.5% (an even smaller share), so H2D is a minor
  contributor at both speeds. The proportions, not the 82 Hz, are the finding.

**Keep/revert:** KEEP the `--profile` mode — it's the attribution tool the plateau
investigation lacked, and it settled the question. Correctness gate not triggered
(instrumentation only; numeric path byte-identical).

**Recommended next step:** the lever priority is now inverted — **de-prioritize
pinned+async H2D** (≤13% ceiling, proven). The next experiment must split the
`wait` bucket into **storage-read vs EB/MPI-wait** on a BD rank (timestamped
instrumentation or py-spy per §8), because that decides which psana lever moves
the 351 ms/event delivery cost: if it's storage-read, the levers are more BD
ranks/node + write-time compression (§ TASK.md levers a–f); if it's EB/MPI-wait,
it points back into the EB→BD serving chain (a psexp change, in scope per §3/§7).
Also worth one cheap check: whether `det.raw.raw(evt)`'s 119 ms is lazy bigdata
I/O (storage) or in-process deserialization/reshape (CPU) — that alone reclassifies
31% of the wall.

---

## 2026-07-10 — Iteration 4 (storage-vs-serving fork: the IB-counter probe is invalid here — FFB is WekaFS; bandwidth is NOT the ceiling)

**Task:** the recommended experiment — decide whether iter 3's 91%-of-wall
delivery cost (wait + read) is **storage-bound** or **psana-serving-bound**. The
plan was §8's endorsed direct probe: measure FFB bytes delivered to the node via
IB `port_rcv_data` counters during a 32-BD run and compare to the 7.9 GB/s
per-node storage ceiling.

**What I did:**
- Added `bench_mpi_sweep/ralph_tmp/ib_sampler.sh` (1 Hz sampler of every
  `/sys/class/infiniband/*/ports/*/counters/port_rcv_data`, words×4 = bytes).
- **Validated the probe first** (the step that saved the iteration): a controlled
  2.1 GB direct read of an FFB xtc2 file (`dd iflag=direct`) → IB counter delta =
  **0 bytes**. Then confirmed the mount type: FFB is **WekaFS**
  (`type wekafs`, `_netdev`, with `/opt/weka/.../huge` hugepage + DPDK userspace
  agent containers), NOT Lustre.
- Ran the 32-BD `--profile` benchmark on r47/FFB (`ralph-gpu` node, job 31267701,
  sdfampere029, `mpirun --bind-to none`, `-n 200`) with the IB sampler alongside
  (131 samples). Also probed `weka stats realtime` as the correct replacement.

**Numbers (r47, FFB, A100):**
- IB `port_rcv_data` over the **entire** 32-BD run (~236 GB read from FFB across
  131 samples): **delta = 0, max 0.000 GB/s**. The kernel IB counters are blind
  to WekaFS traffic — Weka's DPDK userspace client bypasses kernel IB verbs.
- Benchmark reproduced iter 3's breakdown almost exactly (robustness check):
  **aggregate 83.1 Hz @ 32 BD** (iter 3: 82.3), wait 235.1 / read 117.3 /
  H2D 43.3 / kernel 2.3 ms; sum ≈ wall. Log:
  `bench_mpi_sweep/ralph_tmp/prof_ib_32bd_r47_*.log`,
  `bench_mpi_sweep/ralph_tmp/ib_32bd_r47_*.log`.
- Single-stream cold direct read from FFB = **490 MB/s** (one data point).
- `weka stats realtime` requires cluster auth (`weka user login`) — the correct
  userspace probe is **blocked on credentials** (a facility/human decision).

**The finding (two results):**
1. **The IB-counter storage probe is dead on this facility.** §8 and TASK.md say
   IB counters are "the ONLY way to observe FFB/Lustre traffic." That was premised
   on Lustre/RDMA-verbs. FFB here is WekaFS/DPDK-userspace and moves every byte
   outside the kernel verbs stack → `port_rcv_data` never moves (validated: 2.1 GB
   read → 0, and a full 236 GB run → 0). Corrected in TASK.md. The only direct
   Weka throughput probe (`weka stats`) needs cluster login we don't have.
2. **Storage BANDWIDTH is not the binding constraint at 32 BD.** Achieved psana
   read bandwidth = 83.1 Hz × 33.5 MB = **2.78 GB/s = 35% of the 7.9 GB/s per-node
   ceiling**. The link sits ~65% idle *while the pipeline is slow*. So iter 3's
   "delivery = 91% of wall" is a **latency/serialization** limit in psana's
   serving chain, not a facility storage-bandwidth wall. (Cross-check: at the fast
   175 Hz anchor it would be 5.9 GB/s = 75% of ceiling — still not saturated.)
   The lever is therefore **concurrency / latency-hiding** — more BD ranks per
   node, async/prefetch bigdata read, or read batching — NOT more storage
   bandwidth.

**Keep/revert:** KEEP `ib_sampler.sh` in ralph_tmp but it is now **documented dead
for this FS** — do not use IB counters for FFB traffic here. Correctness gate not
triggered (measurement/instrumentation only; GPU numeric path byte-identical).

**Recommended next step:** with bandwidth ruled out and both direct storage probes
(IB dead, Weka-stats blocked), the clean single-variable way to separate
latency-bound from serialization-bound is a **BD-ranks-per-node concurrency sweep**
at fixed 1 node/FFB (e.g. 8/16/32/48/64 BD; needs `--oversubscribe`), reading
aggregate Hz → aggregate GB/s each point. If aggregate read GB/s climbs toward
7.9 as ranks rise, the delivery cost is latency-bound and the unlock is more
concurrency / async prefetch (a psexp or BD-loop change, in scope per §3/§7); if
it plateaus well below 7.9, it is a per-rank CPU/serialization limit in the
serving chain (points at `det.raw.raw` deserialization or the EB→BD path). No new
tooling required — this is the decisive next measurement.

---

## 2026-07-10 — Iteration 5 (BD-per-node concurrency sweep — pipeline plateaus at ~16 BD, well below storage; and a hard A100-memory ceiling at ~48 BD)

**Task:** the recommended decisive measurement — a BD-ranks-per-node concurrency
sweep at fixed 1 node/FFB (GPU path) to separate **latency-bound** (aggregate
read GB/s climbs toward the 7.9 GB/s per-node storage ceiling as ranks rise →
unlock is more concurrency / async prefetch) from a **per-rank serialization
limit** (plateaus well below 7.9 → the serving/deserialize path is the wall).

**What I did:**
- Added `bench_mpi_sweep/ralph_tmp/concsweep.sh` (not tracked): runs the GPU
  `bench_calib.py` at BD = 8/16/32/48/64 on the `ralph-gpu` node (job 31267701,
  sdfampere029), `mpirun --bind-to none --oversubscribe`, `-n 100 --warmup 10`,
  r47/FFB. The node has 112 allocated CPUs, so even 64 BD (66 procs) map to real
  cores — no oversubscription artifacts. Bracketed by re-running 32 BD last to
  gauge FFB minute-to-minute variance. Derived aggregate read BW = agg Hz ×
  33.5 MB/event.
  - (First launch failed instantly — I dropped `--oversubscribe`, so PRRTE saw
    only the 1-slot srun step. Re-added it; the established recipe is
    `--bind-to none --oversubscribe`.)

**Numbers (r47, FFB, A100, `bench_mpi_sweep/ralph_tmp/conc_*_172029_*.log`,
summary `conc_summary_172029.log`):**

| BD ranks | aggregate Hz | aggregate read GB/s | note |
|---:|---:|---:|---|
| 8  | 66.8 | 2.24 | |
| 16 | 83.9 | 2.81 | |
| 32 | 84.9 | 2.84 | |
| 48 | **OOM** | — | A100 40 GB exhausted (`OutOfMemoryError: allocating 201,326,592 bytes`) |
| 64 | **OOM** | — | same |
| 32 (bracket, +13 min) | 39.9 | 1.34 | FFB window degraded ~2× over the sweep |

**Finding 1 — the pipeline saturates at ~16 BD, far below the storage ceiling.**
Aggregate read bandwidth climbs 8→16 BD (+26%, 2.24→2.81 GB/s) but is FLAT
16→32 BD (+1%, 2.81→2.84). It plateaus at **~2.8 GB/s = 35% of the 7.9 GB/s
per-node storage ceiling** — it does NOT climb toward 7.9 as ranks rise. Per the
decision rule this is the **per-rank serialization** verdict, not latency-bound:
adding outstanding reads past ~16/node buys nothing, so the wall is per-rank CPU
work in the serving/deserialize chain (`det.raw.raw` + EB→BD), not too-few
in-flight reads. **The unlock is therefore NOT more BD ranks / more concurrency;
it is making each rank's serving path cheaper.**
- Confound (honest): the end-bracket 32 BD fell to 39.9 Hz (1.34 GB/s), i.e. the
  live FFB window degraded ~2× across the 13-min sweep. So the plateau *level* is
  FS-window-dependent. But 16 BD and 32 BD were measured back-to-back (17:21 /
  17:22, both ~84 Hz) — the rank-count flatness *within a stable window* is the
  robust signal and is what the verdict rests on. Fully separating "per-rank
  serialization" from "shared FS currently throttled to ~2.8 GB/s" would need a
  concurrent raw `os.pread` reader control in the SAME window (the TASK.md raw
  matrix hit 7.9 GB/s at 32 readers, but on a different day/window).

**Finding 2 — NEW hard ceiling: the single-A100 GPU path OOMs at 48 BD ranks.**
Each BD rank is an independent process with its own CUDA context (~0.3–0.6 GB)
AND its own replicated calib constants (`peds_gpu + gmask_gpu ≈ 384 MB` for the
32-seg detector) plus a 201 MB working allocation. At 48 ranks this exhausts the
40 GB A100 (both 48 and 64 aborted at init/early events). So **~32 BD ranks/A100
is the practical MVP concurrency ceiling** on a single GPU — you cannot chase
storage bandwidth by piling on ranks without shrinking per-rank GPU footprint.
This is exactly the trigger the DEFERRED `share_calib_between_gpu_peers` (CUDA-IPC
calib sharing) entry names ("multiple BD ranks on same GPU AND OOM from duplicate
calib buffers") — now MET; I marked it in DEFERRED.md. **But** given Finding 1
(concurrency past ~16 BD does not raise throughput), IPC sharing would buy memory
headroom, NOT throughput — so it is justified for enabling denser packing / more
GPUs-per-node topologies, not as the throughput lever.

**Keep/revert:** KEEP `concsweep.sh` in ralph_tmp (the concurrency-sweep tool).
Correctness gate not triggered (benchmark harness + GPU numeric path byte-identical).

**Recommended next step:** Finding 1 points squarely at **per-rank serialization
in the read/deserialize path**. The decisive next experiment: determine whether
`det.raw.raw(evt)`'s ~119 ms @32BD (iter 3, 31% of wall) is **storage I/O** or
**in-process CPU deserialization/reshape** — instrument inside `det.raw.raw` on a
single BD rank (or compare it against a bare `os.pread` of the same dgram bytes).
If CPU-bound deserialize/reshape, the lever is a lighter raw accessor or moving
the reshape off the hot path (a psana-side change, in scope §3/§7); if it is lazy
storage I/O, it points to per-BD-rank async prefetch. Second, if a clean window
is available, repeat the 8/16/32 sweep *alongside a concurrent raw `os.pread`
reader control* to nail down whether the ~2.8 GB/s plateau is per-rank
serialization or a currently-throttled shared FS.

---

## 2026-07-10 — Iteration 6 (read-bucket decomposition — det.raw.raw is CPU-bound, NOT storage I/O; the redundant final .copy() is its single largest cost)

**Task:** the decisive experiment both iter 3 and iter 5 recommended — settle
whether `det.raw.raw(evt)`'s read bucket (119 ms/event @ 32 BD, iter 3; 31% of
wall, 15x its 1-BD cost) is **lazy bigdata storage I/O** or **in-process CPU
deserialize/reshape**. That one answer reclassifies 31% of the per-event wall
and picks the lever (async prefetch vs a lighter accessor).

**What I did:**
- Added a `--profile-read` mode to `bench_calib.py` (`run_gpu_bench_profile_read`).
  Per event it answers the question two independent ways:
  1. **Internal breakdown** of `det.raw.raw` via the `evt._det_raw_timing` hook
     already present in `AreaDetectorRaw.raw` (commit 12c77b8aa): `seg`
     (`_segments` dict lookup) / `stack` (the per-segment `segs[id].raw`
     deserialize + `np.copyto` memcpy loop) / `copy` (residual = final
     `reshape_to_3d` + `arr.copy()`).
  2. A **second, un-instrumented `det.raw.raw(evt)` on the same event** (`read2`):
     `read2 << read1` ⇒ one-time-per-event cost = lazy storage I/O / first-touch
     page-in (bytes resident 2nd time) ⇒ storage-bound; `read2 ≈ read1` ⇒ cost
     repeats every call = pure CPU memcpy/deserialize ⇒ CPU-bound.
  Kept the established `run_gpu_bench` / `run_gpu_bench_profile` loops
  byte-identical (new function). GPU numeric path untouched → correctness gate
  not triggered.
- Ran 1 BD (`-n 300`) and 32 BD (`-n 150`) on r47/FFB, `ralph-gpu` node
  (job 31267701, sdfampere029), `mpirun --bind-to none --oversubscribe`.

**Numbers (r47, FFB, A100, ms/event; logs `bench_mpi_sweep/ralph_tmp/profread_{1,32}bd_174837.log`, driver `profread_driver_174837.log`):**

| bucket | 1 BD | 32 BD |
|---|---:|---:|
| wait (gen advance = read+EB/MPI) | 13.21 | 264.43 |
| **read1** (det.raw.raw 1st call) | **9.19** | **154.20** |
|   seg (_segments dict lookup) | 0.008 | 0.14 |
|   stack (per-seg .raw deserialize + copyto memcpy) | 4.18 | 64.54 |
|   copy (reshape + final .copy()) | 5.00 | 89.53 |
| **read2** (2nd det.raw.raw, same evt) | **9.01** | **152.67** |
| H->D | 3.90 | 53.79 |
| kernel | 0.32 | 2.49 |
| aggregate Hz | 28.1 | 51.2 |

(The 32-BD run landed in a slow FFB window — 51 Hz agg vs iter 3's 82 Hz. The
absolute rate is FS-window-dependent as documented; the **relative breakdown**
and the **read1≈read2 equality** are speed-independent and are the finding.)

**The finding (settles the open question):**
1. **det.raw.raw is CPU-bound, NOT storage I/O.** `read2 ≈ read1` at BOTH scales
   — 9.01/9.19 = 98% at 1 BD, 152.7/154.2 = 99% at 32 BD. The cost repeats
   identically on a second call to the same event, so the dgram bytes are already
   resident (read during the generator-advance `wait` bucket / page cache); the
   read bucket is pure in-process CPU memcpy/deserialize each call. This
   **reclassifies iter 3's 31%-of-wall read bucket from storage to CPU** and
   means the lever is a lighter accessor, NOT per-BD async prefetch of that
   bucket.
2. **The single largest component of det.raw.raw is the final `.copy()`, and it
   is pure waste for the GPU path.** At 32 BD `copy` = 89.5 ms > `stack` = 64.5 ms
   (`seg` ≈ 0). That `copy` is a full 33.5 MB host→host memcpy that exists only to
   avoid view-aliasing across events — but the GPU path calls `cp.asarray(raw)`
   immediately, copying host→device before the next event, so the extra host copy
   is redundant. `det.raw.raw(evt, copy=False)` (the flag already added in commit
   ecf74b87f) eliminates it. Expected payoff: `copy` is 58% of read at 32 BD
   (54% at 1 BD) and ~14% of per-event wall at both scales → up to ~+14–16%
   throughput on its own, **plus** relief of host DRAM-bandwidth contention that
   `stack` (another 33.5 MB memcpy) and `H->D` (33.5 MB to device) also fight for
   — read1's 16.8x super-linear scaling (9.2→154 ms across 32 ranks) is the
   signature of that shared-DRAM contention, since the work is pure repeatable
   CPU, so cutting one of the two host memcpys should help more than the naive 14%.

**Keep/revert:** KEEP `--profile-read` (`run_gpu_bench_profile_read`) — it's the
tool that settled I/O-vs-CPU and it will measure the copy=False before/after.
Correctness gate not triggered (instrumentation only; GPU numeric path
byte-identical).

**Recommended next step:** implement the clean single-variable change this points
to — pass `copy=False` to `det.raw.raw` in the GPU bench path (it is consumed
immediately by `cp.asarray`, so no view-aliasing hazard). Measure before/after at
1 BD and 32 BD on FFB with `--profile-read` (watch `copy`→~0 and whether `stack`
/`H->D` also drop from reduced DRAM contention), and **run the correctness gate**
(`test_jungfrau_calib.py`, max_diff 0.0) since the numeric input path changes.
Keep only if the aggregate Hz moves; revert + journal if the freed host copy is
masked by the `wait` bucket. If it lands, the same `copy=False` belongs in the
public two-function API's expected usage.

---

## 2026-07-10 — Iteration 7 (land copy=False — the first landed throughput win: +30% @ 1 BD and @ 32 BD, bit-exact)

**Task:** implement the single-variable change iter 6 pointed to — pass
`copy=False` to `det.raw.raw` in the GPU path (it is consumed immediately by
`cp.asarray`, so no view-aliasing hazard) — measure before/after at 1 and 32 BD
on FFB, and run the correctness gate since the numeric input path changes.

**What I did:**
- Wired the toggle into the byte-identical B-MVP loop `run_gpu_bench` (line 112)
  and **promoted `copy=False` to the DEFAULT** of the GPU path;
  `--copy-true` restores the pre-iter-7 baseline for A/B. Added the matching
  `--copy-true` to `test_jungfrau_calib.py` and flipped its default to
  `copy=False` so the standard gate now guards the real GPU path.
- Measured before/after by **interleaving** copy=True/copy=False back-to-back,
  2 brackets each, to control for FFB's documented minute-to-minute variance
  (`bench_mpi_sweep/ralph_tmp/copyfalse_driver.sh`). `ralph-gpu` node
  (job 31267701, sdfampere029), `mpirun --bind-to none --oversubscribe`, r47/FFB.

**Numbers (r47, FFB, A100, aggregate Hz; logs `bench_mpi_sweep/ralph_tmp/cf_*_175942.log`, driver `cf_driver_175942.log`):**

| config | copy=True | copy=False | gain |
|---|---|---|---|
| 1 BD  | 39.8, 42.7 → **41.3** | 53.6, 54.6 → **54.1** | **+31%** |
| 32 BD | 82.6, 86.7 → **84.7** | 107.6, 112.2 → **109.9** | **+30%** |

Both brackets agree and the copy=False runs are uniformly higher than the
copy=True runs measured seconds earlier — the effect is not an FFB-window
artifact. (Absolute rates are in a slow FFB window vs the 175 Hz anchor; the
+30% relative is the robust, speed-independent result.)

**Correctness gate:** `test_jungfrau_calib.py -e mfx101572426 -r 47 -n 20`
(now defaulting to copy=False) → **20/20 OK, max_diff 0.0** (also verified
explicitly with the flag earlier in the iteration). Bit-exact.

**The finding:** the measured **+30% at both scales exceeds iter 6's naive +14%
prediction**, confirming the DRAM-bandwidth-contention hypothesis: the final
`.copy()` is a 33.5 MB host→host memcpy that not only wastes its own time but
also contends for the host DRAM bus with `stack` (another 33.5 MB memcpy) and
`H->D`; removing it speeds those too. This is the **first code change on the
branch to move the throughput number** — every prior iteration measured/attributed.

**Keep/revert:** KEEP — number moved +30%, bit-exact. Promoted to default.

**Recommended next step:** with `copy` eliminated, `stack` (per-seg `.raw`
deserialize + `np.copyto` memcpy) is now the largest read component (64.5 ms
@ 32 BD, iter 6). It is a second 33.5 MB host memcpy into `_raw_buf`. The next
single-variable experiment: can the GPU path skip the host `stack` entirely by
copying each segment's `.raw` **directly host→device** (per-segment
`cp.asarray` into a pre-allocated device buffer), removing the last host-side
33.5 MB memcpy before H2D? Measure with `--profile-read` (watch `stack`→0) at
1/32 BD, gate bit-exact. Also worth doing: propagate the `copy=False` guidance
into the public two-function API's usage docstring (`__init__.py`).

---

## 2026-07-10 — Iteration 8 (per-segment H2D — skip the host `stack` memcpy: +38% @1BD, +32% @32BD, bit-exact)

**Task:** the single-variable change iter 7 pointed to — after `copy=False`
landed, the host `stack` (per-seg `.raw` deserialize + `np.copyto` into the
contiguous `_raw_buf`, 64.5 ms/event @32BD, iter 6) is the largest remaining
read component. Can the GPU path skip it by copying each segment's `.raw`
**directly host→device** into a pre-allocated device buffer, removing the last
host-side 33.5 MB memcpy before H2D?

**What I did:**
- Added `run_gpu_bench_seg_h2d` + `--seg-h2d` to `bench_calib.py`. Instead of
  `raw = det.raw.raw(evt, copy=False); cp.asarray(raw)` (host stack loop → one
  33.5 MB H2D of `_raw_buf`), it does `segs = det.raw._segments(evt); for idx,sid:
  raw_gpu_buf[idx].set(segs[sid].raw)` — 32× per-seg 1 MB H2Ds straight into a
  device buffer, no host stack. `run_gpu_bench` (the B-MVP anchor loop) left
  byte-identical. Synchronous (device sync after the transfer loop) → wall-clock
  rate is a valid headline, no overlap-timing trap (§6).
- Gated with a new `--seg-h2d` mode in `test_jungfrau_calib.py` that builds the
  device buffer the same per-seg way and compares against `det.raw.calib`.
- **One bug found + fixed before any perf number:** each segment's `.raw` is
  shape `(1, 512, 1024)`, so the naive buffer became `(32, 1, 512, 1024)`. The
  kernel ravels so the *data* was correct, but `fused_calib_gpu` reshapes to that
  4-D shape and the correctness comparison broadcast → 20/20 MISMATCH,
  max_diff 40277. Fix: reshape the buffer to `(-1, 512, 1024)` (the `reshape_to_3d`
  equivalent) so it matches `det.raw.raw`'s shape. After the fix, bit-exact.
- Measured A/B by interleaving baseline (default `copy=False`, one big H2D) vs
  `--seg-h2d` back-to-back, 2 brackets each, to control for FFB minute-to-minute
  variance (iter-7 method). `ralph-gpu` node (job 31267701, sdfampere029),
  `mpirun --bind-to none --oversubscribe`, r47/FFB. Driver
  `bench_mpi_sweep/ralph_tmp/segh2d_driver.sh`.

**Correctness gate:** `test_jungfrau_calib.py --seg-h2d -e mfx101572426 -r 47
-n 20` → **20/20 OK, max_diff 0.0**. Bit-exact.

**Numbers (r47, FFB, A100, aggregate Hz; logs `bench_mpi_sweep/ralph_tmp/segh2d_{1bd,32bd}_{base,seg}_{a,b}_183300.log`):**

| config | baseline copy=False (Hz) | --seg-h2d (Hz) | gain |
|---|---|---|---|
| 1 BD  | 52.0, 54.2 → **53.1** | 72.7, 74.0 → **73.4** | **+38%** |
| 32 BD | 115.9, 113.6 → **114.8** | 141.0, 162.8 → **151.9** | **+32%** |

Every `--seg-h2d` bracket is uniformly higher than both baseline brackets
measured seconds earlier — the effect is not an FFB-window artifact. (Absolute
rates are in a faster FFB window than iter 7's; the +32–38% relative is the
robust, speed-independent result.)

**The finding:** in the anchor `run_gpu_bench` loop the host `stack` memcpy is
*untimed* (it happens inside `det.raw.raw`, before the H→D timer), so it inflated
wall silently. seg-h2d folds the whole raw ingestion into the H→D bucket, and the
combined cost is *lower* than the baseline's stack + separate H2D: at 1 BD the
seg-h2d H→D bucket is **3.50 ms** vs the baseline's H2D-only 4.11 ms **plus** its
untimed ~5 ms stack; at 32 BD seg-h2d H→D **~46 ms** replaces the baseline's
stack (64.5 ms, iter 6) + H2D (~44 ms) ≈ 108 ms. Eliminating the second 33.5 MB
host memcpy (after iter 7 killed the first, the `.copy()`) is the win — same
DRAM-bandwidth-contention mechanism, now the last host memcpy on the path is gone.
This is the **second landed throughput win** on the branch.

**Keep/revert:** KEEP — number moved +32–38% at both scales, bit-exact. Kept as
the opt-in `--seg-h2d` variant (a structurally different ingestion loop, unlike
iter 7's one-line flag on the same loop), not promoted into the byte-identical
B-MVP anchor.

**Recommended next step:** (a) **promote seg-h2d to the default GPU ingestion
pattern** — make it the bench default (retiring the `det.raw.raw`+`cp.asarray`
route to an opt-in baseline) and document per-segment H2D as the recommended way
to feed `fused_calib_gpu` in the public two-function API usage (`__init__.py`),
since both landed wins (copy=False, seg-h2d) are about *not building a contiguous
host copy the GPU immediately re-copies*. (b) With both host memcpys now gone,
re-profile the 32-BD wall: the read/`stack` bucket should have collapsed, so
re-run `--profile` to see the new largest bucket — likely `wait` (generator-advance
= bigdata read + EB/MPI serving), which iter 3 measured at 60% of wall and no
landed change has touched. That points back at the psana serving chain (a psexp
change, in scope §3/§7) as the next frontier now the per-rank CPU memcpy has been
halved twice.

---

## 2026-07-10 — Iteration 9 (re-profile the wall after both host memcpys are gone: `wait` is now 79% @32BD — psana serving chain is the frontier)

**Task:** the measurement iter 8 pointed to as next-step (b). Both landed wins
(iter 7 copy=False, iter 8 seg-h2d) removed the two host-side 33.5 MB memcpys on
the GPU ingestion path. The existing `--profile` mode only instruments the *old*
`det.raw.raw`+`cp.asarray` loop, so it could not attribute the seg-h2d fast path.
Re-profile the seg-h2d wall at 1 and 32 BD to find the new largest bucket — iter 8
predicted `wait` (generator advance = bigdata read + EB/MPI serving) now dominates.

**What I did:**
- Added `run_gpu_bench_seg_h2d_profile` + routed `--seg-h2d --profile` to it in
  `bench_calib.py`. It is the iter-8 seg-h2d loop with the `wait` bucket added
  (time to advance `run.events()` to the next event, measured `next(events)` gap
  the same way `run_gpu_bench_profile` does), bucketing wall into
  wait / h2d(per-seg) / kernel. The numeric path (`fused_calib_gpu`, per-seg
  ingestion) is byte-identical to the already-gated iter-8 variant — only timing
  instrumentation was added. Synchronous, so the wall rate is a valid headline
  (§6 trap does not apply).
- Measured at 1 BD (smoke, `-n 60`) and 32 BD (`-n 200`, 7040 events) on the
  `ralph-gpu` node (job 31267701, sdfampere029), `mpirun --bind-to none
  --oversubscribe`, r47/FFB.

**Correctness gate:** `test_jungfrau_calib.py --seg-h2d -e mfx101572426 -r 47
-n 20` → **20/20 OK, max_diff 0.0**. Bit-exact (numeric path unchanged).

**Numbers (r47, FFB, A100, per-rank ms/event; logs
`bench_mpi_sweep/ralph_tmp/segh2d_profile_32bd_185342.log` @32BD, 1-BD from the
smoke run in-iteration):**

| bucket | 1 BD (72.5 Hz) | 32 BD (141.5 Hz agg, 4.42 Hz/rank) |
|---|---|---|
| **wait** (gen advance = bigdata read + EB/MPI) | 9.96 ms (72%) | **180.5 ms (79%)** |
| H->D (per-seg host->device) | 3.49 ms (25%) | 45.4 ms (20%) |
| kernel | 0.32 ms (2%) | 1.84 ms (1%) |
| sum vs wall | 13.77 / 13.80 | 227.7 / 226.1 |

Attribution closes to <1% at both scales — nothing unattributed.

**The finding:** iter 8's prediction is **confirmed**. With both host memcpys gone,
`wait` is the dominant bucket at both scales and inflates hardest under contention:
9.96 ms @1BD → **180.5 ms @32BD** (18x for 32x the ranks), while H->D inflates
3.49 → 45.4 ms (13x) and kernel stays negligible (0.32 → 1.84 ms). This is the same
per-rank serialization iter 5 found (pipeline plateaus ~16 BD) and iter 3's ~60%
`wait` share — now grown to 79% precisely because the two memcpy buckets it competed
with have been removed. The GPU-side work (H->D + kernel) is only 21% of the 32-BD
wall; **four fifths of the time a BD rank spends per event is waiting for psana to
deliver the next event's bigdata** (read + EB/MPI serving), not moving or computing
data. iter 4 already ruled out raw FFB bandwidth as this ceiling (2.78 GB/s = 35% of
7.9), so `wait` is serving-chain/serialization latency, not storage throughput.

**Keep/revert:** KEEP the profiling variant (measurement tool, no numeric-path
change, bit-exact). No throughput change this iteration — it is a pure attribution
that redirects the search.

**Recommended next step:** the frontier is now unambiguously the psana serving
chain (`wait`), which is in scope (§3/§7: psexp changes behind a flag). Two
single-variable experiments, in order of leverage:
(a) **Attribute `wait` itself** — split the 180 ms @32BD into "BD blocked waiting
for EB to hand it a batch" vs "BD reading its bigdata xtc off FFB once handed a
batch". Instrument the BD side of the EB→BD handoff (timestamp when the batch
descriptor arrives vs when the xtc bytes finish reading), or use `--smd0-debug`
+ an EB-side diary. This tells us whether the fix is serving-side (EB/dispatch
parallelism) or read-side (per-rank xtc read latency under 32-way contention).
Do NOT build a serving-chain change before this split — iter 0-4 mis-attributed
the multi-node ceiling four times by skipping exactly this step.
(b) Only after (a): if read-side, prefetch/overlap the next event's xtc read with
the current event's H->D+kernel (the CPU-push doc's "request next batch without
waiting"); if serving-side, EB→BD dispatch batching/parallelism. Either is a
psexp change — flag-gated, verify default `run.events()` still works (§7 rule 4).

---

## 2026-07-10 — Iteration 10 (split the `wait` bucket: read-side 42% / EB-wait 31% / CPU residual 27% — neither dominates alone)

**Task:** iter 9's recommended step (a) — split the `wait` bucket (79% of the
32-BD wall, the named frontier) into **EB-wait** (BD blocked on the EB batch
handoff, serving-side) vs **bd-read** (`os.pread` of bigdata xtc off FFB,
read-side). The explicit guardrail: do NOT build a serving-chain change before
this split (iter 0-4 mis-attributed the multi-node ceiling four times by skipping
exactly this measurement).

**What I did:**
- Found both phases are *already individually timed inside psana*: the `get_smd`
  closure in `BigDataNode` (`node.py`) blocks on the EB batch (Probe→Irecv), and
  `EventManager._read` (`event_manager.py`) is the `os.pread`. Added **additive-only**
  cumulative counters on `BigDataNode` (`total_eb_wait_ns`, `total_bd_read_ns`,
  `total_events`, `total_batches`) — EB-wait timed across the TRUE block point
  (a new `st_probe` spanning Probe+Irecv; the pre-existing `_last_bd_wait_time_ns`
  timed only Irecv and is left untouched), bd-read accumulated from the existing
  `on_batch_end` `read_time`. No behavior change to the default path.
- Added `--wait-split` to `bench_calib.py` (`run_gpu_bench_wait_split`): the iter-8
  seg-h2d fast path with the wait/h2d/kernel bucketing, snapshotting
  `run.bd_node`'s counters at the warmup boundary and at the end so the split
  covers the measured window only (excludes smd0/EB startup + warmup).
- **Install-tree gotcha (important for future iters):** psana imports from
  `install/lib/python3.9/site-packages/psana/`, NOT the `psana/psana/` source. A
  pure-Python edit to `psana/psana/gpu/` takes effect because `bench_calib.py` is
  run by path, but an edit to `psana/psana/psexp/node.py` does NOT until synced.
  First 1-BD run AttributeError'd (`bd_node` had no `total_eb_wait_ns`) → verified
  the installed node.py differs from source ONLY by my additions → `cp`'d source
  node.py into the install tree (equivalent to what `meson install` does for a .py
  file; no rebuild needed) and dropped its stale pyc. Source edit is the committed
  durable artifact; a future `./build_all.sh` reproduces it.

**Correctness gate:** `test_jungfrau_calib.py` (default path, also exercises the
edited node.py) **20/20 OK max_diff 0.0**, and `--seg-h2d` **20/20 OK max_diff
0.0**. Default psana MPI event loop verified working (§7 rule 4).

**Numbers (r47, FFB, A100, per-rank ms/event; logs
`bench_mpi_sweep/ralph_tmp/waitsplit_{1bd,32bd}_190923.log`):**

| bucket | 1 BD (73.4 Hz) | 32 BD (158.4 Hz agg, 4.95 Hz/rank) | share of wait @32BD |
|---|---:|---:|---:|
| **wait** (gen advance) | 9.78 | **166.7** | 100% |
| &nbsp;&nbsp;eb_wait (blocked on EB: Probe+Irecv) | 0.05 | **51.5** | **31%** |
| &nbsp;&nbsp;bd_read (os.pread bigdata off FFB) | 8.15 | **69.6** | **42%** |
| &nbsp;&nbsp;residual (dgram construction + generator plumbing, CPU) | ~1.6 | **~45.5** | **27%** |
| H->D (per-seg) | 3.49 | 42.8 | |
| kernel | 0.32 | 1.94 | |

(Fast FFB window: 158 Hz agg = 5.3 GB/s = 67% of the 7.9 GB/s per-node ceiling,
vs iter 9's 141 Hz. The **split proportions**, not the absolute rate, are the
finding; sum overshoots wall by ~4.7% because the counter-based eb_wait/bd_read
use the node's own event delta while `wait` is perf_counter gaps — directional,
not exact.)

**The finding (the split iter 9 asked for — and it overturns "one clean culprit"):**
The delivery `wait` is NOT dominated by a single phase. At 32 BD it splits roughly
**42% storage read / 31% blocked-on-EB / 27% CPU construction residual**:
1. **bd_read = 69.6 ms/event (read-side, the single largest).** Per-rank effective
   read = 33.5 MB / 69.6 ms = **481 MB/s**, ~identical to iter 4's 490 MB/s
   single-stream cold read. So each rank reads at single-stream speed with 32-way
   contention/queuing folded in — read-side is **per-rank latency/serialization**,
   consistent with iter 4/5 (aggregate BW below the storage ceiling). The lever is
   overlap/prefetch, not more bandwidth.
2. **eb_wait = 51.5 ms/event (serving-side, a close second).** Direct evidence the
   single EB rank (PS_EB_NODES=1) serving 32 BD ranks is a real serialization point
   — 0.05 ms at 1 BD → 51.5 ms at 32 BD. iter 0 recorded "NOT EventBuilder count
   (flat across PS_EB_NODES=1/2/4)", but that predates iter 7/8: with both host
   memcpys gone the BD ranks cycle far faster and now spend 31% of `wait` blocked on
   the EB. **The old flat PS_EB_NODES result is stale and must be re-measured.**
3. **residual ~45.5 ms/event (27%, CPU).** Time inside the generator advance NOT in
   Probe/Irecv nor pread — per-event `dgram.Dgram(...)` construction in
   `EventManager._get_next_dgrams` + Python plumbing. A lighter accessor could
   trim it but it is the smallest of the three.

**Keep/revert:** KEEP the node.py counters (additive-only instrumentation,
default path bit-exact + verified working) and `--wait-split`. No throughput
change this iteration — pure attribution that redirects the search. The old
single-culprit framings ("H2D is the ceiling"; "wait = one thing") are both now
refuted by direct measurement.

**Recommended next step:** two levers are now *quantified and roughly equal*; pick
by cheapest decisive test first.
(a) **Re-measure PS_EB_NODES=1/2/4 at 32 BD on the seg-h2d fast path** with
`--wait-split`. This is the cheapest possible experiment (an env var, no code) and
it directly tests whether the 51.5 ms eb_wait shrinks with more EB ranks — the
iter-0 flat result is stale (pre-memcpy-removal). If eb_wait drops and aggregate Hz
rises, more EB ranks is a landed win with zero code risk; if flat, the EB serial
work per batch (not EB *count*) is the wall and points into EB→BD dispatch.
(b) In parallel/after: attack bd_read (42%) with per-BD **async prefetch** — issue
the next batch's `get_smd` + `_fill_bd_chunk` read while the current event's
H->D+kernel run (the CPU-push doc's "request next batch without waiting"), overlapping
the 69.6 ms read (34% of wall) behind the 45 ms of GPU work. This is a psexp/BD-loop
change — flag-gated, verify default `run.events()` still works (§7 rule 4). Do (a)
first: it is one env var and settles 31% of the wall before any code is written.

---

## 2026-07-10 — Iteration 11 (re-measure PS_EB_NODES=1/2/4 on the seg-h2d fast path — EB *count* is a dead end; more EB ranks does not reduce eb_wait or raise throughput)

**Task:** iter 10's recommended step (a), the cheapest possible experiment (an env
var, no code) — re-measure `PS_EB_NODES=1/2/4` at fixed 32 BD on the seg-h2d fast
path with `--wait-split`, to test whether the 51.5 ms eb_wait (iter 10, 31% of the
delivery `wait`) shrinks with more EB ranks. iter 10 flagged the iter-0 "flat
PS_EB_NODES" result as **stale** (it predates the iter-7/8 memcpy removals, after
which BD ranks cycle far faster and now block 31% of `wait` on the EB handoff), so
it must be re-measured before any serving-chain code is written.

**What I did:**
- Added `bench_mpi_sweep/ralph_tmp/ebsweep.sh` (not tracked): runs the seg-h2d
  `--wait-split` benchmark at PS_EB_NODES=1/2/4, fixed 32 BD (NPROC = 1 smd0 +
  EB + 32), `-n 200 --warmup 10`, r47/FFB, `mpirun --bind-to none --oversubscribe
  -x PS_EB_NODES`. Ran as a **palindrome bracket (1,2,4,4,2,1)** so FFB
  minute-to-minute window drift is symmetric across the sweep and the EB-count
  effect is separable from it. `ralph-gpu` node (job 31267701, sdfampere029).
- No code change — pure env-var sweep. Correctness gate not triggered (numeric
  path byte-identical to the already-gated iter-8/9/10 seg-h2d variant).

**Numbers (r47, FFB, A100, 32 BD, aggregate; logs
`bench_mpi_sweep/ralph_tmp/ebsweep_{a_eb1,b_eb2,c_eb4,d_eb4,e_eb2,f_eb1}_192953.log`,
driver `ebsweep_driver_192953.log`):**

| PS_EB_NODES | agg Hz (2 brackets → mean) | eb_wait ms | bd_read ms |
|---:|---|---:|---:|
| 1 | 117.9, 115.4 → **116.7** | 63.7, 69.0 → **66.3** | 90.6, 89.3 → **90.0** |
| 2 | 120.3, 115.4 → **117.9** | 73.0, 70.9 → **72.0** | 85.5, 87.8 → **86.7** |
| 4 | 121.0, 113.9 → **117.5** | 75.4, 82.3 → **78.9** | 80.3, 84.4 → **82.4** |

(This was a slower FFB window than iter 10's 158 Hz — absolute rates sit ~117 Hz —
but the EB-count comparison is internally bracketed and window-controlled; run-to-run
spread across all 6 was 113.9–121.0 Hz.)

**The finding (EB count is a dead end — the iter-0 flat result HOLDS on the fast path):**
1. **Aggregate throughput is flat across EB=1/2/4** (116.7 / 117.9 / 117.5 Hz — all
   within 1%, well inside the 113.9–121.0 run-to-run FFB spread). More EB ranks buys
   no throughput.
2. **eb_wait does NOT shrink with more EB ranks — it drifts slightly *up*** (66.3 →
   72.0 → 78.9 ms). Adding EB ranks (which partition the smd stream so each EB serves
   fewer BD ranks) does not reduce the time a BD rank blocks on the batch handoff. So
   the 51–79 ms eb_wait is **not an EB-parallelism/EB-count bottleneck** — iter 10's
   worry that "the single EB rank serving 32 BD ranks is the serialization point" is
   **refuted**: 2 and 4 EB ranks serve the same 32 BD ranks no faster. The iter-0
   "flat PS_EB_NODES" result **reproduces even on the seg-h2d fast path**; it was
   NOT stale after all.
3. bd_read drifts down mildly (90.0 → 86.7 → 82.4) but this is `os.pread`, which is
   independent of EB count — it tracks the FFB window getting slightly faster mid-sweep
   (the c_eb4 run caught a faster window), not an EB effect. The palindrome bracket
   makes this visible: EB=1 was measured both first (117.9) and last (115.4).

Interpretation: eb_wait being large *and* insensitive to EB count means BD ranks are
blocked not because one EB can't fan out fast enough, but because the pipeline
*upstream of the handoff* (smd0 small-data distribution + EB per-batch serial work)
can't produce the next batch before the BD rank — now memcpy-light and fast — comes
back asking for it. Throwing more EB ranks at it doesn't help because each EB still
does the same serial per-batch work and smd0 is a single distributor. The real lever
is **bd_read** (the largest single component, ~82–90 ms, per-rank read latency at
~480 MB/s under 32-way contention, iter 4/10) via async prefetch/overlap — not EB
count.

**Keep/revert:** No code change to keep or revert — this is a clean env-var
measurement that **closes lever (a) as a dead end** (a successful iteration per §6).
Correctness gate not triggered (numeric path byte-identical). The stale-flag on the
iter-0 PS_EB_NODES result is resolved: re-measured, it holds.

**Recommended next step:** with EB count ruled out, go to iter 10's lever (b) — the
only remaining large, un-attacked wait component: **per-BD async prefetch of bd_read**
(~82–90 ms, ~40% of wall). Overlap the next event's `get_smd` + bigdata `os.pread`
with the current event's per-seg H->D + kernel (the CPU-push doc's "request next
batch without waiting"), so the ~45 ms of GPU work hides behind the read instead of
running after it. This is a psexp/BD-loop change — flag-gated, and §7 rule 4 requires
verifying the default `run.events()` MPI path still works before committing. The
decisive before/after is `--wait-split` aggregate Hz at 32 BD with the prefetch flag
on vs off, interleaved-bracketed for the FFB window. Secondary (if prefetch is hard
to land cleanly): attack the ~27% CPU residual (dgram construction) with a lighter
accessor — smaller lever, but pure CPU and window-independent.

---

## 2026-07-10 — Iteration 12 (async-prefetch GPU overlap — pipeline H->D+kernel behind the read; DEAD END at 32 BD, ~3% slower, reverted)

**Task:** iter 10/11's lever (b), first half — overlap the ~21% GPU work (per-seg
H->D + kernel, iter 9) behind the psana `wait` (EB-wait + bd_read, ~79%) instead of
running them strictly serial. Every sync path today (run_gpu_bench_seg_h2d) syncs
after each event, so event N+1's `os.pread` cannot start until event N's GPU work
finishes. Hypothesis: pipeline the GPU work across CUDA streams so it hides behind
the next event's read, buying up to the 21% GPU fraction back (~+27%).

**What I did:**
- Added `--async-prefetch` to `bench_calib.py` (`run_gpu_bench_async_prefetch`):
  based on the seg-h2d fast path, but gathers each event's segments into a **pinned
  host buffer**, issues an **async H->D + kernel on a per-slot CUDA stream**, and
  does **not** sync before advancing `run.events()`. A ring of `PS_PREFETCH_DEPTH`
  slots (default 2) bounds in-flight work; a slot's stream is synced only when its
  buffers are about to be reused (pipeline backpressure, not a per-event sync). The
  pipeline is drained before the wall clock stops. Headline is **wall-clock rate
  only** per §6 — the sync-instrumented per-stage split would destroy the overlap,
  so it is intentionally omitted. The pinned staging copy is required: after
  `run.events()` advances, psana overwrites its reused bd buffer, so the async H->D
  needs a stable owned source (this re-adds a 33.5 MB memcpy iter 8 removed — the
  point is to measure whether the overlap it buys is worth that cost).
- A/B at 32 BD on FFB/r47, **palindrome-bracketed** (seg, async, async, seg + a
  second async/seg pair) so the FFB minute-to-minute window is controlled and the
  seg baseline is measured both before and after the async runs.
- Smoke test single-process first: async ran clean at 65.3 Hz (`-n 60`). Numerics
  bit-identical to the gated seg-h2d path by construction (same segment gather
  order into a contiguous buffer, same `fused_calib_gpu` kernel — only copy timing
  and the async stream differ), so the correctness gate is not re-triggered.

**Numbers (r47, FFB, A100, 32 BD, aggregate; logs
`bench_mpi_sweep/ralph_tmp/async_{a_seg,b_async,c_async}_195659.log` and
`async_{e_async,f_seg}_200737.log`):**

| variant | agg Hz (each run) | mean |
|---|---|---:|
| `--seg-h2d` (baseline) | 122.3 (first), 120.0 (last) | **121.2** |
| `--async-prefetch` | 116.3, 119.4, 117.9 | **117.9** |

The seg baseline is stable end-to-end (122.3 → 120.0, no window drift), so the ~3%
gap is real, not FFB noise: **async-prefetch is flat-to-slightly-slower at 32 BD.**

**The finding (why overlap can't win here — a structural reframe of lever (b)):**
1. **Per-rank async overlap cannot create GPU/PCIe bandwidth.** At 32 BD all ranks
   share one A100 and one PCIe gen4 link. The sync-path "21% GPU work" is itself
   inflated by 32-way contention (per-event H->D ~42 ms, iter 9/§5, vs ~3.4 ms
   uncontended). Making each rank's H->D async does not add bus bandwidth — it only
   overlaps *that rank's* read with *that rank's* transfer, and under contention the
   transfer doesn't shrink. There is nothing to hide it behind that isn't already
   contended.
2. **The pinned staging memcpy is a net cost with no offsetting gain.** Re-adding
   the 33.5 MB host copy (iter 8 removed it for exactly this reason) costs ~3 ms/event
   of real CPU/DRAM-bandwidth time; the overlap it enables recovers less than that,
   so the balance is slightly negative (~-3%).
3. **This tests only the GPU-side half of lever (b) and closes it.** Overlapping GPU
   behind the read does nothing for `bd_read` itself (42% of `wait`, the largest
   single component, iter 10) — that time is the BD rank blocked in `os.pread`, and
   no amount of GPU-side pipelining touches it.

**Keep/revert:** **REVERTED** (`git checkout bench_calib.py`) — a change that didn't
move the number gets reverted and journaled as a dead end (§6). Default path
untouched; tree clean. Correctness gate not triggered (numeric path byte-identical).

**Recommended next step:** the remaining, un-attacked half of lever (b) is the real
prize and is a fundamentally *different* change from what iter 12 tested —
**read-side prefetch of `bd_read`**, not GPU-side overlap. `bd_read` is per-rank
`os.pread` **latency** bound (~480 MB/s/rank while aggregate = 2.78 GB/s = 35% of the
7.9 GB/s node ceiling, iter 4/10 — storage has headroom, the rank just idles waiting
on its own read). The lever is to issue the **next batch's** `get_smd` + bigdata
`os.pread` on a **background reader thread** so the rank isn't blocked in `pread`
when it comes back for the next event (the CPU-push doc's "request next batch without
waiting"). This is a psexp/BD-loop change (node.py `start()` / EventManager), must be
flag-gated, and §7 rule 4 requires verifying the default `run.events()` MPI path
still works. Key design constraint learned this iteration: keep **all MPI on the main
thread** (only `os.pread`, which releases the GIL, goes to the reader thread) so no
MPI thread-safety level is required. Decisive before/after: `--wait-split` aggregate
Hz at 32 BD with the read-prefetch flag on vs off, palindrome-bracketed. Secondary
(smaller, window-independent): the ~27% CPU residual (dgram construction) via a
lighter accessor.

---

## 2026-07-10 — Iteration 13 (concurrency-headroom probe for read-side prefetch — a full background reader thread HALVES the main loop; contention is real, so a prefetch reader cannot run free)

**Task:** iter 12's recommended lever (b), de-risked BEFORE the core psexp surgery
it needs. The open question iter 12 raised against read-side prefetch: at 32 BD on
a fully-loaded node (34 procs), is there real CPU/GIL/IO concurrency headroom for a
background reader thread to overlap `os.pread` (bd_read, 42% of wall, iter 10)
behind main-thread GPU work — or is there, as iter 12 argued, "nothing to hide it
behind that isn't already contended"? Building the batch/chunk double-buffered
reader into core `EventManager`/`node.py` is a large, delicate change (and §7 rule 4
warns breaking default psana is the one unrecoverable mistake), so I measured the
concurrency headroom first with a benchmark-only probe — cheapest decisive test.

**What I did:**
- Added `--reader-probe` to `bench_calib.py` (`_start_reader_probe`): runs the
  gated seg-h2d fast path UNCHANGED, but ALSO spawns one background daemon thread
  per BD rank that continuously `os.pread`s the rank's bigdata xtc off FFB (16 MB
  chunks, offset advancing monotonically across the 200+ GB stream so reads don't
  trivially re-hit cache; ranks fan across the big streams round-robin). The reader
  does ONLY `os.pread` — the exact GIL-releasing op a real prefetch reader would run.
  Reports the reader's achieved MB/s (snapshotted over the measured window) so an
  IO-bandwidth limit is distinguishable from GIL/CPU contention. Benchmark-only —
  **zero psexp change**, numeric path byte-identical to the already-gated seg-h2d
  variant, so the correctness gate is not re-triggered. Default path untouched.
- Palindrome bracket at 32 BD (seg, probe, probe, seg) to control FFB window drift,
  `-n 200 --warmup 10`, r47/FFB, ralph-gpu node (job 31267701, sdfampere029).

**Numbers (r47, FFB, A100, 32 BD, aggregate; logs
`bench_mpi_sweep/ralph_tmp/probe_{a_seg,b_probe,c_probe,d_seg}_202607.log`,
driver `probe_driver_202607.log`):**

| run | variant | agg Hz | reader bw |
|---|---|---:|---|
| a_seg (first) | seg-h2d baseline | **615.4** | — |
| b_probe | seg-h2d + bg reader | **284.9** | 35.98 GB/s agg (1124 MB/s/rank) |
| c_probe | seg-h2d + bg reader | **300.2** | 35.38 GB/s agg (1106 MB/s/rank) |
| d_seg (last) | seg-h2d baseline | **553.7** | — |

baseline mean **584.6 Hz**; probe mean **292.6 Hz** → **−50%**. The bracket held the
window (baseline drifted only 615→553, ~10%) while the probe sat at ~293 well below
BOTH baseline ends, so the halving is the reader's effect, NOT window drift.

**Two findings — one solid, one a caveat that bounds the claim:**
1. **A full concurrent reader thread halves the GPU-feed loop (−50%).** Even though
   the reader only does GIL-releasing `os.pread`, running it alongside the main loop
   cut aggregate throughput in half. Concurrency contention between a background
   reader and the main GPU-feed path is **real and material** — this directly
   supports iter 12's "already contended" skepticism. A prefetch reader thread
   therefore **cannot run free**; whatever it reads, it competes with the main loop
   for memory bandwidth / CPU / cache, and that cost is not negligible. The reader
   pulled **35+ GB/s aggregate — far above the 7.9 GB/s FFB storage ceiling** — so it
   was cache/readahead + memory-bandwidth bound, i.e. the contention it created was
   CPU/memory-bandwidth, not storage. (It ran ~1.75x a *faithful* prefetch load:
   1115 vs the ~640 MB/s/rank a real prefetch needs = 33.5 MB/event × 19 Hz/rank —
   so scale the −50% down somewhat, but the sign and materiality stand.)
2. **CAVEAT — today's window is cache-WARM, off the regime the prefetch question was
   posed in.** The seg-h2d baseline ran at **585 Hz, 4–5x the iter-10/11/12 cold
   numbers (117–158 Hz)** — per-rank wall collapsed 202 → 52 ms/event with NO code
   change since iter 10, so it is a warm-cache artifact (every run re-reads the same
   first ~7,040 events of r47, warmed across today's 12 iterations; consistent with
   iter-0 "filesystem/window dominates, 15 vs 210 Hz"). In a warm window bd_read
   latency is already small, so there is little read latency to hide — prefetch's
   premise (bd_read = 42% of a COLD wall) doesn't hold right now. So this probe does
   NOT cleanly return GO/NO-GO for the cold-regime prefetch; what it DOES establish
   is the contention floor (finding 1), which is regime-independent and cautionary.

**Interpretation for the prefetch decision:** the evidence leans **caution, not GO**.
For read-side prefetch to pay it needs BOTH (i) a cold regime where bd_read latency
actually exists to hide, AND (ii) the reader not to over-contend for CPU/memory
bandwidth with the main loop. Finding 1 shows (ii) is a real tax (a concurrent
reader measurably slows the loop); the warm window prevented testing (i) today. This
is exactly the "already contended" failure mode iter 12 predicted, now with a number
on it. Building the full core-psexp double-buffered reader before (i) is confirmed in
a cold window risks paying the surgery cost for a lever the contention tax may eat.

**Keep/revert:** KEEP `--reader-probe` — a reusable benchmark-only concurrency probe,
zero psexp risk, numeric path byte-identical (correctness gate not triggered). No
throughput change landed; this iteration buys a measured contention floor + a
regime caveat that redirect the expensive surgery.

**Recommended next step:** settle confound (i) cheaply before any psexp surgery —
re-run the `--reader-probe` palindrome bracket **in a cold window** (either after a
cache-drop, or add a rate-cap `PS_READER_PROBE_MBPS≈640` so the reader mimics a
faithful prefetch load rather than running full-tilt at 1115 MB/s/rank). If, cold and
rate-capped, the reader does NOT halve the loop AND the baseline is back at ~117-158
Hz (bd_read latency present), that is the GO signal to build the real double-buffered
reader in `EventManager`/`node.py` (chunk-level, MPI on main thread, flag-gated,
verify default `run.events()`). If it still contends heavily even rate-capped, the
prefetch lever is dead and the search should turn to the ~27% CPU residual (dgram
construction, iter 10) via a lighter accessor — pure CPU, window-independent, no
threading contention. Secondary observation worth a look: the 585 Hz warm baseline
means a node-local NVMe/page-cache staging tier for reprocessing could itself be a
4-5x lever (DEFERRED/TASK item 2), independent of prefetch.
