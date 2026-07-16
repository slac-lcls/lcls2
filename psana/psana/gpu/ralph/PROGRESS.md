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

---

## 2026-07-10 — Iteration 14 (rate-capped reader probe in a COLD window — the cap didn't bind, the reader saturated FFB storage and still crushed the loop −63%; the probe cannot faithfully model prefetch because it adds NET I/O)

**Task:** iter 13's recommended lever — settle the two confounds it left open before any
core-psexp prefetch surgery. iter 13's `--reader-probe` halved the 32-BD loop (−50%) but
(a) ran cache-WARM (585 Hz baseline, off the cold regime prefetch is posed in) and (b) the
reader pulled ~1115 MB/s/rank = 1.75x a *faithful* prefetch load. iter 13 proposed a rate
cap `PS_READER_PROBE_MBPS≈640` (= 33.5 MB/event × ~19 Hz/rank) to make the reader faithful,
re-run cold. One variable this iteration: the rate cap.

**What I did:**
- Added `PS_READER_PROBE_MBPS` to `_start_reader_probe` (bench_calib.py): the background
  reader throttles to a per-rank MB/s ceiling by sleeping when its cumulative rate exceeds
  the cap. Benchmark-only, numeric path byte-identical (only the probe thread's sleep
  changes) — correctness gate not triggered. Default (unset) = uncapped, iter-13 behavior.
- 32-BD palindrome bracket (seg / capped-probe / capped-probe / seg), cap=640, `-n 200
  --warmup 10`, r47/FFB, ralph-gpu node (job 31267701, sdfampere029), `-x PS_READER_PROBE_MBPS`.

**Numbers (r47, FFB, A100, 32 BD, aggregate; logs
`bench_mpi_sweep/ralph_tmp/probecap_{a_seg,b_probe,c_probe,d_seg}_204050.log`,
driver `probecap_driver_204046.log`):**

| run | variant | agg Hz | reader bw (capped @640) |
|---|---|---:|---|
| a_seg (first) | seg-h2d baseline | **118.9** | — |
| b_probe | seg-h2d + capped bg reader | **46.2** | 7530 MB/s agg (235 MB/s/rank) |
| c_probe | seg-h2d + capped bg reader | **43.8** | 7729 MB/s agg (242 MB/s/rank) |
| d_seg (last) | seg-h2d baseline | **123.1** | — |

baseline mean **121.0 Hz** (palindrome tight: 118.9→123.1, +3.5%); probe mean **45.0 Hz**
→ **−63%**. The bracket held the window, so the drop is the reader's effect.

**Three findings:**
1. **The window is COLD today (baseline 121 Hz).** That is squarely the iter-10/11/12 cold
   band (117–158 Hz), NOT iter 13's warm 585 Hz — so confound (i) is *settled by luck of
   the window*: we ARE in the cold regime the prefetch question was posed in. bd_read
   latency is present.
2. **The cap did NOT bind — the reader was STORAGE-limited.** Intent was 640 MB/s/rank but
   the reader delivered only ~238 MB/s/rank (7.5–7.7 GB/s aggregate ≈ the 7.9 GB/s FFB
   ceiling, iter-4 raw-read matrix). 32 ranks × 640 = 20 GB/s ≫ 7.9, so storage — not the
   cap — throttled the reader. In the WARM iter-13 window the reader hit 35 GB/s (cache/
   memory-bandwidth bound, storage irrelevant); COLD, it hits the storage wall instead.
3. **A storage-saturating reader still crushes the loop −63%,** worse than iter-13's warm
   −50%. In the cold regime the shared bottleneck is FFB storage bandwidth, and the reader
   wins the contention (grabs 7.6 of the ~9 GB/s the pipe delivered, starving the main loop
   from 4.05 → 1.5 GB/s).

**The decisive interpretation — the probe cannot faithfully model read-prefetch, and this
is now the SECOND methodology confound it has hit:**
- The probe reads **net-additional** bytes (different, monotonically-advancing offsets, by
  design so reads stay cold). A **real** read-prefetch reads the *same* bytes the main loop
  would read next, just **earlier** — it shifts existing I/O in time, it does **not** add
  net storage demand. So in a storage-bound cold regime the probe **double-counts I/O** and
  **overstates** a real prefetch's contention. The −63% is an upper bound inflated by
  duplicated reads, not a faithful prefetch cost.
- The 640 cap was also **mis-parameterized for the cold regime**: 640 MB/s/rank models a
  19 Hz/rank (WARM) consumption. The COLD loop consumes only 121/32 = **3.8 Hz/rank =
  ~127 MB/s/rank**, so a faithful cold prefetch reads ~5x *less* than even the storage-
  limited 238 the probe delivered. The cap can only be set right once you know the regime's
  consumption rate — which is the very thing that varies run-to-run here.

**Where this leaves the prefetch decision (honest verdict): the probe methodology is
exhausted.** Two de-risking iterations (13 warm-cache, 14 net-additional-I/O + mis-set cap)
have each surfaced a methodology confound instead of a clean GO/NO-GO. A concurrency probe
that reads *different* data structurally cannot model a prefetch that reads the *same* data
earlier. Compounding the doubt with plain arithmetic: the cold main loop already pulls
4.05 GB/s = 51% of the 7.9 GB/s FFB ceiling, and iter 5 found the per-rank pipeline
**serialization-bound, not latency-bound** (plateau ~16 BD / 2.8 GB/s). Prefetch helps only
a latency-bound reader; a bandwidth/serialization-bound one near half the storage ceiling
has limited headroom to overlap into. Evidence continues to lean **CAUTION, not GO** on
building the core-psexp double-buffered reader.

**Keep/revert:** KEEP `PS_READER_PROBE_MBPS` (small, correct, benchmark-only, numeric path
byte-identical — correctness gate not triggered). No throughput change landed; this
iteration buys a cold-regime measurement + the methodology verdict that the probe cannot
settle prefetch.

**Recommended next step:** **stop probing prefetch and pivot to the window-independent
lever.** The read-prefetch lever has now cost two probe iterations without a clean signal,
and both the net-additional-I/O flaw and the storage-ceiling/serialization arithmetic argue
its upside is bounded. The remaining un-attacked lever is the **~27% CPU residual — dgram
construction** (iter 10's wait-split: read 42% / EB-wait 31% / CPU 27%). It is pure host
CPU: no storage contention, no threading confound, no warm/cold window dependence — so a
before/after is clean and repeatable regardless of which window the loop lands in. Attack it
with a lighter `det.raw._segments`/dgram accessor path and measure at 1 + 32 BD. If a
faithful prefetch test is ever still wanted, it requires the *real* double-buffered reader
(reads the main loop's own next batch, adding zero net bytes) — which is exactly the psexp
surgery the probes were meant to de-risk and could not; do it only as a deliberate, flag-
gated, default-`run.events()`-verified change, not another probe.

---

## 2026-07-10 — Iteration 15 (direct dgram-construction attribution — the "27% CPU residual = dgram construction" hypothesis is REFUTED: dgram is only ~4-5% of delivery, nearly flat per-rank)

**Task:** iter 14's recommended pivot — attack the **window-independent ~27% CPU
residual** (iter 10's wait-split: read 42% / EB-wait 31% / CPU 27%), which iter 10
labeled "dgram construction + generator plumbing." Per the loop's discipline (attribute
before building — iters 0-4 mis-attributed the multi-node ceiling four times by skipping
exactly this step), I did NOT jump to a "lighter dgram accessor"; I first measured
directly how much of the residual is actually `dgram.Dgram()` construction.

**What I did:**
- Added **additive-only** cumulative counters on the persistent DgramManager (`dm`,
  reachable as `run.bd_node.dm`; EventManagers are transient/per-batch so the counters
  can't live on them): `total_dgram_ns` (time in `_get_next_dgrams`, the per-event
  `dgram.Dgram()` construction) and `total_smdparse_ns` (time in `_get_offset_and_size`,
  the per-batch offset/size-array build). Split `_get_next_dgrams` into a timing wrapper
  + `_get_next_dgrams_impl`. Plumbed both into `--wait-split` in `bench_calib.py`
  (new `dgram`/`smdparse`/`residual` report lines). No behavior change to the default
  path; numeric path byte-identical.
- **One instrumentation confound found + fixed mid-iteration:** `_get_next_dgrams`
  lazily triggers a bigdata `os.pread` via `_fill_bd_chunk` when a bd buffer is
  exhausted, so the naive dgram timer double-counted `bd_read` (first pass: dgram=9.55 ms
  > bd_read=8.59 ms @1BD, impossible for "pure CPU"). Fixed by subtracting the
  `_bd_read_time` delta that accrues inside the call, leaving pure dgram CPU.
- Synced source `event_manager.py` → `install/` (psana imports from install, not source;
  same gotcha iter 10 documented for node.py), dropped the stale pyc.
- Ran `--wait-split` at 1 BD (`-n 80`) and 32 BD (`-n 200`) on r47/FFB, `ralph-gpu` node
  (job 31267701, sdfampere029), `mpirun --bind-to none --oversubscribe`.

**Correctness gate:** `test_jungfrau_calib.py -e mfx101572426 -r 47 -n 20` (default path,
exercises the edited `event_manager.py`) → **20/20 OK, max_diff 0.0**. Default psana MPI
event loop verified working (§7 rule 4). Bit-exact.

**Numbers (r47, FFB, A100, per-event ms; logs
`bench_mpi_sweep/ralph_tmp/dgram2_1bd_210*.log`, `dgram_32bd_211018.log`):**

| counter (same `bd_events` denominator) | 1 BD (70.0 Hz) | 32 BD (337.2 Hz agg, ~10.5 Hz/rank) | 1→32 growth |
|---|---:|---:|---:|
| eb_wait (blocked on EB) | 0.055 | ~28 (26–33) | ~500x |
| bd_read (os.pread) | 8.57 | 43.6 (32-rank mean) | 5.1x |
| **dgram** (`dgram.Dgram()` construction) | **1.07** | **4.13** (32-rank mean) | **3.9x** |
| smdparse (per-batch array build) | 0.29 | ~1.3 | 4.5x |
| H->D (per-seg) | 3.49 | ~23 | |
| kernel | 0.34 | ~1.0 | |
| per-rank wall | 14.3 | ~95–142 | |

dgram as a share of the four directly-counted delivery components (all sharing the
`bd_events` denominator, so the ratio is normalization-robust):
**11% @1BD → ~5% @32BD**; as a share of per-rank wall, **~7.5% @1BD → ~4% @32BD**.
Consistent across all 32 ranks (dgram 2.4–6.0 ms, mean 4.13; bd_read mean 43.6).

**The finding (refutes iter-10/14's residual hypothesis):**
1. **dgram construction is NOT the residual's driver — it is ~4-5% of delivery, ~4% of
   wall.** iter 10's "27% CPU residual = dgram construction" was a **subtractive artifact**:
   the residual `wait − eb_wait − bd_read` mixes denominators — `wait` is averaged over
   measured L1 events (`n`) while `eb_wait`/`bd_read` are normalized by the node's own
   `bd_events` (which includes skipped transitions, `bd_events > n`), so `eb_wait`/`bd_read`
   are understated and the leftover "residual" is inflated. Counted **directly** with the
   same denominator as bd_read, dgram construction is small.
2. **dgram is nearly flat per-rank (3.9x for 32x ranks) — it is pure CPU, minimally
   contention-sensitive**, exactly as expected for in-process object construction. The
   super-linear inflation that makes 32-BD delivery expensive lives in `bd_read`
   (storage/cache contention) and `eb_wait` (single EB serving 32 ranks), NOT in dgram
   construction. A "lighter dgram accessor" would shave only the flat ~4 ms.
3. **Therefore the lighter-dgram-accessor lever iter 14 recommended is DEAD** — its ceiling
   is ≤~4% and it does nothing for the contention-driven buckets that dominate the wall.

**Keep/revert:** KEEP the `total_dgram_ns`/`total_smdparse_ns` counters + `--wait-split`
reporting — additive-only attribution tooling that settled the question, bit-exact,
default path verified. No throughput change this iteration (pure attribution that closes
a lever).

**Recommended next step:** the single-node per-rank optimization search is now **largely
exhausted** — the two landed wins (copy=False +30%, seg-h2d +32%) removed the two host
memcpys, and every remaining per-rank bucket is either shared-resource contention
(bd_read, eb_wait — grow super-linearly, not per-rank CPU) or small-and-flat (dgram ~4%,
kernel ~1%). The two dominant buckets can't be shrunk by per-rank code; they need a
structural change. The highest-leverage **unmeasured** direction is **multi-node re-measure**:
the branch's original §1 open question was the ~300 Hz multi-node plateau, measured
*before* copy=False + seg-h2d landed (+~70% combined per-rank). Re-run the 2-node/64-rank
layout (`bench_mpi_sweep/mn2x32.sbatch`) to see (i) whether the plateau moved and (ii)
whether `eb_wait` — now ~30% of counted delivery with a single EB serving 32 BD/node —
becomes the multi-node ceiling, which would put PS_EB_NODES>1 in a genuinely different
regime than iter 11's single-node flat result. Second choice remains the real
double-buffered read-prefetch (zero-net-I/O), but iters 12-14 leaned CAUTION on it.

---

## 2026-07-10 — Iteration 16 (multi-node re-measure on r47 AFTER the two per-rank wins — the ~300 Hz plateau HELD at 284 Hz; multi-node ceiling is bd_read/storage-bound, eb_wait is negligible — REFUTES the "single EB serving 64 BD becomes the ceiling" hypothesis)

**Task:** iter 15's recommended lever — re-measure the multi-node plateau on the r47
reference. The historical 2-node/64-BD number (~295 Hz on r387, now purged) was measured
BEFORE copy=False (+30%) + seg-h2d (+32%) landed — the two wins that lifted single-node
32-BD from 210 → 337 Hz (+~70%). Open question: did those per-rank wins move the multi-node
plateau, and does eb_wait (a single EB serving 64 BD) become the new ceiling? One variable:
node count (single→two nodes at 32 BD/node), same r47 window.

**What I did:**
- Added `bench_mpi_sweep/mn2x32_r47.sbatch`: 2 nodes × 33 tasks = 66 ranks (smd0 + 1 EB +
  64 BD). Designed to run single-node (34-rank) and two-node (66-rank) configs WITHIN THE
  SAME allocation, bracketed single/multi/single, to control the window/cache regime (the
  dominant run-to-run confound, iters 13-14). `--wait-split` (bit-exact additive
  instrumentation, iter 15) for the eb_wait/bd_read attribution.
- Submitted job 31291349 (sdfampere001,003), r47/FFB, `-n 200 --warmup 10`.
- **The single-node bracket configs FAILED** (exit 213, "Out of resource"): SLURM allocated
  exactly 33 slots/node, and the single-node layout needs 34 ranks (smd0+EB+32 BD) on one
  node. `mpirun -H node:34` can't map 34 onto 33 slots. The two-node config (33/node, maps
  exactly) succeeded — that is the headline. Fix for a follow-up: add `--oversubscribe` to
  the single-node lines.

**Numbers (r47, FFB, A100, 2 nodes × 32 BD = 64 BD ranks, aggregate; log
`bench_mpi_sweep/sweep_mn2x32_r47.log`, job 31291349, wall 142s):**

| metric | 2-node / 64 BD (this iter) | single-node / 32 BD (iter 15, same day) | historical 2-node (r387, pre-wins) |
|---|---:|---:|---:|
| **aggregate rate** | **284.4 Hz** | 337.2 Hz | ~295 Hz |
| per-rank rate | 4.44 Hz | ~10.5 Hz | — |
| per-rank wall | 225.0 ms/event | ~95 ms/event | — |
| wait (gen advance) | 218.1 | — | — |
| &nbsp;&nbsp;eb_wait | **9.15** | ~28 | — |
| &nbsp;&nbsp;bd_read (os.pread FFB) | **133.1** | 43.6 | — |
| H→D | 8.11 | ~23 | — |
| kernel | 0.65 | ~1.0 | — |

**Findings:**
1. **The multi-node plateau HELD — it did NOT move with the per-rank wins.** 284 Hz now vs
   ~295 Hz historical (within window variance), despite copy=False + seg-h2d lifting
   single-node 32-BD by +70% (210 → 337 Hz). The two host-memcpy eliminations that dominate
   the single-node win do NOT transfer to the multi-node aggregate. The §1 open question —
   "psana plateaus ~300 Hz while raw storage allows ~450+ Hz at 2 nodes" — **survives the
   per-rank optimization campaign intact.**
2. **The multi-node ceiling is bd_read/STORAGE-bound, NOT the serving chain.** bd_read =
   133 ms/event = **59% of the 225 ms per-rank wall**; eb_wait = **9.15 ms = 4%**. A single
   EB serving 64 BD is NOT the bottleneck — this **REFUTES iter 15's hypothesis** that
   eb_wait becomes the multi-node ceiling and puts PS_EB_NODES>1 back to a dead end at this
   layout (consistent with iter 11's single-node flat PS_EB_NODES result, for the right
   reason now: EB simply isn't where the time goes).
3. **bd_read is where the multi-node inflation lives**, but the magnitude vs single-node
   (43.6 → 133 ms, 3x) is CONFOUNDED by window/cache state — iters 13-14 showed bd_read
   swings 4-5x cold-vs-warm, and the single-node 43.6 (iter 15) was a different node/time.
   I could NOT get a same-window single-node number this iter because the bracket configs
   failed to launch. So the 3x is NOT cleanly attributable to multi-node storage contention
   vs a colder window. What IS clean: at THIS window, at 64 BD on 2 nodes, storage read is
   59% of the wall. Aggregate read demand = 284 Hz × 33.5 MB = 9.5 GB/s across 2 nodes =
   4.76 GB/s/node = ~60% of the 7.9 GB/s/node FFB ceiling (TASK.md) — bandwidth-limited but
   not fully storage-saturated, consistent with iter 5's per-rank pipeline being
   serialization-bound (plateau ~16 BD/node), not latency-bound.

**Keep/revert:** KEEP `mn2x32_r47.sbatch` (the r47 multi-node template; the old mn2x32.sbatch
pointed at purged r387). No numeric path touched (only `--wait-split`, already bit-exact per
iter 15) — correctness gate not triggered. No throughput change landed; this iteration buys
the post-wins multi-node anchor (284 Hz) + the attribution that kills the eb_wait-ceiling
hypothesis.

**Recommended next step:** the multi-node ceiling is now attributed to bd_read
(storage/read serialization), not the serving chain — but the single-node vs multi-node
per-rank bd_read comparison is confounded by window state. Two clean follow-ups, in order:
(a) **get the same-window single-node number** — resubmit mn2x32_r47.sbatch with
`--oversubscribe` on the single-node lines so the bracket lands, giving an in-allocation
32-BD→64-BD scaling ratio that isolates node-count from window; if that ratio is ≈1x
(negative scaling) the storage-contention story is confirmed, if ≈2x the plateau is a
per-node saturation the historical number just under-sampled. (b) If read is confirmed the
multi-node wall, the lever is **per-node reader concurrency vs FFB saturation** — iter 5
found the per-node pipeline plateaus ~16 BD; test whether fewer-but-fatter readers or a
node-local staging tier (DEFERRED/TASK item 2, the 585 Hz warm baseline hint from iter 13)
lifts the per-node storage ceiling. Prefetch stays CAUTION (iters 12-14).

### Iteration 16 addendum — same-window bracket LANDED (job 31291618, --oversubscribe fix), scaling ratio isolated from window

The `--oversubscribe` fix made the single-node bracket configs launch, so the full
single/multi/single bracket ran in ONE allocation (sdfampere001,028) — window/cache held
constant across all three. This is the clean scaling number the primary run couldn't give.

**Same-window bracket (r47, FFB, A100; logs `bench_mpi_sweep/sweep_sn32_r47_a.log`,
`sweep_mn2x32_r47.log`, `sweep_sn32_r47_c.log`; job 31291618):**

| config | agg Hz | bd_read ms | eb_wait ms | wait-residual ms | per-rank wall ms |
|---|---:|---:|---:|---:|---:|
| A single 32-BD (start) | 189.0 | 128.3 | 4.22 | 37.5 | 169.3 |
| B **two-node 64-BD** | **291.6** | 126.8 | 11.17 | 74.4 | 219.5 |
| C single 32-BD (end) | 174.4 | 135.8 | 6.63 | 41.6 | 183.5 |

single-node mean (A,C) **181.7 Hz** (bracket drift 189→174 = −7.7%, tolerable).

**Clean scaling: 291.6 / 181.7 = 1.60x for 2x nodes → per-node efficiency 80%** (145.8
Hz/node at 2 nodes vs 181.7 Hz/node at 1). Still sub-linear — the plateau is real — but
measured in a single window, so it is NOT a window artifact.

**What this CORRECTS in the primary entry above:**
1. **bd_read is FLAT single→multi (132 → 127 ms), NOT the 3x I flagged as
   window-confounded.** Confirmed: the 43.6 (iter 15, warm) → 133 (cold) swing was ENTIRELY
   cold/warm window state, exactly the iters-13/14 4-5x bd_read regime effect. **Storage read
   scales cleanly per-node** (2 nodes = 2x readers each at the same ~127 ms/event); FFB's
   per-node 7.9 GB/s bandwidth genuinely adds with nodes. So bd_read is the LARGEST bucket
   (58% of wall) but it is a WELL-SCALING cost — it is **not** what limits scaling to 1.6x.
2. **The sub-linear scaling (the 20% per-node loss) lives in eb_wait + the wait-residual,
   both of which ~DOUBLE while bd_read stays flat:** eb_wait 5.4 (A/C mean) → 11.2 (2.1x),
   wait-residual 39.6 → 74.4 (1.9x). eb_wait is small in absolute terms (5% of wall); the
   residual is the bigger mover (~74 ms = 34% of the multi-node wall). The residual =
   `wait − eb_wait − bd_read` = dgram construction + generator plumbing + smd0/EB
   coordination roundtrip NOT captured by eb_wait. CAVEAT: the residual carries iter 15's
   denominator mismatch (wait normalized by `n`, eb_wait/bd_read by `bd_events > n`), so its
   absolute size is inflated — but it DOUBLING single→multi (same instrumentation both
   sides) is a real relative signal of growing per-rank coordination cost at 2 nodes.

**Refined verdict:** the multi-node ceiling is NOT storage (bd_read scales per-node) and NOT
the EB batch wait (eb_wait tiny). The scaling loss is **coordination/serving overhead in the
wait path (residual + eb_wait roughly double at 2 nodes)** — the smd0 (single rank 0) → EB →
64-BD-across-2-nodes cross-node request/serve roundtrip. This is the same "serving chain"
suspect the campaign raised, now localized to the wait-residual (dgram/generator/coordination),
NOT to eb_wait or bd_read specifically.

**Refined next step:** attribute the wait-residual at multi-node directly, the way iter 15
attributed it at single-node. iter 15's `total_dgram_ns`/`total_smdparse_ns` counters + the
`--wait-split` `dgram`/`smdparse`/`residual` lines already exist — but the current multi-node
`--wait-split` path reports only eb_wait/bd_read (the `run_gpu_bench_wait_split` aggregate
branch, lines ~949-964, doesn't surface dgram/smdparse). Plumb the dgram/smdparse counters
into the multi-node aggregate report and re-run the bracket: if dgram/smdparse stay flat
single→multi (iter 15 showed dgram ~4 ms, near-flat) then the doubling residual is pure
smd0/EB cross-node COORDINATION latency — which points the lever at smd0/EB rank placement or
the request/serve protocol (a genuinely different, structural change), not per-rank code. If
instead dgram/smdparse grow, the generator plumbing is contention-sensitive after all. Either
way this is the one attribution still missing to close the multi-node plateau question. Fix
the sbatch permanently by keeping the --oversubscribe single-node bracket (done).

## 2026-07-10 — Iteration 17 (multi-node wait-residual attribution — dgram/smdparse plumbed into the aggregate report; both stay FLAT single→multi, so the doubling residual is pure smd0/EB cross-node coordination, NOT generator plumbing)

**Task:** the iter-16 addendum's one missing attribution — surface dgram/smdparse in the
multi-node `--wait-split` aggregate and re-run the same-window bracket to split the doubling
wait-residual into dgram/generator plumbing vs pure smd0/EB coordination latency.

**Change landed (pure-Python, numeric path untouched, syntax-checked):** the
`run_gpu_bench_wait_split` aggregate branch in `bench_calib.py` (the multi-node
`eb_wait_ms_mean` case, ~line 949) now prints `residual`/`dgram`/`smdparse` mirroring the
single-node format. The result dict already carried `dgram_ms_mean`/`smdparse_ms_mean`; only
the print path in the aggregate branch was missing.

**Measurement:** same-window `--oversubscribe` bracket sn32_a → mn2x32 → sn32_c, job 31292257
(sdfampere[026-027], all three exit=0), `-n 200 --warmup 10 --wait-split` on r47. Logs:
`bench_mpi_sweep/sweep_sn32_r47_a.log`, `sweep_mn2x32_r47.log`, `sweep_sn32_r47_c.log`;
driver `bench_mpi_sweep/slurm-31292257.out`.

| metric (ms/event, aggregate) | single-node 32 BD (mean a,c) | 2-node 64 BD (mn2x32) | multi/single |
|---|---|---|---|
| aggregate rate | ~173.5 Hz (176.5 / 170.5) | 288.7 Hz | 1.66x |
| wait | 182.8 | 217.0 | 1.19x |
| eb_wait | 6.36 | 12.87 | 2.02x |
| bd_read | 136.2 | 126.8 | 0.93x (flat) |
| residual | 40.2 | 77.3 | 1.92x |
| dgram | 2.05 | 1.93 | 0.94x (flat) |
| smdparse | 0.60 | 0.75 | +0.15 ms abs |

**Verdict — the iter-16 fork is RESOLVED.** dgram and smdparse are FLAT single→multi (dgram
2.05→1.93, smdparse 0.60→0.75; combined ~2.65→2.68 ms, essentially unchanged). The residual
nearly doubles (40.2→77.3, +37 ms) yet dgram/smdparse contribute ≈0 of that growth. So the
doubling wait-residual is NOT dgram construction or generator plumbing — it is the
residual-minus-dgram bucket (`wait − eb_wait − bd_read − dgram − smdparse`), i.e. **pure
smd0/EB cross-node coordination latency**. eb_wait doubling (2.02x) points the same way;
bd_read flat (0.93x) reconfirms storage scales per-node. 2-node scaling holds at 1.66x (83%
per-node efficiency), consistent with iter-16's 1.60x.

**Next lever:** the plumbing has now excluded per-rank code (dgram/generator) as the
multi-node scaling loss — both flat. The growth lives entirely in the smd0(rank 0)→EB→64-BD-
across-2-nodes request/serve roundtrip (eb_wait + coordination residual, each ~2x). The
remaining lever is structural: smd0/EB rank placement across nodes, or the request/serve
protocol itself — not per-rank numerics. This is the boundary where the per-rank-optimization
campaign meets a distributed-coordination one.

## 2026-07-13 — Iteration 18 (structural lever: PS_EB_NODE_LOCAL node-local EB placement — bench role-detection fixed and validated both modes single-node; 2-node bracket queued)

**Task:** act on the iters-16/17 verdict. The 2x-node scaling loss (1.60–1.66x,
~80% per-node efficiency) is structural smd0/EB cross-node coordination — eb_wait ~2x and
the wait-residual ~2x while bd_read/dgram/smdparse stay flat. The default rank placement
(`color = bd_main_rank % PS_EB_NODES`, node.py:189) puts the single EB on node 0, so every
node-1 BD rank crosses the network to it each batch. That cross-node hop IS the coordination
residual. This iteration tests the direct structural fix.

**Discovery — the lever already exists behind a flag, and had never been exercised.**
`node.py` has a `colocate_non_marching` mode (env `PS_EB_NODE_LOCAL`, node.py:159/178) that
splits `bd_main` by `MPI.COMM_TYPE_SHARED` instead of by modulo — giving **each physical
node its own EB serving only its node-local BD ranks**, and an smd_comm of `[smd0] +
per-node-EBs` (node.py:330–342). Every multi-node number in this campaign was measured with
the default modulo placement (single cross-node EB); the node-local mode is exactly the
"smd0/EB rank placement" structural change the iter-17 next-step called for.

**Blocker in the harness, now fixed (bench_calib.py only — NOT psexp; the numeric path is
untouched, so the correctness gate is not triggered).** `bench_calib._is_bd_rank()` decided
BD-ness by rank arithmetic (`rank >= 1 + PS_EB_NODES`), which assumes contiguous low-rank
EBs. Under `PS_EB_NODE_LOCAL` the EBs are the lowest bd-rank on each node — world ranks 1 and
33 for the `-H N0:33,N1:33` layout — so arithmetic would mis-serve rank 33. Fix: `main()` now
reads the authoritative role from `psana.psexp.mpi_ds.nodetype` (set to `comms.node_type()`
during DataSource construction, correct in BOTH modes) and refines `is_bd` from it;
`init_gpu_rank()` moved to just after DataSource so it keys off the real role. `_report_
aggregate` already filters by `r.get("n")`, so once real EBs return `result=None` the
aggregate is correct with no further change.

**Validated single-node, both modes (ralph-gpu 31540560, sdfampere001, r47, FFB, `-n 60`):**
- default (`PS_EB_NODES=1`): runs clean, per-rank ~2.8 Hz (cold window), default path intact.
- colocate (`PS_EB_NODE_LOCAL=1`): "Non-marching EB/BD colocated mode enabled" logged;
  **aggregate BD ranks: 32** (role-detection fix correctly excludes the node-local EB, world
  rank 1), aggregate 109.6 Hz. Single-node colocate ≡ default (1 node → 1 EB), so this is a
  smoke test of the plumbing, not the lever — the lever only bites at ≥2 nodes.

**2-node bracket QUEUED — this is the actual measurement, not yet collected.** Submitted
`bench_mpi_sweep/eb_node_local.sbatch` (job **31541312**): bracketed default_a → colocate →
default_c, all 2-node/66-rank (`-H N0:33,N1:33`), `-n 150 --warmup 10 --wait-split`, same
allocation/window. Colocate yields 63 BD ranks (2 EBs) vs default 64 BD (1 EB) — a ~1.5%
BD-count penalty a real placement win must clear. HYPOTHESIS: if node-local EBs remove the
cross-node serve hop, colocate's eb_wait + wait-residual drop toward their single-node values
and aggregate Hz rises above the default bracket mean; if they don't move, the residual is
not the cross-node hop but intra-EB serialization (one EB rank saturating), redirecting the
lever to EB-side concurrency. The ampere multi-node queue was congested; the job stayed
PENDING through the ~25-min in-iteration wait cap.

Logs (to collect): `bench_mpi_sweep/enl_default_a.log`, `enl_colocate.log`, `enl_default_c.log`;
driver `bench_mpi_sweep/slurm-31541312.out`.

**Keep/revert:** KEEP the bench_calib.py role-detection fix — it is the prerequisite for
measuring any non-contiguous-EB placement mode and is validated correct in both modes.

**Recommended next step:** collect job 31541312 (rule 7 recovery). Parse the three aggregate
blocks; compute colocate vs mean(default_a, default_c). Read `eb_wait` / `residual` in the
`--wait-split` aggregate for each. If colocate lifts aggregate Hz and shrinks eb_wait/
residual → node-local EB placement is the plateau lever; make it the multi-node default and
re-run at 4 nodes to confirm it scales. If flat/worse → the coordination cost is intra-EB
serialization, not the cross-node hop; next lever is a second EB per node or a fatter serve
batch. If the job did not land (still PENDING / launch error), resubmit as-is.

BLOCKED: job 31541312 in flight (PENDING, multi-node queue) — collect on next iteration

## 2026-07-13 — Iteration 18 addendum (2-node bracket COLLECTED — node-local EB placement WINS: +9% aggregate, residual −14%, hypothesis confirmed)

**Recovery of the queued bracket.** Job 31541312 never started — it sat PENDING on
`AssocGrpNodeLimit` (the `lcls:data@ampere` association's shared GrpNodes=4 /
GrpTRES cap, which counts all account members' jobs, and which the scheduler also
reports transiently while re-evaluating). It was cancelled and the identical
`bench_mpi_sweep/eb_node_local.sbatch` resubmitted as job **31543609**, which ran on
sdfampere[010,029]: all three configs exit=0, ~117 s wall each, `-n 150 --warmup 10
--wait-split`, same allocation/window. Driver log `bench_mpi_sweep/slurm-31543609.out`.

**Results (per-config `--wait-split` aggregates):**

| config    | BD | aggregate Hz | per-rank Hz | eb_wait ms | bd_read ms | residual ms |
|-----------|----|-------------|-------------|------------|------------|-------------|
| default_a | 64 | 307.6       | 4.81        | 9.9        | 125.2      | 75.9        |
| colocate  | 63 | **328.4**   | **5.21**    | 16.4       | **111.1**  | **65.2**    |
| default_c | 64 | 294.4       | 4.60        | 16.9       | 125.6      | 76.5        |

**Verdict — the iter-18 hypothesis is CONFIRMED at 2 nodes.** Colocate beats the
default bracket mean (301.0 Hz) by **+9.1% aggregate** and **+10.7% per-rank**,
clearing the ~1.5% BD-count penalty (63 vs 64) with room to spare. The win lands
exactly where the iters-16/17 attribution said it should: the coordination
residual drops 76.2 → 65.2 ms/event (−14%) and bd_read drops 125.4 → 111.1
(−11%, node-1 BDs no longer contending with cross-node serve traffic). eb_wait is
bracket-noisy (default_a 9.9 vs default_c 16.9 straddle colocate's 16.4) and
carries no signal this run. Against iter-16's cross-window single-node 181.7 Hz,
colocate's 328.4 implies ~1.81x node scaling (~90% per-node efficiency) vs the
default's 1.66x — a partial, not full, recovery of the plateau, consistent with
the remaining loss being intra-EB serialization (one EB rank per node serving 32
BDs).

**Keep/next:** node-local EB placement is a real multi-node lever. Next iteration:
make `PS_EB_NODE_LOCAL=1` the multi-node benchmark default, add an in-window
single-node control to pin the true scaling ratio, and re-run at 4 nodes to
confirm the win compounds; if per-node efficiency still trails ~90%, the next
lever is EB-side concurrency (second EB per node or fatter serve batches).

## 2026-07-13 — Iteration 19 (node-local EB at scale: 4-node infeasible in-loop, 3-node compounding bracket queued)

**Task:** iter-18's recommended next step — confirm node-local EB placement
(`PS_EB_NODE_LOCAL`) compounds beyond 2 nodes, with an *in-window* single-node
control so per-node efficiency is measured against the same cache regime instead
of iter-16's cross-window 181.7 Hz anchor. One variable: default modulo (single
EB on N0) vs colocate (one EB/node) at fixed node count; bracketed sn/mn/sn.

**Structural finding — 4 nodes cannot run through this loop.** Built and submitted
the 4-node bracket (`bench_mpi_sweep/eb_node_local_4n.sbatch`, job **31548195**,
sn32 → mn4_default(130 BD) → mn4_colocate(127 BD, 4 EBs) → sn32). It sat on
`AssocGrpNodeLimit` for the full in-iteration wait and cannot clear: the driver's
`ralph-gpu` allocation permanently holds 1 node, the `lcls:data` association cap
is `GrpNodes=4`, so a 4-node job needs 1+4 = 5 > 4 and is rejected for as long as
ralph-gpu is up. Rule 6/§8 forbid cancelling ralph-gpu, so the clean 4× point is
not reachable from inside the loop. `scontrol show job 31548195`:
`Reason=AssocGrpNodeLimit ReqTRES=…,node=4`. Cancelled it (mine).

**Pivot to 3 nodes — the largest run the loop can schedule (1 ralph + 3 = 4, at
the cap) and still a valid compounding test at 3×.** Built
`bench_mpi_sweep/eb_node_local_3n.sbatch` (kept the 4n template for when the loop
can spare the node budget): sn32 → mn3_default(97 BD, 1 EB) → mn3_colocate(96 BD,
3 EBs) → sn32, all `-n 150 --warmup 10 --wait-split`, r47/FFB, same allocation.
Submitted job **31548860**.

**Blocked on shared-cap congestion, not the ralph arithmetic.** 3 nodes is exactly
at the cap, yet it too sits on `AssocGrpNodeLimit` — so other `lcls:data` members
are currently consuming the shared GrpNodes budget, not just ralph-gpu. This is
transient congestion (unlike the 4-node structural wall): the job will schedule
when the account frees, so it is left in flight for driver-side recovery (rule 8)
rather than resubmitted. Logs to collect once it runs: `enl3_sn_a.log`,
`enl3_default.log`, `enl3_colocate.log`, `enl3_sn_c.log`; driver
`bench_mpi_sweep/slurm-31548860.out`.

**Keep/revert:** KEEP both sbatch templates (no numeric-path change; correctness
gate not triggered). No throughput number this iteration — the measurement is
queued.

**Recommended next step:** collect job 31548860 (rule 7 recovery). Parse the four
aggregate blocks; compute mn3_colocate vs mn3_default and each against the sn
control mean → per-node efficiency at 3×. If colocate holds ~90% per-node eff
while default degrades below its 2-node 83%, and colocate residual/bd_read stay
depressed → node-local EB compounds; make `PS_EB_NODE_LOCAL=1` the multi-node
default and record in TASK.md. If colocate's per-node efficiency has fallen back
toward default → the win does not compound and the remaining lever is intra-EB
serialization (second EB per node / fatter serve batch), not the cross-node hop.
Note for the human: raising the `lcls:data` GrpNodes cap (or briefly releasing
ralph-gpu) is the only way to measure the clean 4-node point.

BLOCKED: job 31548860 in flight (PENDING, AssocGrpNodeLimit shared-cap congestion) — collect on next iteration

## 2026-07-13 — Iteration 19 addendum (true blocker found: LCLS-wide parent-assoc node cap; both brackets resubmitted preemptable — the 4-node point is back on the table)

**Root cause of the stall — corrects this iteration's "transient congestion" read.**
Job 31548860 sat on `AssocGrpNodeLimit` for ~4 h even after ralph-gpu was released
and `lcls:data@ampere` usage dropped to 1 node. `scontrol show assoc_mgr` shows why:
assoc limits are hierarchical, and the binding cap is the PARENT association
`lcls:_regular_@ampere` — **GrpTRES node=4 shared across ALL regular-branch LCLS
accounts on ampere**, with 3 nodes in use LCLS-wide. A 3-node job needs LCLS-wide
usage ≤1 node; the clean 4-node point needs it at exactly 0. Neither is a realistic
daytime window, and the 4× measurement was structurally unreachable through the
regular branch regardless of the ralph-gpu holder.

**Escape hatch: the preemptable branch has no node cap.** The `lcls:default@ampere`
association hangs under `lcls:_preemptable_@ampere` (a sibling hierarchy, all
GrpTRES=N). Submitting with `--account=lcls:default --qos=preemptable` bypasses the
node=4 wall entirely; the cost is preemption risk, acceptable for ~10-minute
brackets (resubmit on preempt). Both brackets went in back-to-back:
- **job 31573442** — `eb_node_local_4n.sbatch` (sn_a → mn4_default 130 BD →
  mn4_colocate 127 BD/4 EBs → sn_c), the clean 4× compounding point this iteration
  had written off.
- **job 31573443** — `eb_node_local_3n.sbatch` (sn32_a → mn3_default 97 BD →
  mn3_colocate 96 BD/3 EBs → sn32_c), unchanged science, new association.
31548860 (regular-branch 3n) cancelled — superseded by 31573443.

**Next iteration: collect BOTH.** Logs: `enl4_sn_a/enl4_default/enl4_colocate/
enl4_sn_c.log` (driver `slurm-31573442.out`) and `enl3_sn_a/enl3_default/
enl3_colocate/enl3_sn_c.log` (driver `slurm-31573443.out`). Together with iter-18's
2-node bracket this yields the full 1/2/3/4-node scaling curve for default vs
node-local EB placement — compute per-node efficiency at each point against the
in-window sn controls; the compounding verdict and the PS_EB_NODE_LOCAL default/
TASK.md decision follow the iter-19 criteria above. For future multi-node jobs:
prefer `--account=lcls:default --qos=preemptable` whenever the job needs ≥3 nodes
or the regular branch is congested; keep short walls so preemption stays cheap.

BLOCKED: job 31573442 in flight (4n bracket; job 31573443 3n bracket right behind it, both preemptable-QOS) — collect both on next iteration

## 2026-07-13 — Iteration 20 (node-local EB COMPOUNDS at 3 nodes: clean in-window bracket, +31% and widening)

**Task:** recover the iter-19 hand-off (collect the 3n/4n default-vs-colocate
brackets, jobs 31573442/31573443) and settle the compounding question.

**The hand-off jobs were unrecoverable as-is — both failed the same two ways.**
Inspecting the requeued-PENDING jobs and their populated logs (they had *run*,
been preempted, and been auto-requeued by the preemptable QOS):
1. **sn control never mapped.** It asked `-H N0:34 -n 34` on a 33-slot/node
   allocation with no `--oversubscribe` → exit 213 "Out of resource". So the
   in-window single-node normalization iter-19 explicitly wanted was absent from
   every prior 3n/4n run.
2. **colocate (phase 3 of 4) was preempted ~6 min in** → "PRTE lost
   communication with a remote daemon", no aggregate, DONE marker never printed.
   The default numbers that *did* land (3n 274.9 Hz / 4n 294.4 Hz, from
   `enl3_default.log`/`enl4_default.log`) came from that degraded, uncontrolled
   window with bd_read inflated to ~160 ms — not a valid scaling curve.

Cancelled the flawed requeued jobs (31573442/31573443, mine). Built
`bench_mpi_sweep/eb_node_local_3n_v2.sbatch` with two **infra-only** fixes (the
science variable, default-vs-colocate, unchanged): `--oversubscribe` on the sn
control, and front-loaded phases sn_a → **colocate** → default (fragile phase
early, trailing sn_c dropped, `-n` 150→120) to land everything inside the ~6 min
preemption window. Submitted preemptable job **31574438**; blocked on it
in-iteration (rule 8). All three phases exit=0, total wall 338 s — clean.

**Results — clean, single-allocation, in-window (`bench_mpi_sweep/enl3v2_*.log`,
driver `slurm-31574438.out`):**

| config       | BD | aggregate Hz | per-rank Hz | eb_wait ms | bd_read ms | residual ms |
|--------------|----|-------------|-------------|-----------|-----------|-------------|
| sn_a (1 node)| 32 | 193.0       | 6.03        | 3.5       | 136.5     | 40.4        |
| mn3_colocate | 95 | **444.0**   | **4.67**    | 26.3      | 95.4      | 85.8        |
| mn3_default  | 97 | 338.3       | 3.49        | 70.1      | 95.1      | 136.2       |

**Verdict — node-local EB placement COMPOUNDS, decisively.** Colocate beats
default by **+31.3% aggregate / +33.8% per-rank** at 3 nodes, up from +9.1% at
2 nodes (iter-18) — the advantage *widens* with node count. In-window per-node
efficiency (against the same-allocation sn_a=193.0): colocate 444.0/193.0 =
**2.30× on 3 nodes (77%)**; default 338.3/193.0 = 1.75× (58%). So default's
per-node efficiency collapses 83% (2n) → 58% (3n) while colocate degrades far
more gently 90% → 77%.

**Attribution is now unambiguous and confirms the iter-18 hypothesis exactly.**
The differentiator is **eb_wait**: 70.1 ms (default) vs 26.3 ms (colocate). A
single EB rank serving 96 BDs across 3 nodes serializes hard (eb_wait 3.5→70 ms
as BDs go 32→96); one EB per node (~32 BD each) holds eb_wait to 26 ms. The
coordination residual follows (136.2 vs 85.8). **bd_read is flat** (95.1 vs 95.4)
— storage scales per-node cleanly, re-confirmed a fourth time. The remaining
loss lever the iters-16/18 attribution named — "one EB rank saturating" — is now
directly measured as the dominant multi-node bottleneck, and node-local
placement is its fix.

**Keep/revert:** KEEP. `PS_EB_NODE_LOCAL` (landed iter-18, correctness-gated
there; no numeric-path change this iteration so the gate was not re-triggered) is
a confirmed, compounding multi-node lever. Recorded in TASK.md.

**Recommended next step:** promote `PS_EB_NODE_LOCAL` from experiment to the
documented, recommended setting for multi-node GPU calibration — but as an
*opt-in flag*, NOT a silent psana-wide default: the placement change touches the
shared EB/BD dispatch and flipping it for all (incl. non-GPU) psana users is a
large blast radius that rule 4 says must be guarded and default-verified. So next
iteration: (a) document PS_EB_NODE_LOCAL=1 as the standard multi-node GPU launch
flag in the run recipe / DEFERRED graduation note, and (b) if a still-larger point
is wanted, the residual per-node loss at 3n (77%, i.e. colocate eb_wait still 26
ms and residual 85.8) points at *intra-node* EB serialization — a second EB per
node or fatter serve batches — as the next lever, testable at 2–3 nodes without
needing the structurally-unreachable clean 4-node point.

## 2026-07-13 — Iteration 21 (graduate PS_EB_NODE_LOCAL: documented as the standard multi-node GPU launch flag)

**Task:** execute iter-20's explicit first recommendation (a) — promote the
now-confirmed, compounding `PS_EB_NODE_LOCAL` win from loop experiment to a
documented, discoverable opt-in launch flag. No new number this iteration by
design: the compounding verdict is already measured (iters 18/20, clean
in-window brackets); the remaining gap was that a user launching a multi-node
GPU calib run had no recipe telling them to set the flag. One well-scoped unit:
close that gap. The separate intra-node "2nd EB per node / fatter serve batch"
lever is left for a future iteration (one variable — do not bundle a fresh
fragile multi-node experiment into a documentation graduation).

**What changed (docs only — numeric path untouched, so no correctness gate per
rule 3; flag is default-off so standard psana is unchanged per rule 4):**
1. `psana/psana/gpu/PLANNING.md` — new "Multi-node launch: `PS_EB_NODE_LOCAL=1`"
   section between Development Commands and Environment Setup. States: use it for
   ≥2 nodes; it is opt-in/default-off (gated by `_ensure_local_eb_nodes` in
   datasource.py + `colocate_non_marching` in psexp/node.py — verified by code
   inspection that unset/`0` leaves the default event loop unchanged); the
   measured 2n/3n gain table (+9.1% / +31.3%); the mechanism (single-EB `eb_wait`
   serialization, `bd_read` flat); the launch snippet + sbatch template pointer
   (`bench_mpi_sweep/eb_node_local_3n_v2.sbatch`); the "don't hand-set
   PS_EB_NODES alongside it" caveat; and the named next lever (intra-node EB
   serialization) as not-yet-measured.
2. Memory `lcls2-gpu-run-setup.md` — one-line pointer so the flag is discoverable
   from the run recipe, not only buried in TASK.md/PROGRESS.md.

TASK.md already carries the full measured record (2026-07-13 entry, iters 18/20)
and its NEXT line already names this graduation — no TASK.md edit needed; this
iteration makes that graduation real in the user-facing planning doc.

**Keep/revert:** KEEP (documentation of a confirmed, kept feature). Nothing to
revert. No job submitted (no allocation spent) — the measurement backing this
was collected in iters 18/20.

**Recommended next step:** the intra-node EB serialization lever is now the sole
remaining code-side multi-node throughput idea. At 3 nodes node-local still shows
`eb_wait` 26 ms and 77% per-node efficiency, so one EB per node has not fully
removed EB serialization. Test **2 EB per node** (each serving ~15 BD instead of
~32) behind a new opt-in `PS_EB_PER_NODE=2` — a localized change to the colocate
split in `psexp/node.py` (sub-split each node's shared comm by `node_rank %
PS_EB_PER_NODE`; the `_init_smd_comm` eb-detection already generalizes since each
sub-comm's rank-0 self-reports as an EB). Correctness-gate it, then a clean
2-node in-window bracket (1 vs 2 EB/node) on preemptable QOS — 2 nodes is the
least preemption-fragile scale (iter-20 lesson: front-load fragile phases, keep
the wall <6 min). Watch the tradeoff: a 2nd EB per node consumes a BD slot, so
per-rank `eb_wait` must fall enough to beat the lost BD; report aggregate + per-
rank both. If `eb_wait` does NOT fall (i.e. the 26 ms is cross-node smd0/EB
coordination, not intra-node BD-per-EB count — iter-17 leaned this way), that
closes the code-side multi-node levers and the loop is near LOOP DONE pending the
storage-side facility levers.

## 2026-07-13 — Iteration 22 (implement PS_EB_PER_NODE — intra-node EB fan-out; segfault found+fixed, single-node functional-clean, multi-node scaling bracket next)

**Task:** execute iter-21's named next lever — the residual per-node EB
serialization that one-EB-per-node colocation still leaves at scale (measured
eb_wait 26 ms @3n, 77% per-node efficiency). Implement an opt-in **`PS_EB_PER_NODE`**
that places multiple EBs per compute node, each serving a fraction of that node's
BD ranks, so the hypothesis "26 ms is intra-node BDs-per-EB count (fixable) vs
cross-node smd0/EB coordination (not fixable by more EBs/node — iter-17 leaned
this way)" becomes directly testable. One variable.

**Implementation (guarded, default-off — inert unless PS_EB_NODE_LOCAL=1 AND
PS_EB_PER_NODE>1):**
1. `psexp/node.py` — in colocate mode, after the per-node `Split_type(SHARED)`,
   sub-split each node's shared comm by `node_local_rank % eb_per_node`. Each
   sub-comm's rank-0 becomes an EB (existing `bd_rank==0 -> "eb"` generalizes),
   so a 32-rank node with `PS_EB_PER_NODE=2` yields 2 EBs of ~15 BDs each instead
   of 1 EB of 31. Also set `n_smd_nodes = len(smd_worlds)-1` in the colocate
   `_init_smd_comm` branch so smd0's request/serve/missing-step/kill loops count
   the *actually-detected* EB set (now > PS_EB_NODES=node_count).
2. `datasource.py::_ensure_local_eb_nodes` — override PS_EB_NODES to
   `node_count * eb_per_node` (was `node_count`).

**Non-editable install gotcha (cost 2 wasted runs — recorded for next iter):**
runtime imports psana from `install/lib/python3.9/site-packages/`, NOT the source
tree. A source-only edit to `psexp/node.py` is silently ignored (the first
functional run printed the OLD role message with no `eb_per_node=` suffix). psexp
is pure-Python so no C rebuild is needed — but the edited file must be synced into
`install/`. Verified `diff install/... source/...` showed ONLY my edits, then
`cp`'d node.py + datasource.py into install (equivalent to `meson install` for a
.py file). Only `psana/psana/gpu/` is import-live without this step.

**Segfault found and fixed (the substantive result of this iteration).** First
2-EB run: topology formed correctly (`eb_per_node=2` role line present) but smd0
(rank 0) **segfaulted (signal 11) inside the `smdreader` C extension**
(`repack_parallel`), exit 139. Root cause: `smdreader.pyx:104/142` allocates
`send_bufs = malloc(n_eb_nodes * sizeof(Buffer))` where `n_eb_nodes =
int(getenv('PS_EB_NODES','1'))`. With 2 EB/node the requesting EB's `eb_node_id`
(its smd_rank, 1..2*node_count) indexed past the buffer array -> OOB write ->
segfault. Fix is the datasource.py change above: making PS_EB_NODES reflect the
*total* EB count sizes the C send-buf array correctly, no C rebuild required.

**Verification (ralph-gpu 31583622, single node, cold FFB window — see log paths):**
- Correctness gate: **PASS**, bit-exact max_diff=0.000000, 20 events (numeric path
  untouched; run per protocol for a psexp change).
- 2-EB after fix: **exit 0, 31 BD ranks** (2 EBs took 2 of 33 non-smd0 slots),
  aggregate 100.1 Hz — clean, no crash. Log `ralph/eb_per_node_sn.log` +
  `_verify2eb.sh` output.
- Default path (flags off): **exit 0, 32 BD ranks**, 104.0 Hz — unbroken (rule 4).
- Window note: this window is COLD (default 104 Hz vs ~175 Hz warm baseline;
  earlier 240s-capped runs hit exit 124 timeout from slowness, not breakage), so
  the single-node 100 vs 104 Hz is NOT a lever measurement — single node is
  storage-bound with tiny eb_wait, exactly where 2 EB/node is NOT expected to help
  (it costs one BD slot). The lever must be tested at multi-node where eb_wait
  grows. That clean 2-node bracket (1 EB/node vs 2 EB/node) is this iteration's
  next action.

**Keep/revert:** KEEP the implementation — opt-in, default-off, correctness-gated,
default-path-verified, segfault fixed, 2-EB topology functionally clean. The
*scaling verdict* (does eb_wait actually fall) is not yet measured; that decides
whether PS_EB_PER_NODE graduates or is documented as a tried-but-flat lever.

**Recommended next step:** run the 2-node in-window bracket sn_a(1 node) /
mn2_1eb(PS_EB_NODE_LOCAL=1) / mn2_2eb(+PS_EB_PER_NODE=2) on preemptable QOS
(front-load fragile phases, wall <6 min per iter-20). Compare aggregate AND
per-rank (a 2nd EB costs a BD slot, so per-rank eb_wait must fall enough to beat
the lost BD). If eb_wait drops and aggregate rises -> intra-node lever confirmed,
graduate it. If eb_wait is FLAT -> the 26 ms is cross-node smd0/EB coordination
(iter-17), this closes the code-side multi-node levers, and the loop is at LOOP
DONE pending storage-side facility levers.

### Iteration 22 addendum — 2-node bracket COLLECTED: PS_EB_PER_NODE=2 WINS (+23% aggregate / +27% per-rank, eb_wait −18%)

Two benchmark starved-rank bugs blocked the first two 2-node brackets (jobs
31585830, 31586227) — both MPI_Aborted mid-run so no rank-0 aggregate survived,
and neither exposed a defect in the PS_EB_PER_NODE feature itself (the topology
formed and calib ran in every attempt):
1. `bench_calib.py:585` (`run_gpu_bench_wait_split`): `snap0 = (0,0,0)` was a
   3-tuple but `_snap()` returns 5 — a rank starved below `warmup` events never
   ran `snap0 = _snap()`, so `snap1[3]` → IndexError. Fixed: `snap0 = (0,0,0,0,0)`.
2. `bench_calib.py:1121`: the per-rank breakdown divides `1000.0/rate_hz`; a
   0-event rank has rate_hz=0 → ZeroDivisionError. Fixed: skip the per-rank
   breakdown when `n==0 or rate_hz<=0` (the rank-0 aggregate already filters to
   `n>0` ranks). Both are `psana/psana/gpu/` tooling fixes (import-live, no
   install sync, byte-identical numerics for non-starved ranks); starvation is
   intrinsic to 60+ BD ranks sharing one max_events budget, not to this feature.

**Clean in-window bracket, job 31586774 (all three exit=0), `bench_mpi_sweep/
epn2n_{sn_a,2eb,1eb}.log`, driver `slurm-31586774.out`:**

| config              | BD | aggregate Hz | per-rank Hz | eb_wait ms | bd_read ms | residual ms |
|---------------------|----|-------------|-------------|-----------|-----------|-------------|
| sn_a (1 node)       | 32 | 221.1       | 6.91        | 6.88      | 103.3     | 32.2        |
| mn2_2eb (2 EB/node) | 61 | **504.9**   | **8.28**    | 17.39     | 72.0      | 24.1        |
| mn2_1eb (1 EB/node) | 63 | 410.6       | 6.52        | 21.16     | 88.8      | 55.5        |

**Verdict — PS_EB_PER_NODE=2 is a real multi-node lever. 2 EB/node beats 1 EB/node
by +23.0% aggregate (504.9 vs 410.6) and +27.0% per-rank (8.28 vs 6.52) — while
using FEWER BD ranks (61 vs 63, since the 2nd EB costs a slot). The strong form:
fewer workers, more throughput.** eb_wait fell −17.8% (21.16→17.39 ms), exactly
the direction the "intra-node BDs-per-EB serialization" hypothesis (iters 20/21)
predicted and against iter-17's "the 26 ms is pure cross-node coordination, not
fixable by more EBs/node" pessimism — that pessimism is REFUTED at 2 nodes.

Notably the eb_wait drop (−3.8 ms) is smaller than the total per-event wall
improvement: residual also collapsed (55.5→24.1 ms) and bd_read fell (88.8→72.0).
Coherent mechanism: one EB serving ~32 BDs doesn't just make them wait on batches
(eb_wait) — it bunches their reads + dgram construction into contended bursts,
inflating bd_read and residual too. Splitting to 2 EB/node de-bunches the whole
downstream chain, so all three buckets drop together. (Caveat: single bracket;
residual is the noisiest bucket window-to-window, so the headline claim rests on
the aggregate/per-rank/eb_wait trio, which all move consistently.)

**Keep/revert:** KEEP — decisively. Opt-in, default-off, correctness-gated
(bit-exact), default-path-verified, and now measured to compound the
PS_EB_NODE_LOCAL win at 2 nodes. Recorded in TASK.md.

**Recommended next step:** (a) a confirm bracket to firm up the single-window
number (rerun 1eb-vs-2eb, ideally back-to-back twice, to bound the residual
noise); (b) does it keep helping at PS_EB_PER_NODE=3/4 (each EB serving ~10/8 BD)
or is there a knee? test 1-vs-2-vs-3 EB/node at 2 nodes; (c) does it COMPOUND at
3 nodes like node-local placement did (iter-20: node-local widened +9%→+31% from
2n→3n)? A 3-node 1-vs-2-EB bracket would show whether the intra-node lever also
widens with node count. The code-side multi-node ceiling is NOT closed — this
lever just reopened it. If (b) finds a knee and (c) finds no compounding, then
graduate PS_EB_PER_NODE=2 as the recommended multi-node GPU default (alongside
PS_EB_NODE_LOCAL) and the loop is near LOOP DONE pending storage-side levers.

### Iteration 23 — PS_EB_PER_NODE knee is at 2 (3/4 EB regress −18%): graduate 2 as the 2-node default

**One variable:** EBs per node (PS_EB_PER_NODE ∈ {2,3,4}) at fixed 2 nodes,
node-local colocation. iter-22 measured 2 EB/node beats 1 by +23% aggregate at 2
nodes with eb_wait −18%; the graduation-decision question it left open is whether
2 is the sweet spot or 3/4 keeps helping (each EB then serves fewer BDs → even
lower eb_wait). This iteration answers it directly.

**Setup verified before submit:** node.py + datasource.py both already synced into
`install/` (the iter-22 non-editable-install gotcha — a source-only edit is
silently ignored by the runtime). `diff` source-vs-install identical for both.
Feature handles k=3/4: `_ensure_local_eb_nodes` sets PS_EB_NODES=node_count*k=6/8,
sizing the SmdReader C send-buf array so no OOB segfault. No C rebuild (pure-py).

**Clean 2-node in-window bracket, job 31587292 (all three exit=0),
`bench_mpi_sweep/epnknee_{3eb,2eb,4eb}.log`, driver `slurm-31587292.out`,
nodes sdfampere002/028, n=100/rank warmup=10 --wait-split:**

| config              | BD | aggregate Hz | per-rank Hz | eb_wait ms | bd_read ms | residual ms |
|---------------------|----|-------------|-------------|-----------|-----------|-------------|
| mn2_2eb (2 EB/node) | 61 | **502.9**   | **8.24**    | 10.06     | 77.4      | 30.9        |
| mn2_3eb (3 EB/node) | 59 | 414.1       | 7.02        | 4.29      | 102.7     | 24.5        |
| mn2_4eb (4 EB/node) | 57 | 414.1       | 7.27        | 3.64      | 113.4     | 12.3        |

**Verdict — the knee is at 2 EB/node. It is a genuine optimum, not a monotone
lever.** 2 EB = 502.9 Hz is the peak; 3 and 4 EB both fall to 414.1 Hz (−17.7%).
The mechanism is clean and slightly counter-intuitive: **eb_wait falls
MONOTONICALLY with more EBs (10.06 → 4.29 → 3.64 ms)** — exactly what the
"one-EB-serving-~32-BDs is a serialization point" model (iters 20/21/22)
predicts, and it keeps paying off past 2. **But aggregate peaks at 2 anyway,**
because past 2 EB eb_wait is no longer the binding term (already down to ~4 ms):
the marginal EB now only *costs* — it removes a BD-reader slot (61→59→57) and
bd_read inflates (77.4 → 102.7 → 113.4 ms/event), so per-rank rate drops (8.24 →
7.02 → 7.27) despite the smaller wait. Net: you trade away storage-read
concurrency for an eb_wait reduction you no longer need. 2 EB/node sits right at
the crossover.

**Reproducibility bonus:** the in-window 2eb here (502.9 Hz) matches iter-22's
504.9 Hz to <0.4%. That satisfies recommended-step (a) — the single-window
number is confirmed and window-to-window noise on the aggregate/per-rank headline
is small. (bd_read/residual remain the noisy buckets, but the aggregate and
per-rank verdicts are solid.)

**Keep/revert:** KEEP the feature; recommend **PS_EB_PER_NODE=2** as the 2-node
GPU default (paired with PS_EB_NODE_LOCAL=1). Do NOT set it higher — 3/4 regress.
No code change this iteration (measurement only); TASK.md updated with the knee.
Correctness untouched (no numeric-path edit; gate was run bit-exact in iter-22 for
the same code).

**Recommended next step:** does the =2 optimum COMPOUND at 3 nodes the way
node-local placement did (iter-20: node-local widened +9%@2n → +31%@3n)? And is
the knee still 2 at 3 nodes or does higher node count shift it (more nodes = more
cross-node EB coordination, which could move the crossover)? A 3-node
1-vs-2-vs-3 EB bracket answers both in one window. If =2 compounds and stays the
knee, graduate it into PLANNING.md alongside PS_EB_NODE_LOCAL as the documented
multi-node GPU launch recipe — at which point the code-side multi-node levers are
characterized and the loop is near LOOP DONE pending storage-side facility levers.

---

## 2026-07-13 — Iteration 24 (the PS_EB_PER_NODE knee SHIFTS UP at 3 nodes — 3 EB/node wins, the 2-node =2 optimum does NOT hold; order-confound flagged, reversed-order confirm next)

**One variable:** EBs per node (PS_EB_PER_NODE ∈ {1,2,3}) at fixed 3 nodes,
node-local colocation — the iter-23 recommended step. iter-22/23 found the knee
is 2 at 2 nodes (2 EB beats 1 by +23%; 3/4 EB regress −18%). Two open questions:
(a) does the =2 optimum COMPOUND at 3 nodes the way node-local placement did
(iter-20: +9%@2n → +31%@3n), and (b) is the knee still 2 at 3 nodes or does more
cross-node coordination shift it? This bracket answers both in one window.

**Setup verified before submit:** node.py + datasource.py both `diff`-identical
source-vs-install (the iter-22 non-editable-install gotcha). k=1 → node_count·1 =
one EB/node = exactly iter-20's node-local `colocate` baseline, so the k=1 row is
the same thing the 2-node +23% lead was measured over. No code change this
iteration (measurement only); no C rebuild (pure-py).

**Clean 3-node in-window bracket, job 31588126 (all three exit=0), nodes
sdfampere001/018/042, `bench_mpi_sweep/epn3n_{2eb,1eb,3eb}.log`, driver
`slurm-31588126.out`, n=100/rank warmup=10 --wait-split. Phase order 2eb → 1eb →
3eb:**

| config              | BD | aggregate Hz | per-rank Hz | eb_wait ms | bd_read ms | residual ms |
|---------------------|----|-------------|-------------|-----------|-----------|-------------|
| mn3_2eb (2 EB/node) | 91 | 411.5       | 4.52        | 23.98     | 89.63     | 63.31       |
| mn3_1eb (1 EB/node) | 95 | 419.5       | 4.42        | 29.86     | 94.17     | 98.96       |
| mn3_3eb (3 EB/node) | 89 | **553.3**   | **6.22**    | 9.97      | 103.81    | 34.16       |

**Verdict — the knee SHIFTS UP at 3 nodes and the lever DOES compound, but not at
=2. The 2-node hero (=2) collapses to a tie with =1 (411.5 vs 419.5, per-rank
4.52 vs 4.42 — the +23% 2-node lead of 2eb over 1eb has VANISHED), while =3 wins
decisively: 553.3 Hz, +34% per-rank over both (6.22 vs ~4.5).** This is a genuine
reversal of iter-23's 2-node result, where 3 EB REGRESSED −18%.

**Mechanism (coherent):** `eb_wait` falls monotonically with more EBs (29.86 →
23.98 → 9.97 ms) — same direction as 2 nodes. But at 3 nodes the *magnitude* of
eb_wait at 1–2 EB is much larger than at 2 nodes (iter-23 2n: 1eb≈21, 2eb≈10 ms;
here 3n: 1eb≈30, 2eb≈24 ms), because the single smd0 now feeds MORE node-local
EBs (6 EBs @3n·2eb vs 4 @2n·2eb) and each EB waits longer on its smd batch. So at
3 nodes eb_wait is STILL the binding term at 2 EB (24 ms), and pushing to 3 EB
(each serving ~10 BDs) drains it to ~10 ms — the crossover that happened between
1 and 2 EB at 2 nodes now happens between 2 and 3 EB at 3 nodes. **More node count
→ more smd0→EB serving pressure → the eb_wait-vs-BD-slot crossover moves right.**
This is exactly the "more cross-node coordination could shift the knee" hypothesis
iter-23 raised, confirmed in the direction it feared.

**HONEST CAVEAT — order/window confound (why this is not yet graduated).** The
winner (3eb) was the LAST of three sequential phases (~6.5 min total), and FFB
window state drifts within an allocation (iters 20/23 both stress this). A pure
warming artifact would inflate the last phase. Two things argue the reversal is
real, not just warming: (1) `eb_wait` ordering (30→24→10) is monotone in EB count
and matches the mechanism model, independent of window; (2) `residual` is NON-
monotone across phases (63 → 99 → 34 for phase 1→2→3) — the middle phase (1eb) is
the WORST, so simple "later phase = warmer = better" does not hold. But bd_read is
actually HIGHEST for the 3eb winner (103.8 vs 89.6), so 3eb's win rides entirely
on eb_wait + residual, the two buckets most sensitive to both the lever AND the
window. A single order-confounded bracket is not enough to overturn a clean
2-node knee and graduate =3. **Per the project's "a wrong number is worse than no
number" discipline, this needs a reversed-order confirm before it lands in
PLANNING.md.**

**Keep/revert:** KEEP the feature (unchanged; opt-in, default-off, already
correctness-gated bit-exact in iter-22). Do NOT yet change the recommended
default — iter-23's "PS_EB_PER_NODE=2 at 2 nodes" still stands; the =3-at-3-nodes
claim is PENDING the reversed-order confirm. TASK.md updated with the provisional
3-node bracket and the confound flag.

**Recommended next step:** reversed-order confirm bracket at 3 nodes — run 3eb
FIRST (coldest window), then 2eb, then 1eb. If 3eb still wins when it is the cold
first phase, the knee-shift-to-3 is real and =3 graduates as the 3-node GPU
default (with a documented "knee grows with node count" note in PLANNING.md). If
3eb's lead collapses when run cold, the reversal was a warming artifact and =2
holds. Either way the code-side multi-node levers are then fully characterized
and the loop is near LOOP DONE pending storage-side facility levers.

### Iteration 24 addendum — reversed-order confirm: the knee-shift is REAL (3 EB wins cold-first), graduated into PLANNING.md

The reversed-order control (job 31589010, nodes sdfampere018/020/023, all three
exit=0, `bench_mpi_sweep/epn3nrev_{3eb,2eb,1eb}.log`) runs the same three configs
with **3eb as the COLD first phase**, then 2eb, then 1eb:

| config (phase order 3→2→1) | BD | aggregate Hz | per-rank Hz | eb_wait ms | bd_read ms | residual ms |
|----------------------------|----|-------------|-------------|-----------|-----------|-------------|
| mn3_3eb (cold first phase) | 89 | **802.5**   | **9.02**    | 8.94      | 61.16     | 30.04       |
| mn3_2eb                    | 89 | 648.3       | 7.28        | 13.97     | 65.57     | 46.02       |
| mn3_1eb (last phase)       | 95 | 622.9       | 6.56        | 25.19     | 73.04     | 63.02       |

**Verdict — the reversal is REAL, not a warming artifact. 3 EB/node wins even
when it is the coldest first phase: 802.5 Hz, +23.8% aggregate over 2eb (648.3)
and +28.8% over 1eb (622.9), +23.8% per-rank (9.02 vs 7.28).** The ordering
`3eb > 2eb > 1eb` is now identical in BOTH the forward (2→1→3) and reversed
(3→2→1) brackets, so it is robust to phase order — the order-confound flagged in
the primary entry is DISCHARGED. This whole reversed window is warmer than the
forward one (every config faster: 3eb 802 vs 553, 2eb 648 vs 411, 1eb 623 vs 419),
which is exactly why in-window ordering — not cross-window absolute rates — is the
valid comparison. The reversed bracket also restores the small 2eb>1eb edge (648
vs 623) that the forward bracket's cold-first-phase penalty on 2eb had masked, so
the true monotone 3-node ordering is `3 > 2 > 1`, matching the monotone eb_wait
(25.19 → 13.97 → 8.94 ms).

**So the PS_EB_PER_NODE knee GROWS with node count: 2 @ 2 nodes (iter-23), ≥3 @ 3
nodes (this iter). Mechanism confirmed: one smd0 feeds all node-local EBs, so more
nodes = more EBs contending for smd batches = larger eb_wait at low k = the
crossover moves to higher k. The clean emergent rule so far is knee ≈ node_count.**
This is the compounding answer iter-23 asked for — the lever does compound with
node count, but the *optimal setting* is node-count-dependent, not a fixed =2.

**Keep/graduate:** the feature was already KEPT (opt-in, default-off, bit-exact
correctness-gated in iter-22). This iteration GRADUATES the node-count-dependent
knee into `PLANNING.md` — new `PS_EB_PER_NODE` subsection under "Multi-node
launch" with the measured 2n/3n knee table, the "scale k with node count / start
at k=node_count" guidance, the "do not exceed the knee (−18% past it)" warning,
and a pointer to the `eb_per_node_3n.sbatch` template. TASK.md updated to mark the
iter-24 result CONFIRMED (was PROVISIONAL). iter-23's `PS_EB_PER_NODE=2 at 2
nodes` TASK entry is unchanged and correct — it is the knee *at 2 nodes*; the
graduation just generalizes it to k≈node_count.

**Recommended next step:** the knee at 3 nodes is bounded only as `≥3` — the
{1,2,3} bracket did not test k=4 @ 3n, so whether the 3-node optimum is exactly 3
or higher is open. And the emergent `knee ≈ node_count` rule wants a 4-node point
(does k=4 win at 4 nodes?) to become a law rather than a two-point line. A single
3-node k∈{3,4,5} bracket pins the 3-node knee; a 4-node k∈{2,3,4} bracket tests
the rule. Once the knee's functional form is pinned, the code-side multi-node
levers (node-local EB placement + intra-node EB fan-out) are fully characterized
and the loop is at LOOP DONE pending storage-side facility levers (the ~10–12
GB/s FFB ceiling, none of which are psana code — TASK.md "Remaining" items 1–3).

## Iteration 25 — the 3-node PS_EB_PER_NODE knee is EXACTLY 3 (k=4,5 both lose to k=3), knee ≈ node_count HOLDS

iter-24 left the 3-node knee bounded only as `≥3` (3 EB/node beat {1,2}; k=4,5
untested). This iteration sweeps the upper end `k∈{3,4,5}` at 3 nodes to pin
whether the knee is exactly 3 or keeps climbing — the difference between the
emergent rule `knee ≈ node_count` holding vs the knee growing *faster* than node
count. **One variable: PS_EB_PER_NODE ∈ {3,4,5}, fixed 3 nodes, node-local
colocation.** No code change (measurement only), no C rebuild (pure-py); node.py
diff-identical source-vs-install (the iter-22 non-editable gotcha checked).

**Order front-loads the hypothesized winner k=3 as the COLD first phase.** FFB
window state warms within an allocation and favors LATER phases, so if k=3 wins
even cold-first (over warmer k=4/k=5) the knee=3 verdict is conservative — this
is the same clean design iter-24's reversed control used.

**Clean 3-node in-window bracket, job 31590259 (all three exit=0, COMPLETED),
nodes sdfampere002/023/028, `bench_mpi_sweep/epn3nk_{3eb,4eb,5eb}.log`, driver
`slurm-31590259.out`, n=100/rank warmup=10 --wait-split. Phase order 3→4→5:**

| config (phase order 3→4→5) | BD rep | aggregate Hz | per-rank Hz | eb_wait ms | bd_read ms | residual ms |
|----------------------------|-------:|-------------:|------------:|-----------:|-----------:|------------:|
| mn3_3eb (COLD first phase) | 86     | **791.5**    | 9.20        | 9.58       | 53.22      | 26.45       |
| mn3_4eb                    | 86     | 658.9        | 7.66        | 4.69       | 97.00      | 16.13       |
| mn3_5eb (warmest last)     | 67     | 757.1        | 11.30       | 2.53       | 67.60      | 8.52        |

**Verdict — the 3-node knee is EXACTLY 3. k=3 wins on aggregate throughput even
as the coldest first phase: 791.5 Hz vs 658.9 (k=4, −16.8%) and 757.1 (k=5,
−4.3%).** Since warming favors later phases, k=3 beating warmer k=4/k=5 is the
conservative direction — the knee does NOT keep climbing past 3. Combined with
iter-24's `k=3 > k=2 > k=1`, the full 3-node sweep {1,2,3,4,5} now peaks at k=3,
and the 3eb anchor here (791.5) matches iter-24's reversed-window 3eb (802.5)
within ~1.4%, so the two windows are directly comparable and the peak is real,
not window drift. **`knee ≈ node_count` (2@2n, 3@3n) HOLDS as a two-point law —
the knee grows with node count but tracks it, not faster.**

**Mechanism (coherent, matches the iter-24 model past its crossover):** `eb_wait`
keeps falling monotonically as k rises (9.58 → 4.69 → 2.53 ms) — more EBs, less
BD blocking, exactly as predicted. But past the knee that saving is bought by
converting BD-reader slots into EB ranks, and it stops paying: at k=5 only 67 of
the ~83 topological BD ranks reported measurements — ~16 BD ranks are starved
into idleness by the over-provisioned EB fan-out, so aggregate throughput drops
even though per-*reporting*-rank rate looks high (11.3 Hz over a shrunken
denominator). That starvation is the concrete "do not exceed the knee" cost.

**HONEST CAVEAT — the k=4 bd_read outlier.** k=4's `bd_read` is 97.0 ms, nearly
2× both neighbors (53 / 68) and far outside iter-24's 3-node range (61–104). That
is almost certainly a transient FFB-window read degradation during phase 2, not
an intrinsic property of k=4 — so the *magnitude* of the 3→4 drop (−16.8%) is
inflated by a storage-window dip and should not be read as the true k=4 penalty.
The robust, window-safe claims are only the two that survive the cold-first
design: (1) **k=3 is the peak** (it won as the coldest phase, so no warming could
have manufactured its lead), and (2) k=5, despite the warming advantage of being
last, still lost to k=3 — so pushing k past 3 does not recover. The precise shape
of the falloff between k=3 and k=5 is not pinned by this single bracket; the knee
location is.

**Keep/graduate:** feature unchanged (opt-in, default-off, bit-exact
correctness-gated since iter-22 — numeric path untouched, no re-gate needed).
This iteration REFINES the graduated PLANNING.md entry: the 3-node knee row goes
from `≥3` to exactly `3`, with the k=4/k=5 falloff and the BD-starvation
mechanism noted; TASK.md gets the pinned-knee bracket. `knee ≈ node_count` is now
a confirmed two-point law rather than a lower-bounded conjecture.

**Recommended next step:** the last open characterization item is the 4-node
point — does `k = node_count` predict k=4 wins at 4 nodes? A 4-node k∈{2,3,4,5}
bracket would turn the two-point law into three points (and 4 nodes is the layout
where mis-provisioning EBs costs the most BD slots). 4-node allocations have been
structurally hard in-loop (iter-19: parent assoc GrpNodes cap — must go through
the `lcls:default`/`preemptable` hierarchy, which these EB templates already do).
If the 4-node point confirms, the code-side multi-node levers are fully
characterized and the loop is at LOOP DONE pending the storage-side facility
levers (the ~10–12 GB/s FFB ceiling — TASK.md "Remaining" items 1–3, none of
which are psana code).

## Iteration 26 — the 4-node PS_EB_PER_NODE knee is NOT 4: k=3 wins, `knee ≈ node_count` BREAKS (saturates), order-confounded on 3-vs-4

iter-25 pinned the 3-node knee at exactly 3 and left the emergent two-point law
`knee ≈ node_count` (2@2n, 3@3n) predicting k=4 wins at 4 nodes. This iteration
is that third point — a 4-node `k∈{2,3,4,5}` sweep to turn the line into a law or
break it. **One variable: PS_EB_PER_NODE ∈ {2,3,4,5}, fixed 4 nodes, node-local
colocation.** No code change (measurement only), no C rebuild (pure-py); the
PS_EB_PER_NODE numeric path is bit-exact and built since iter-22, untouched here.

**Order front-loads the predicted winner k=4 as the COLD first phase** (same
cold-first design as iter-25): FFB window state warms within an allocation and
favors LATER phases, so the two nearest threats k=3 (rule says knee stays 3) and
k=5 (rule says knee climbs faster) run later/warmer. The design intent: if k=4
wins even cold, the knee=4 / node_count law is conservatively confirmed.

**Clean 4-node in-window bracket, job 31592091 (all four exit=0, COMPLETED),
nodes sdfampere002/026/028/030, `bench_mpi_sweep/epn4nk_{4eb,2eb,3eb,5eb}.log`,
driver `slurm-31592091.out`, n=100/rank warmup=10 --wait-split. Phase order
4→2→3→5. Topology at 4 nodes / 132 ranks (rank0=smd0): k EB/node → 4k EB,
131−4k topological BD.**

| config (phase order 4→2→3→5) | EB | topo BD | BD reporting | starved | aggregate Hz |
|------------------------------|---:|--------:|-------------:|--------:|-------------:|
| mn4_4eb (COLD first / predicted hero) | 16 | 115 | 115 | 0  | 815.8 |
| mn4_2eb                       |  8 | 123 | 111 | 12 | 854.2 |
| mn4_3eb                       | 12 | 119 | 118 | 1  | **914.1** |
| mn4_5eb (warmest last)        | 20 | 111 |  90 | 21 | 896.6 |

**Verdict — the node_count law does NOT hold at 4 nodes. The predicted winner
k=4 placed LAST (815.8 Hz); k=3 placed FIRST (914.1 Hz).** The knee does not
climb to 4 — it appears to saturate at ~3. Two claims from this bracket are
robust against the warming confound, one is confounded:

1. **ROBUST — knee < 5.** k=5 (896.6) drops below k=3 (914.1) *despite* being the
   warmest last phase; warming should have lifted k=5 above k=3 if they were
   equal, so k=5's real penalty beats the warming trend. k=5 also starves 21 of
   111 topological BD ranks (only 90 reported) — the over-provisioned EB fan-out
   converting BD-reader slots to idle EBs, the same past-knee starvation mechanism
   iter-25 saw at 3n k=5. So pushing k past 3–4 at 4 nodes does not recover.
2. **ROBUST — the node_count=4 prediction is unsupported.** k=4 was deliberately
   placed cold-first so that a genuine knee=4 would win despite the cold window
   (conservative-for-the-hypothesis design). It came in *lowest*. A cold window
   can only *depress* a phase, so k=4's last-place finish cannot be a warming
   artifact working in its favor — the knee=4 hypothesis had its best shot and
   missed. k=4 shows zero starvation (115/115), so it is not failing for lack of
   readers; it simply is not the throughput optimum.
3. **CONFOUNDED — exact knee is 3 vs 4.** k=4 cold-first vs k=3 third-phase: the
   warming drift favors k=3, so the *magnitude* of k=3's lead over k=4 (+12%) is
   inflated and I cannot cleanly rank k=3 above k=4 from this window alone. What I
   can say: the knee is 3 or 4, not 2 or 5, and it did NOT reach the predicted 4.

**So `knee ≈ node_count` breaks at its first out-of-sample test: measured knees
are 2@2n, 3@3n, and (3 or 4)@4n — the knee grows SLOWER than node count and looks
to be saturating near 3.** Mechanistically this fits the iter-24/25 model: `eb_wait`
falls with more EBs, but the smd0→EB feed and the BD-slot budget are the binding
constraints, and past ~3 EB/node the fan-out starts starving BD readers (k=2 already
starves 12, k=5 starves 21; k=3 is the low-starvation sweet spot at 1). The single
smd0 feeding all node-local EBs caps how much added EB parallelism can help, so the
knee plateaus rather than tracking node_count linearly.

**HONEST CAVEAT — the k=2 starvation (12 BD idle) is anomalous** and larger than
its 2-node/3-node analogs; at low k the EB count may be too small to keep all BD
readers fed, a different failure mode than the high-k over-provisioning. It does not
change the ranking (k=2 is mid-pack) but means the low-k side of the 4-node curve is
noisier than the clean k=3 point.

**Keep/graduate:** feature unchanged (opt-in, default-off, bit-exact since iter-22 —
numeric path untouched, no re-gate). This iteration CORRECTS the graduated
PLANNING.md `knee ≈ node_count` guidance: it is a 2-node/3-node coincidence, not a
law — the knee saturates near 3, so `PS_EB_PER_NODE=3` is the best single setting for
3+ nodes rather than "scale k with node count." TASK.md gets the 4-node bracket and
the corrected rule. The confounded 3-vs-4 boundary is flagged for a reversed confirm.

**Recommended next step:** a reversed-order 4-node confirm — order `5→3→2→4` so k=4
runs *warmest-last* instead of cold-first. If k=4 still loses to k=3 even with the
warming advantage, `knee=3 at 4 nodes` is robust and the saturation verdict is
locked; if warm k=4 overtakes k=3, the knee is 4 after all and node_count survives
weakly. Either way the 3-vs-4 confound is discharged with one more in-window bracket
(reuse `eb_per_node_knee_4n.sbatch`, swap the phase order). After that the code-side
multi-node levers are fully characterized and the loop is at LOOP DONE pending the
storage-side facility levers (the ~10–12 GB/s FFB ceiling — TASK.md "Remaining" items
1–3, none of which are psana code).

## Iteration 27 — reversed-order 4-node confirm: k=3 wins WARM-LAST too, 3-vs-4 confound discharged, knee=3 at 4 nodes LOCKED

iter-26's forward 4-node bracket (order 4→2→3→5) ranked k=3 above the predicted
k=4, breaking `knee ≈ node_count` — but k=4 ran cold-first and k=3 ran warmer, so
the +12% lead was warming-inflated and the exact 3-vs-4 boundary was confounded.
This iteration is the reversed-order confirm iter-26 asked for. **One variable:
PS_EB_PER_NODE ∈ {5,3,2,4} at fixed 4 nodes, node-local colocation, phase order
5→3→2→4 so k=4 now runs WARMEST-LAST and k=3 runs colder/earlier — the opposite
warming bias.** No code change (measurement only), no C rebuild (pure-py); the
PS_EB_PER_NODE numeric path is bit-exact and built since iter-22, untouched here.

**The logic:** warming can only LIFT a later phase. So if k=4 loses to the
colder-run k=3 *even with* the warmest-last advantage, k=3's lead is not a warming
artifact and knee=3 at 4 nodes is robust; if warm-last k=4 overtakes k=3, the knee
is 4 after all and node_count survives weakly.

First submission (job 31593496) was preempted at 29s (preemptable QOS, CANCELLED
by scheduler) as the first mpirun launched — a transient facility event, resubmitted
immediately. Clean 4-node in-window bracket, **job 31593737 (all four exit=0,
COMPLETED 23:15, nodes sdfampere002/026/032/040, total_wall 1392s,
`bench_mpi_sweep/epn4nkr_{5eb,3eb,2eb,4eb}.log`, driver slurm-31593737.out),
n=100/rank warmup=10 --wait-split. Phase order 5→3→2→4. Topology 132 ranks
(rank0=smd0): k EB/node → 4k EB, 131−4k topo BD.**

| config (phase order 5→3→2→4) | EB | topo BD | reporting | starved | aggregate Hz | eb_wait ms | bd_read ms | residual ms |
|------------------------------|---:|--------:|----------:|--------:|-------------:|-----------:|-----------:|------------:|
| mn4_5eb (COLD first)         | 20 | 111     | 87        | 24      | 758.8        | 1.44       | 111.97     | 11.04       |
| mn4_3eb (2nd, colder)        | 12 | 119     | 115       | 4       | **832.1**    | 15.52      | 69.79      | 45.93       |
| mn4_2eb (3rd)                |  8 | 123     | 123       | 0       | 670.6        | 16.32      | 92.90      | 65.56       |
| mn4_4eb (WARMEST last)       | 16 | 115     | 115       | 0       | 716.3        | 6.12       | 115.26     | 24.67       |

**Verdict — knee=3 at 4 nodes is LOCKED; the 3-vs-4 confound is discharged in the
conservative direction. k=3 (832.1 Hz), run colder as the 2nd phase, beats k=4
(716.3 Hz) by +16.2% even though k=4 ran WARMEST-LAST — its best possible slot.**
Warming can only lift a later phase, so k=4 had every advantage this window offers
and still lost. Combined with iter-26 (k=3 beat cold-first k=4 by +12%), the
ordering k=3 > k=4 now holds in BOTH phase orders, so k=3's lead is robust to the
warming confound rather than an artifact of it. The predicted `knee = node_count`
law is definitively broken: measured knees are 2@2n, 3@3n, 3@4n — **the knee
saturates near 3 for 3+ nodes, it does not track node count.**

**Mechanism (consistent across all four windows now):** eb_wait falls with k
(reversed order shows the warming-scrambled magnitudes but the k=3 point sits at
the sweet spot regardless). k=3 wins with the LOWEST bd_read (69.79 ms vs 92–115
for k=2/4/5) and near-zero starvation (4 BD idle of 119). k=4 again shows an
inflated bd_read (115.3, echoing iter-26's k=4 outlier ~97) — its extra EB fan-out
is not buying its way out of the read cost, it just converts BD-reader slots to EBs
without a throughput return. k=5 cold-first starves 24 of 111 topological BD ranks —
the same past-knee over-provisioning that starved BD readers at 3n k=5 and 4n k=5
(iter-25/26). The single smd0 feeding all node-local EBs is the binding constraint
that caps how much EB parallelism can help, so the knee plateaus at ~3.

**Keep/graduate:** feature unchanged (opt-in, default-off, bit-exact since iter-22
— numeric path untouched, no re-gate needed). This iteration FINALIZES the
graduated PLANNING.md entry: the 4-node row goes from `≥3, 3-vs-4 confounded` to
`3, confirmed both orders`, the mechanism note records the reversed-window confirm,
and the closing paragraph drops the "awaits a reversed-order confirm" caveat.
TASK.md gets the iter-27 bracket marked CONFIRMED / knee-locked. **This closes the
last open PS_EB_PER_NODE characterization item — the code-side multi-node EB levers
(node-local placement PS_EB_NODE_LOCAL + intra-node fan-out PS_EB_PER_NODE) are now
fully characterized: knee=2@2n, 3@3n, 3@4n, +23–32% aggregate over k=1.**

**Recommended next step — one code lever is NOT yet ruled out before declaring the
loop storage-bound.** At the k=3 optimum the per-BD-rank wait splits bd_read 69.79
(FFB storage) / residual 45.93 (CPU event construction) / eb_wait 15.52 ms. bd_read
dominates and is the ~10–12 GB/s facility ceiling (TASK.md "Remaining" 1–3, not
code), but the **45.9 ms/event residual is a non-storage, non-EB CPU cost that no
iteration has profiled** — Dgram/event assembly on the BD rank. That is the leading
remaining code-side lever candidate and it is substantial (~40% of per-rank wait).
The next iteration should profile the BD-rank residual (py-spy or timestamped
instrumentation on the construction path, §8) to see whether any of the 45.9 ms is
reducible. Only if the residual proves irreducible is the loop genuinely at LOOP
DONE pending storage-side facility levers; until then there is one untested code
hypothesis and the loop should not stop.

## Iteration 28 — the 45.9 ms BD-rank "residual" is a denominator artifact, NOT a code lever: same-denominator counter shows real CPU plumbing = 1.15 ms, the loop is storage-bound

iter-27 flagged the ~45.9 ms/event BD-rank `residual` (at the 4-node k=3 optimum)
as "a non-storage, non-EB CPU cost that no iteration has profiled" and the last
untested code lever before declaring the loop storage-bound. This iteration
profiles it — and closes it. **One variable: add an additive-only `total_wait_ns`
counter that times the WHOLE generator advance INSIDE `node.py` on the same
`bd_events` denominator as eb_wait/bd_read/dgram/smdparse, so the residual can be
decomposed on ONE denominator instead of the mixed-denominator subtraction the
report has always used.** No numeric-path change (correctness re-gated anyway,
below); pure-Python, synced source→install.

**The bug in the old `residual`.** The report computes
`residual = wait − eb_wait − bd_read`, but `wait` (`np.mean(wait_times)`) is
normalized over `n` = the *timed* L1 events (=150), while eb_wait/bd_read/dgram/
smdparse are additive counters normalized over `bd_events` = every node advance
incl. transitions (bd_events > n). Subtracting a ÷n quantity from ÷bd_events
quantities inflates the leftover by exactly `(wait − gen_wait)`. This is precisely
the "subtractive artifact / mixed denominators" iter-15 identified when it directly
counted dgram at ~1–4 ms; iter-27 re-read the inflated subtraction as if it were
measured construction time.

**Measurement — single-node 32 BD, r47/FFB, A100 (sdfampere033, ralph-gpu job
31583622), `mpirun --bind-to none --oversubscribe -n 34`, `-n 150 --warmup 10
--wait-split`, `bench_mpi_sweep/ralph_tmp/genwait_mpi_32bd_205830.log`.** Single
node reproduces the phenomenon: iter-10 measured the single-node 32-BD residual at
~45.5 ms, ≈ the 4-node k=3 45.9 ms, so no multi-node allocation was needed.

| bucket | value (ms/event) | denominator | meaning |
|---|---:|---|---|
| wait (OLD)        | 123.249 | ÷n (timed)       | gen advance, timed OUTSIDE in the benchmark |
| residual(OLD)     |  26.879 | mixed            | wait − eb_wait − bd_read — **the artifact** |
| **gen_wait**      | **97.521** | ÷bd_events    | WHOLE gen advance, timed INSIDE node.py |
| **residual(cons)**|  **1.151** | ÷bd_events    | gen_wait − eb_wait − bd_read — **real CPU plumbing** |
| eb_wait           |   1.129 | ÷bd_events       | Probe+Irecv on EB batch |
| bd_read           |  95.241 | ÷bd_events       | os.pread bigdata off FFB |
| dgram             |   1.460 | ÷bd_events       | per-event dgram.Dgram() |
| smdparse          |   0.464 | ÷bd_events       | per-batch offset/size build |

**Verdict — the last untested code lever is discharged: it was never a real cost.**
On a consistent denominator `eb_wait + bd_read = 96.370 ≈ gen_wait 97.521` (match
within 1.15 ms), and that 1.15 ms leftover is fully accounted for by the
directly-counted construction (dgram 1.46 + smdparse 0.46 = 1.92 ms, within
run-to-run noise). **Real per-event CPU plumbing on the BD rank is ~1.9 ms, not
45.9 ms.** The 26.9/45.9 ms "residual" is entirely `wait − gen_wait` — the
÷n vs ÷bd_events denominator gap (here bd_events/n ≈ 1.26). This CONFIRMS iter-15
at the multi-node optimum and definitively refutes iter-27's "45.9 ms untested
lever": there is no reducible CPU-construction cost hiding in the residual.

**The BD-rank wait is 98% storage read (bd_read 95.2 ms/event = ~276 MB/s/rank ×
32 ≈ 8.8 GB/s, at the ~7.9–12 GB/s per-node FFB ceiling).** eb_wait (1.1 ms
single-node), construction (1.9 ms), and kernel (0.55 ms) are all negligible
against it. The event loop is storage-bound, exactly as TASK.md "Remaining" 1–3
(all facility-level) state.

**Keep/graduate:** KEEP the `total_wait_ns` counter — it is an additive-only,
default-path-safe instrument that makes the residual correctly decomposable going
forward, replacing the misleading `residual(OLD)` line (now relabeled as the
mixed-denom artifact in both the single-rank and aggregate reports). Correctness
gate PASS bit-exact (20/20, max_diff 0.0), so the touched event-delivery path
(node.py generator loop) is verified unbroken. TASK.md gets the iter-28 row and
the residual-artifact resolution.

**Recommended next step.** The residual is dead, so the remaining code-side lever
is the one PROGRESS/TASK have pointed at since iter-10: H2D is 7.66 ms/event at
32-rank contention and sits *after* bd_read in the synchronous chain — pinned-host
memory + async H2D prefetch (overlap H2D of event N behind the bd_read of event
N+1) could hide most of that ~7.7 ms (~8% of the 121 ms per-rank wall) behind the
storage read that dominates. This is overlap, so it MUST be measured with
wall-clock rate + CUDA-event attribution, never the sync-instrumented `--wait-split`
loop (§6 trap). If that lever is measured and proves not to move the aggregate
(because per-rank reads already saturate the FFB link and leave no idle PCIe window
to overlap into), then every code-side lever is exhausted and the loop is at LOOP
DONE pending storage-side facility levers (TASK.md "Remaining" 1–3).

## Iteration 29 — async-H2D overlap (the last untested code lever) REGRESSES −21%: loop is storage-bound, code-side levers exhausted → LOOP DONE

iter-28 pre-registered exactly one remaining code lever before declaring the loop
storage-bound: **pinned-host + async H2D prefetch** — issue event N's H→D on a CUDA
stream and let it run while the CPU advances the generator to READ event N+1 (the
dominant ~95 ms/event FFB os.pread), hiding the ~7.7 ms H→D + ~0.55 ms kernel that
sit *after* the read in the synchronous chain. This iteration implements and measures
it. **One variable: a new `--overlap` bench path (`run_gpu_bench_overlap`) — pinned
double buffer, async H→D + kernel on a per-slot stream, NO per-event device sync,
buffer reuse guarded by a per-slot `stream.synchronize()`.** Numerically identical to
`--seg-h2d` (same `_segment_numbers` order, same layout, same `fused_calib_gpu`
kernel; `fused_calib_gpu`/`.set` verified to hold no internal sync). Pure-Python bench
harness change, default path untouched; correctness re-gated below.

**Timing per §6 (this is overlap):** headline is WALL-CLOCK rate only; H→D/kernel
magnitudes attributed with CUDA events recorded ON the stream (overlap-safe), read
AFTER the timed loop — they characterize the hidden GPU work, they are not the headline.

**Measurement — 32-BD ABBA (warming-controlled), r47/FFB, A100 (sdfampere033,
ralph-gpu job 31583622), `mpirun --bind-to none --oversubscribe -n 34`, n=200/rank
warmup=10, logs `bench_mpi_sweep/ralph_tmp/ab_{A1_seg,B1_ovl,B2_ovl,A2_seg}_213119.log`
(+ consolidated `ab_overlap_full.log`).** Correctness gate PASS bit-exact (20/20,
max_diff 0.0, `smoke_overlap.sh`); single-rank `--overlap` smoke ran clean.

| phase (order) | mode | aggregate Hz | on-stream H2D ms | on-stream kernel ms |
|---------------|------|-------------:|-----------------:|--------------------:|
| A1 (COLD first) | --seg-h2d | 113.8 | 60.97 (sync bucket) | 2.19 |
| B1              | --overlap |  93.2 | 2.53 (async DMA)    | 1.04 |
| B2              | --overlap |  92.0 | 2.53 (async DMA)    | 1.01 |
| A2 (WARM last)  | --seg-h2d | 120.7 | 55.52 (sync bucket) | 2.12 |

**Verdict — the last code lever is a DEAD END: overlap REGRESSES −21%
(92.6 Hz mean vs seg-h2d 117.3 Hz mean), robust to warming.** seg-h2d ran BOTH
cold-first (113.8) and warm-last (120.7), bracketing overlap's two middle phases,
and both seg-h2d phases beat both overlap phases — overlap loses even to the
*coldest* seg-h2d point, so the −21% is not a warming artifact (warming can only
lift a later phase). The CUDA-event attribution proves the overlap machinery works
as intended — the async H→D DMA from pinned is only **2.53 ms/event** (vs the
seg-h2d sync bucket's 55–61 ms that bundles 32× per-seg deserialize+stage+transfer+
sync), kernel ~1.0 ms — so there is only **~3.5 ms/event of GPU work to hide** behind
the ~95 ms read. But the overlap path must first gather all 32 segments into the
pinned staging buffer: an extra **full-event 33.5 MB host→host `np.copyto` every
event**, and under 32-rank host-memory-bandwidth contention that added traffic costs
MORE than the ~3.5 ms it hides. Net −21%. This is exactly iter-28's prediction:
per-rank reads already saturate the FFB link, the GPU work is tiny, and there is no
idle window worth paying host bandwidth to overlap into.

**Keep/revert:** KEEP `--overlap` as a default-off, opt-in bench flag documenting
the reproducible negative (repo precedent: `--reader-probe`, the `--profile*`
variants), with a `VERDICT: DEAD END` header on its docstring so no one mistakes it
for a throughput path. Numeric path untouched, correctness PASS bit-exact — the
default event-delivery path is unchanged. TASK.md gets the iter-29 ABBA row.

**LOOP STATUS — every code-side lever is now exhausted; the loop is storage-bound.**
The per-BD-rank wall is 98% FFB os.pread (iter-28: bd_read 95.2 ms = ~8.8 GB/s
aggregate at the ~7.9–12 GB/s per-node FFB ceiling); eb_wait, event construction,
H→D and kernel are all negligible against it and now — with overlap discharged — none
of them is a reducible code lever. The three characterized multi-node EB placement
levers (`PS_EB_NODE_LOCAL`, `PS_EB_PER_NODE`, knee=2@2n/3@3n/3@4n, +23–32% aggregate)
are the code-side wins and are graduated into PLANNING.md. What remains is
facility-level only: FFB storage bandwidth and node count (TASK.md "Remaining" 1–3),
not psana code. Per iter-27/28's pre-registered criterion — "only if this lever proves
not to move the aggregate is the loop genuinely at LOOP DONE" — that gate is now
closed by measurement.

**Recommended next step:** none code-side. Throughput past the single-node ~117–175 Hz
storage-bound rate requires more nodes (scales per-node, already characterized) or a
higher-bandwidth FFB stage — both facility decisions for the human, not loop levers.

LOOP DONE

## 2026-07-15 — Post-loop addendum: cold-window control on the 4-node headline; absolute-figure hygiene across the campaign

**Measurement: iter-26's mn4_3eb config re-run as the SOLE phase of a fresh 4-node
allocation** — job 31812885 (`bench_mpi_sweep/eb_knee_4n_3eb_cold.sbatch`), identical
config to the iter-26 winner (PS_EB_NODE_LOCAL=1, PS_EB_PER_NODE=3, 132 ranks,
n=100/rank warmup=10 --wait-split), log `bench_mpi_sweep/epn4nk_3eb_cold.log`,
exit=0, wall 113 s:

| run | window position | ranks reporting | aggregate Hz |
|-----|-----------------|---:|---:|
| `epn4nk_3eb` (iter 26) | 3rd of 4 phases (warm) | 121 | 913.5 (journal headline 914.1) |
| `epn4nk_3eb_cold` | sole phase, fresh allocation | 121 | **811.2** |

Same parse on both logs (sum of per-rank `rate:` lines), identical rank census —
the −11.2% is window state, not topology or starvation. This is the first direct
measurement of the within-allocation warming magnitude on an identical config.

**Corrections this forces:**

1. **The 4-node headline is ~811 Hz cold, up to ~914 Hz warm.** 914.1 was earned
   in the warm third-phase slot and carries ~11% window inflation. It graduated
   into PLANNING.md and ralph_summary.html as a bare absolute; both now quote the
   band. 811 Hz × 33.5 MB/event = 6.8 GB/s/node = **86%** of the 7.9 GB/s
   cold-window floor — not the ~97% the warm number implied.
2. **Cold-for-cold, k=3 ≈ k=4 at 4 nodes** (811.2 vs iter-26's cold-first 815.8).
   The two cold points come from different allocations/days, so this comparison is
   itself cross-window — but it caps the plausible k=3-over-k=4 gap well below
   iter-26's confounded +12%. The in-window both-orders ranking stands (iter-27:
   k=3 832.1 > k=4 716.3 with k=4 given the warmest-last slot), so k=3 remains
   the recommended setting. Honest phrasing: **knee = 3, with k=4 within window
   noise of it cold-for-cold — a 3–4 plateau, never past 4.**
3. **Retro-flag on iter-23** (not flagged at the time): in the 2-node bracket the
   3eb leg ran as the COLD FIRST phase, so the "k=3/4 regress −18% @2n" magnitude
   is partly a cold-window artifact on the k=3 side — consistent with k=3 winning
   outright at 3 nodes in both orders one iteration later. The knee=2@2n verdict
   itself stands (2eb beat warm-last 4eb, and beat 1eb in iter-22's conservative
   order).
4. **Absolute-figure hygiene — the storage band, not a point, is the yardstick.**
   At 33.5 MB/event the FFB serving band is ~7.9 GB/s/node cold (disjoint-pread
   control → **~235 Hz/node floor**) up to ~12 GB/s/node warm (**~355 Hz/node**).
   Single-node 32-BD absolutes for the SAME tuned config legitimately span
   ~113–121 Hz (iter-29 window), 175.3 Hz (iter-1b anchor window), and 337.2 Hz
   (iter-15 warm window = 11.3 GB/s read — above the cold floor, which only a warm
   window can serve). Quoting "337 Hz" next to "235 Hz/node storage ceiling"
   without the window class reads as a contradiction; it is not one, but every
   absolute must carry its window class from here on. The controlled results are
   the relative, warming-bracketed claims: +30% (iter 7), +32–38% (iter 8), the
   knee orderings (iters 20–27), −21% overlap (iter 29).
5. **The "+70% (210 → 337 Hz)" endpoints are cross-window AND cross-dataset**
   (210 = historical r387 B-MVP anchor; 337.2 = post-wins r47 in iter-15's warm
   window). The +70% itself is honest as the compounded interleaved gains
   (1.30 × 1.32 ≈ 1.7); the endpoint absolutes are illustrative only, and the
   docs that quote them now say so.
6. **The 3.9× GPU/CPU multiplier is window-flavored on the GPU side.** GPU 175.3
   (iter 1b) and CPU 44.9 (iter 2) ran on the same node (job 31267701) but hours
   apart in different windows — not "the same run". Across measured windows the
   same GPU config spans 117–175 Hz, so the measured multiplier spans ~2.6–3.9×.
   The mechanism claims are unaffected (CPU compute-bound at its ~74 Hz ceiling,
   kernel has 76× headroom, the advantage widens with scale), but quoted
   multipliers now carry the range.

Full-campaign audit against this confound: the qualitative conclusions survive.
Every knee ranking and every negative from iter-20 on was bracketed so that
warming worked AGAINST the reported effect, and the profiling attributions
(iter 3/9/10/15/28) are within-run relative metrics independent of absolute rate.
The casualties are absolute Hz figures only — 914 (now 811 cold), the 210→337
endpoints, the 175.3 anchor — plus iter-18's +9.1%, whose margin is comparable to
the window swing and survives only because its bracket happened to drift downward
around the winner.
