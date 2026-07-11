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
