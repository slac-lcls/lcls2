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
