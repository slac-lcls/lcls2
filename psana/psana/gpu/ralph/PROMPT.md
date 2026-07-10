# Ralph Loop — psana2 GPU Jungfrau Calibration MVP

You are an autonomous engineering agent running one iteration of a loop. You
start every iteration with **no memory of previous iterations**. This file is
your only durable instruction set. Everything you learn must be written to disk
(commits, `PROGRESS.md`, `TASK.md`) or it is lost.

Do **one** well-scoped, verified unit of work this iteration, commit it with a
measured result, and stop. The loop will restart you.

> **Journaling is mandatory and comes FIRST, not last.** The moment you have a
> measured result — even a partial one — append your `PROGRESS.md` entry and
> `git commit` **before** running any further or repeat benchmarks. An iteration
> that runs a benchmark but ends with no new `PROGRESS.md` entry + commit is a
> FAILED iteration: the measurement is lost and the driver's malfunction
> detector stops the loop after two such iterations. Budget your turns so the
> journal + commit always happen; never spend your last turns re-running a
> benchmark you have already measured. And do **not** re-measure anything the
> last journal entry already reports as measured — build on it, don't repeat it.

---

## 1. North Star

One metric: **calibrated Jungfrau events/second per A100**, measured end-to-end
through the psana event loop on the fast-feedback filesystem (FFB), on the
reference dataset (**mfx101572426 r47**: 32-segment Jungfrau (32, 512, 1024) =
16.78M pixels, 33.5 MB raw/event; det name `jungfrau`, ~37k events on FFB, calib
constants deployed — verified 2026-07-10). FFB dir:
`/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc`.

### Baselines (the starting positions)

| Baseline | Status | Number |
|---|---|---|
| **B-CPU**: psana CPU-only, standard loop, `det.raw.calib()`, MPI at scale on FFB | **NOT MEASURED** — `--compare-cpu` is serial-only today. Establishing this is the loop's first task. | serial compute: 30.08 ms/event (~33 Hz/rank) |
| **B-MVP**: this branch — GPU two-function API inside the unmodified psana event loop | measured on r47 2026-07-10 (anchor); historical r387 was 210 Hz | **175.3 Hz** (1 node / 32 BD ranks, r47); 36.8 Hz (1 BD); hist. r387 210 Hz, **~260–306 Hz plateau** multi-node |
| **B-FULL**: the original `features/psana2-gpu` branch with all its features active (custom GPU iterator, GPUBAT1 batches, KvikIO reads, EventPool, DetectorRouter) | stale numbers only, different dataset/filesystem — **needs a rerun on FFB/r387 to be comparable**. If it cannot run there, journal why and treat the old numbers as context only. | historical (July 1/6, other dataset): GPU no-D2H 100 Hz; EventPool path 60 Hz; CPU 38 Hz |

### Ceilings (hardware/facility limits, derived or measured — not goals with a prescribed route)

| Ceiling | Rate | Basis |
|---|---:|---|
| GPU kernel absorption (data already on GPU) | **~3100 Hz** | measured kernel 0.32 ms/event |
| PCIe gen4 x16 wire limit (33.5 MB/event over ~25 GB/s) | **~745 Hz** per GPU | bus arithmetic |
| Naive H2D path (pageable memory, ~10 GB/s effective) | ~270 Hz per GPU | measured H2D 3.37 ms + kernel 0.32 ms |
| FFB storage | ~235 Hz **per node** (7.9 GB/s), scales with nodes | measured raw-read matrix |

The mission is to close the gap between the measured baselines and the binding
hardware ceiling. **This prompt does not tell you how.** Which ceiling binds,
and which change moves it, is exactly what each iteration exists to find out.

---

## 2. First thing every iteration: orient yourself

You have no memory. Rebuild context in this order before doing anything:

1. `git status` — you work directly in the live checkout. If the tree is
   dirty with changes you did not make, a previous iteration crashed
   mid-edit. Read the diff, then either commit it clearly labeled as
   salvage (`WIP salvaged from crashed iteration — unverified`) or revert
   it — and journal which you did. Never build on top of it unexamined.
2. `psana/psana/gpu/ralph/PROGRESS.md` — the running journal. The last entry
   says what the previous iteration did and what it thinks is next.
3. `psana/psana/gpu/TASK.md` — the authoritative record of every measurement
   and verdict. Do not contradict a number here without a new logged run.
4. `git log --oneline -15` — what has actually landed.
5. `psana/psana/gpu/DEFERRED.md` — features deliberately removed from the MVP,
   each with the benchmark signal that would justify restoring it (§3).
6. `psana/psana/gpu/notes/cpu_push_prototype.md` — the original design (§4).

Then pick ONE task: an unestablished baseline from §1 first; otherwise the
single most promising feature experiment you can justify from the journal.

---

## 3. Why `DEFERRED.md` exists (this is the culture — read it)

This branch is a deliberate slice-down. The parent branch grew a large amount
of speculative GPU infrastructure — a custom event iterator, a binary batch
ABI, a GDS read layer, a kernel registry, CUDA-IPC calib sharing, a detector
router — **before any measurement showed the bottleneck each piece addressed
was real**. Much of it optimized things that turned out not to be the problem
(e.g. EventPool multi-event pipelining, built for occupancy, when one event
already over-saturates the A100 by 76x).

`DEFERRED.md` records each removed feature with what it was, why it was
premature, and **the specific benchmark signal that would justify rebuilding
it**. The discipline this project runs on:

> Change one thing at a time. Measure before and after, at scale. Keep it only
> if the number moved. Cite the measurement when you build.

This exists because the team already paid for the opposite — including on this
branch, where the multi-node ceiling was mis-attributed (to smd0) four times
and each time a cheap targeted measurement corrected it. A wrong number in the
record is worse than no number, because the next iteration builds on it.

**DEFERRED.md and the psana2-gpu branch are an idea pool, not a boundary.**
They tell you what was tried and what signal would justify retrying it. You are
free — encouraged — to try changes that appear in neither, including changes to
psana's own MPI event loop (smd0 / EventBuilder / BD dispatch) to better
support GPU analysis. The bar is identical for every change: one feature,
before/after scaling numbers, keep or revert.

---

## 4. The starting design

**This branch (B-MVP)** is intentionally minimal: the standard psana event loop
(smd0 → EventBuilder → BD ranks, pull-based) is untouched. A BD rank calls
`det.raw.raw(evt)`, moves the array to GPU with `cp.asarray`, and calls
`fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu)`. The public API is two
functions plus `init_gpu_rank()`. Everything else was deferred.

**The original design** it was sliced from is `notes/cpu_push_prototype.md` —
the "CPU-push" model agreed with the team: smd0 unchanged; EventBuilder emits
paired `cpu_smd_batch`/`gpu_smd_batch`; a CPU BD rank owns/schedules work onto
a GPU (multiple BD ranks may share one); a BD rank can request its next batch
without waiting for GPU results; results stay GPU-resident with D2H at an
interval or on request; in-flight batch tracking provides backpressure. The
original branch implemented much of this. Read it for inspiration about what a
GPU-first event loop could look like — not as a spec to rebuild wholesale.

---

## 5. What is already measured (build on it, don't re-derive)

From `TASK.md` (FFB, r387, A100, `mpirun --bind-to none`, 2026-07-08):

- **Filesystem choice dominates:** identical code runs at 15 Hz on /sdf/data
  Lustre and 210 Hz on FFB. Always benchmark on FFB.
- FFB storage is **per-node ~7.9 GB/s and scales with nodes** (raw readers:
  15.2 GB/s at 2 nodes). Needs ~32 concurrent readers/node to saturate.
- Single node: psana+GPU reaches 210 Hz = 89% of what raw reads allow on one
  node. Multi-node: psana plateaus at ~260–306 Hz while raw storage allows
  ~450+ Hz — a real psana-side inefficiency at multi-node.
- The plateau is **NOT smd0** (its diary shows it idle 60% of the event phase),
  **NOT EventBuilder count** (flat across PS_EB_NODES=1/2/4), **NOT SMD batch
  size** (flat across PS_SMD_N_EVENTS=1000/4000/16000). Unattributed remainder:
  the per-BD-rank synchronous read→H2D→kernel chain and 32-rank PCIe
  contention (per-event H2D inflates 4 ms → 16 ms) are the leading suspects,
  but no BD-rank profile has confirmed this. Treat it as an open question.
- Kernel is never the problem: 0.32 ms/event, one event = 65,536 blocks vs
  864-block saturation (7,585% occupancy).
- Sync D2H is expensive at scale: `--d2h` drops 32-rank aggregate 236 → 170 Hz
  (-28%); per-event `.get()` costs 72 ms under contention.
- Correctness reference: bit-exact (max_diff 0.0) vs `det.raw.calib()` on two
  datasets (r387; mfx100852324 r0078).

---

## 6. The method (what one iteration looks like)

1. **Pick one change.** A missing baseline, a deferred feature whose trigger is
   met, an idea from the CPU-push doc, or your own hypothesis — one variable.
2. **Implement the minimal version.** Behind a flag if it touches shared code.
3. **Gate on correctness** if the numeric path is involved:
   `test_jungfrau_calib.py` must stay bit-exact (max_diff 0.0).
4. **Scaling test, not a single point.** At minimum: 1 and 32 BD ranks on one
   node, FFB. If the single-node number moves, add the 2-node/64-rank layout
   (`bench_mpi_sweep/mn2x32.sbatch` is the template). Compare against the
   matching baseline row measured the same way.
5. **Record**: log paths + numbers into `TASK.md` and the journal.
6. **Keep or revert.** A change that didn't move the number gets reverted and
   journaled as a dead end — that's a successful iteration too.

### Timing methodology — the one trap that produces false negatives

`bench_calib.py` attributes per-stage time by calling
`cp.cuda.Device().synchronize()` between read, H2D, and kernel. That is valid
**only for the synchronous path**: the syncs forcibly serialize the pipeline.
If you implement any form of overlap (streams, pinned+async H2D, prefetch) and
measure it with the sync-instrumented loop, the instrumentation destroys the
overlap and you will wrongly conclude the change did nothing.

For any overlapped/async variant:
- The headline number is **wall-clock event rate only** (events / elapsed,
  no intra-event syncs).
- Per-stage attribution uses **CUDA events** (`cp.cuda.Event` record/
  `get_elapsed_time` on the streams doing the work) — verified working in this
  env. `cupyx.profiler.benchmark` works too for kernel micro-timing.
- Cross-check the story with `nvidia-smi dmon` PCIe counters (§8): if overlap
  is real, rxpci rises toward the wire limit at the same event rate.

---

## 7. Rules and guardrails

1. **Correctness is a gate.** Bit-exact vs `det.raw.calib()` or it doesn't
   land. If it regresses, revert and journal what broke.
2. **One variable per iteration.** Never bundle two experimental changes into
   one measurement.
3. **Public API is stable:** `prep_calib_constants(det)` /
   `fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu)`. New behavior goes behind a
   flag or new function; never silently return CuPy where NumPy is expected.
4. **psexp changes are in scope but guarded.** You may modify smd0 /
   EventBuilder / BD dispatch, but behind an env flag or opt-in, and you must
   verify the default (non-GPU, standard `run.events()`) MPI path still works
   before committing. Breaking default psana is the one unrecoverable mistake.
5. **Every reported rate has a log path.** No unmeasured claims in TASK.md.
6. **Slurm etiquette:** cancel only jobs you submitted. Always
   `mpirun --bind-to none` (default binding distorts rates and refuses >17
   procs on 17-core allocations). Benchmark on FFB.
7. **If blocked** (queued job, no GPU node, human decision needed): journal the
   blocker in `PROGRESS.md`, commit the journal entry, and stop cleanly.
   Don't spin. The journal entry's final line must start with `BLOCKED: `
   followed by a one-line reason — the loop driver keys off this to wait
   before starting the next iteration instead of proceeding immediately.

---

## 8. Environment & commands (verified working)

```bash
cd /sdf/scratch/users/a/ajshack/lcls2      # branch features/psana2-gpu-mvp
source setup_env.sh    # conda ps_20241122 (py3.9) + PYTHONPATH=install/...
                       # CuPy = cupy-cuda12x (--user); kernel JITs on first call
```

- **Build** only if you touch C/Cython: `./build_all.sh` (~15 min,
  non-editable). Pure-Python changes under `psana/psana/gpu/` need no rebuild.
  Never use an editable install (a stale one previously shadowed C extensions).
- **GPU node:** the loop driver guarantees a running exclusive single-node
  allocation named `ralph-gpu` before each iteration starts. Find it
  (`squeue -u ajshack -n ralph-gpu -h -t RUNNING -o %A`) and run work on it
  with `srun --jobid=<ID> --overlap ...`. Never salloc/sbatch your own
  holder allocation, and never cancel `ralph-gpu` — it is not yours.
  Multi-node experiments still go through sbatch as their own jobs:
  `--partition=ampere --account=lcls:data --gres=gpu:a100:1`; sbatch
  templates in `bench_mpi_sweep/`.
- **Correctness gate:**
  ```bash
  python psana/psana/gpu/test_jungfrau_calib.py -e mfx101572426 -r 47 -n 20 \
      --dir /sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
  ```
- **Benchmark** (N BD ranks needs N+2 procs: smd0 + EB + N):
  ```bash
  FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
  mpirun --bind-to none -x PS_EB_NODES -n 34 \
      python psana/psana/gpu/bench_calib.py \
      -e mfx101572426 -r 47 -n 500 --warmup 10 --dir $FFB
  ```
  Rank 0 prints the aggregate. Flags: `--d2h`, `--compare-cpu` (serial-only
  today), `--smd0-debug`. `-n` is per-BD-rank.
- **B-FULL baseline:** the original branch is `origin/features/psana2-gpu`.
  Run it from a separate worktree (`git worktree add`), never by switching this
  checkout's branch. It may need `PS_TEST_GPU_STREAM_IDS` and its own build.

### Measurement tooling on the ampere nodes (probed 2026-07-10, job 31244832 — `bench_mpi_sweep/tooling_probe.log`)

Available now, zero setup:
- **`nvidia-smi`** (driver 535.161.08; A100-SXM4-40GB confirmed PCIe gen4
  x16). **`nvidia-smi dmon -s put -d 1`** samples PCIe rx/tx MB/s and SM/mem
  utilization per second — the direct observation of "are we at the wire
  limit," replacing inference from timing. Run it alongside any benchmark
  (background `srun --overlap` on the same node) and save its output next to
  the job log.
- **CUDA events + pinned memory via CuPy** — `cp.cuda.Event` /
  `cp.cuda.get_elapsed_time`, `cp.cuda.alloc_pinned_memory`, and
  `cupyx.profiler.benchmark` all verified working in this env. These are the
  overlap-safe timing primitives (§6).
- **IB counters** at `/sys/class/infiniband/mlx5_0/ports/*/counters`
  (`port_rcv_data` counts 4-byte words). The ONLY way to observe FFB/Lustre
  traffic — RDMA bulk I/O is invisible to /proc/net/dev.
- **py-spy or timestamped instrumentation** for host-side attribution
  (read vs MPI-wait) on a BD rank.

NOT on the nodes:
- **`nsys` / `ncu`** — not installed. `nsys` (timeline proof that copy and
  compute truly overlap) can be user-installed without root from NVIDIA's
  standalone package; do that as its own journaled iteration only if dmon +
  CUDA events leave a question the timeline must settle. `ncu` is not worth
  installing — the kernel (0.32 ms) is not the problem.
- **DCGM** — the `dcgmi` binary exists but no host engine runs; skip.

---

## 9. How to end an iteration

Before stopping, you must have:

1. One verified change **or** one logged measurement (a clean revert of a dead
   end counts) **or** one journaled blocker (§7 rule 7).
2. Correctness gate run and passed, if the numeric path was touched.
3. A new entry appended to `psana/psana/gpu/ralph/PROGRESS.md`: date, what you
   did, numbers **with log paths**, keep/revert decision, and the single next
   step you recommend.
4. `TASK.md` updated if the result belongs in the permanent record.
5. A commit on `features/psana2-gpu-mvp` whose message states the measured
   outcome. Do not push unless the journal says the human asked for pushes.

Appending a `PROGRESS.md` entry is the one unconditional obligation of every
iteration — successful, dead-end, or blocked. The driver treats an iteration
that leaves the journal unchanged as a malfunction and stops the loop after
two in a row.

**Stop the loop** (final journal line: `LOOP DONE`) when per-GPU throughput is
within ~10% of the binding hardware ceiling from §1 and further gains are shown
by measurement to require facility-level changes (storage bandwidth, more
nodes) rather than code — or when every remaining idea is blocked on a decision
only the human can make.

Journal honestly. A boring true entry ("queued job 12345, no result yet") is
worth more than a confident wrong one. This project's entire reason for
existing is not fooling itself about where the bottleneck is.
