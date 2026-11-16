# Dense cube capacity: 30, 100 and 200 bins

This stage extends the small GpuTask prototype on the existing 1000-event fixture.
One process, two CUDA streams and one A100; no new XTC, calibration DB or framework
code. The new [Cube](dense_cube.py) derives from the previously reviewed Stage3A
Cube, retaining its identity checks, stream-owned banks, array lifetimes and
count/sum kernels. The Stage2 expansion/calibration implementations are unchanged.

`nbins`, `active_bins`, and `max_state_bytes` are explicit constructor arguments.
The campaign uses 30/100/200 total bins, with the last bin intentionally empty:
`bin = global_event_ordinal % (nbins - 1)`. This still tests synthetic binning,
not real user-defined small-detector coordinates or multi-step source data.
The state cap is 27 GiB for each case. Before allocation the runner checks required
state plus an 8 GiB non-state reserve against actual currently free device memory.
This is a single-process preflight, not a device-wide reservation service or a
change to psana's accounting for task arrays.

## Bounded host export and verification

`finish()` intentionally refuses a whole-cube host allocation. Call
`finish_to(new_directory)` after fully consuming `run.events()`. It joins stream
work, transfers and merges one bin at a time, writes/fsyncs each bin's `.npy`, and
only then creates a success manifest. Its reusable large host buffers are one
float64 image (128 MiB) and one float32 transfer buffer (64 MiB), independent of
bin count. Counts and file metadata are small. Success is idempotent for the same
destination. Incomplete runs cannot export; failures after export begins leave no success
manifest and prohibit retry through the same task. `close()` joins before bank
references are released; CuPy can retain their allocations in its pool.

Every saved pixel is compared against independent CPU-v3 Jungfrau calibration of
CPU-read sampled seeds. The reference expands seeds with NumPy and accumulates
float64 sums plus absolute-value sums one bin at a time. The tolerance is the
same as Stage3A: `gamma_n * sum(abs(float32 calibrated inputs)) + 1e-5`, where
`gamma_n = n*u/(1-n*u)` and `u=2^-24`. All counts must match exactly and the last
bin must remain zero. Expected arrays do not come from GPU output. Comparison
error/bound temporaries are panel-sized.

The CPU seed cache is ~3.13 GiB. Full GPU outputs are written/read back on
node-local scratch (up to 25 GiB for 200 bins) because shared scratch is limited.
After successful validation, bins 0, active_bins-1, and the empty last bin are
copied to shared scratch and hash checked. Reports retain hashes/error metrics
for every bin. Other newly generated local output bins are removed after the
pass report is saved. The job's local workspace is removed only after all three
cases pass. Existing fixtures/results are never removed.

## Measurements and limitations

Each case runs in a fresh process. The runner records:

- Actual CUDA free/total device bytes and CuPy used/reserved pool bytes at
  calibration/accumulation/delivery/export boundaries. These are sampled high
  waters, not continuous exact device peaks. Device usage includes other users
  if any share the device. Process VmHWM records lifetime RSS including CPU
  verification; The pre-validation high water (called gpu_phase_memory in the report) covers
  baseline through completed export, excluding close() and CPU validation. Filesystem
  page cache is not process RSS.
- CUDA event intervals for the unchanged expansion, count and sum kernels,
  plus the full calibration call. Events are retained through completion, with
  no per-stage host synchronization. Sum/count/expansion kernels are compiled
  before iteration. Calibration intervals can include constant uploads, lazy
  initialization, allocations and host launch gaps. A separate warm summary
  excludes the first invocation of each stage on each stream; it is not a
  calibrated steady-state measurement or a warmup event window.
- KvikIO submission wall time, wait time, actual/requested/useful bytes and
  issue-to-observed-completion time, accumulated across framework statistic
  resets. Read timers can overlap and wait time excludes already completed I/O.
- Full native iteration wall time (including scalar error checks, profiling and
  memory sampling), export transfer/merge/write timings, and CPU verification
  time. Export writes include file fsync and are to local scratch, not production
  storage. CUDA interval sums must not be added to read timings to infer total
  throughput, because work overlaps.

KvikIO compatibility mode is ON. Fixture hashing/CPU seed preparation warms caches;
there is no cold-cache claim. No sustained 35kHz, GDS, MPI or scientific-binning
performance qualification is made. Profiling overhead is included. Old reviewed
source and reports remain unchanged.

## Run

From the checkout, choose a new shared-scratch campaign directory:

```bash
capacity_dir="$PWD/notes/mfx_expand10_capacity_20261106"
campaign=/sdf/scratch/users/m/monarin/gpu-validation/mfx-expand10/my-capacity-test
mkdir -p "$campaign/logs"
sbatch --output="$campaign/logs/%j.log" "$capacity_dir/run.sbatch" \
  "$capacity_dir" "$campaign/results"
```

The launcher requests one GPU and 24 GiB-class host allocation (`--mem=24G`),
checks at least 40 GiB local and 2 GiB shared scratch free, executes focused
host/device tests, prepares independent CPU seeds, then runs 30/100/200 cases.
Use [runner.py](runner.py) directly through Stage2 runtime.sh to change the
supported campaign batch size or explicit state/reserve budgets. Do not bypass
admission checks to force a larger case onto a shared device.

## Scheduler interruption

Job 43533593 on sdfampere034 passed the 30- and 100-bin cases, then was preempted
before the 200-bin case completed. The report did not checkpoint its phase. That incomplete report is not acceptance evidence.
The unchanged runner/kernel code is rerun for 200 bins only by
[resume200.sbatch](resume200.sbatch), job 43533968 on sdfampere015, using a fresh
CPU seed cache and output directory. Both jobs run the focused host/device tests.
The completed first-job cases are preserved. Same-node performance scaling
cannot be inferred from these single-run timings. The preempted job's local
workspace is not included in the successful-job cleanup claim.

## Results (November 6)

All three 1000-event cases passed exact counts and every output pixel within the
CPU float32-accumulation error bound. Both streams received 501/499 events. The
200-bin case used the recovery job; the interrupted attempt is excluded.

| Bins | GPU cube state GiB | Sampled device peak GiB | Through-export host peak GiB | Full-process host peak GiB | Iteration seconds | Export seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 3.75 | 6.07 | 1.85 | 3.45 | 2.77 | 5.61 |
| 100 | 12.50 | 14.82 | 1.83 | 3.43 | 2.47 | 18.58 |
| 200 | 25.00 | 27.32 | 1.85 | 3.45 | 2.75 | 43.61 |

Thus **200 bins fit this 40 GB A100 configuration**, with 25 GiB of banks and
27.32 GiB sampled device usage. RSS through export stayed near 1.85 GiB instead of
allocating a 25 GiB host cube; even CPU validation stayed below 3.45 GiB RSS.
Export's explicit reusable arrays were 192 MiB plus small count storage. These
results apply to one BD, two streams, batch size 7, float32 sums and this fixture.

Each case read 3,358,200,000 bytes through the GPU KvikIO reader (JF streams,
not every CPU metadata read). Warm CUDA interval sums over 156 calls were about
165 ms expansion, 700–706 ms calibration, 3.35–3.38 ms count updates and
147.7–147.8 ms image accumulation. Calibration is the largest measured GPU
stage interval; these overlapping intervals do not establish the critical path.
The overall loop includes I/O, Python work, scalar checks, initialization and
profiling. Do not interpret the lower 100-bin wall time as a scaling improvement.

Export is a separate scaling cost: it writes 3.75/12.5/25 GiB of float64 images.
At 200 bins, measured export components were 2.80 s D2H, 6.00 s CPU merge, and
32.65 s local write/fsync (43.61 s total including other work). These figures do
not characterize a production filesystem. A long run amortizes final export
over more events, but output-size and sink bandwidth still need budgeting.

All 5,536,481,280 final cube pixels across the three cases were compared. Maximum
error/bound ratios were 0.1781, 0.4659 and 0.6029, all below one.
[Compact verified evidence](validation-summary.json) links full scratch reports
and retained-bin hashes. Focused tests passed on both allocated nodes, including
third-stream rejection without state mutation and failure after the first bin
write. No sustained 35kHz conclusion follows from this campaign.

Independent Claude review passed implementation `df757a357` with no blocking
findings. [Review responses](review-01/response.md) record the documentation
clarifications and optional follow-ups. No executable source changed in closure.

Measurement contract: read_stats assumes the frozen framework resets counters
outside individual submission/completion calls. Its negative-delta check catches
only resets that decrease the observed counter; it is not a general reset epoch
tracker. All three accepted runs recorded 790 submissions/requests and identical
requested/completed/useful byte totals. Reusing this instrumentation with changed
reader reset semantics requires stronger epoch tracking.
