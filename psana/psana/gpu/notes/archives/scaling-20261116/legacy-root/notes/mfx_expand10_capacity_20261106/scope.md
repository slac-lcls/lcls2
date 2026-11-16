# Dense cube capacity and phase profiling

Authorized: configurable persistent-state budget and bin count, bounded per-bin
final export, 30/100/200-bin correctness and memory/timing checks on the existing
1000-event fixture, one BD/two streams/one A100, independent Claude review (max
three rounds). No new XTC, DB changes, large event dataset or distributed/cache
policy. Preserve prior reviewed source and evidence. Base 60b69ff08.

Create a derived Cube using unchanged Stage3A accumulation kernels/identity/bank
ownership and unchanged Stage2 expansion/calibration. Replace fixed constructor
limits with explicit budgets and replace whole-cube host export with per-bin
files. Fail closed on incomplete iteration/export; export success manifest only
after every file completes. Tests cover budget rejection, third stream, failures,
run isolation and bounded write buffers, plus all final pixels versus independent
CPU-v3 calibration with the existing float32 rounding bound.

Use node-local scratch for temporary dense outputs (max ~25 GiB per case) and a
~3.13 GiB CPU-read seed cache. CPU reference/calibration is evaluated one bin at
a time; no whole reference cube. Shared scratch has ~19 GiB free. Copy reports,
hashes and representative bins only; remove only this job's temporary files after
validation and preservation. Request enough local space before any writes.

Record actual device/free memory snapshots, CuPy pool samples and host high-water
memory; measurements are sampled and driver usage includes all device users.
Time expansion/accumulation kernels and calibration call with CUDA events without
per-stage synchronization; retain event owners until completion. Record reader
submission/wait counters across resets and end-to-end small-run/export wall time.
Exclude first invocation per stream from separately labelled warm intervals;
compile/setup and scalar checks affect overall timing. No 35kHz, GDS or sustained
throughput claim. KvikIO compatibility ON, no cold-cache claim, source one step.
