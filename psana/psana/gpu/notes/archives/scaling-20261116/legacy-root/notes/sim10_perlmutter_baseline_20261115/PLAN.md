# Revised Perlmutter Sim10 bootstrap02r2

The active proposal removes cube payload export and CPU cube reference work at
the user's request. It checks each BD's actual GPU cube shape, dtype, bytes,
counts and content after GPU drain, outside the processing timer. Every pixel
must be finite, populated bins must contain a nonzero pixel, and empty bins must
remain zero. Individual zero pixels are valid. FP32 shape: (100,32,512,1024).
The GPU predicate workspace is bounded to 1MiB chunks; only 100 counts and scalar
flags reach the host. No full-bin D2H, MPI cube merge, .npy output, pixel hashes,
or reference-cost probe. Four BD cubes remain distributed. Basic content checks
do not prove numerical-reference agreement. Small JSON receipts/logs remain.

Historical bootstrap01/job61373306 failed its first ECC step at 600 seconds;
all native GPU tests and descriptor guards passed on four GPUs. The export
directory existed but was empty, placing the stall after event processing.
The exact stalled operation is unproven. This revision removes that phase;
it is not evidence that its root cause has been diagnosed or fixed.
Preserve bootstrap01 and its receipts. No staging repeats completed there.

## Reuse and fixed configuration

Transfer verification: 315 files / 610186592 bytes. Exact PR298 commit
67c327a148a08b2aaf972e67e1e02e367c2b298f and existing install; 727 installed
and 86 test-source hashes checked. Reuse job61359629 qualification. Its sole
100k staging observation (61.3367 s, 5.0990 logical GiB/s) is historical context,
not a repeated baseline. Recheck pins at preparation and runtime, without
rebuilding or copying data. Use the qualified count2 / 1MiB five-stream view.

Four KvikIO workers/BD, one CUDA stream/BD, task 1MiB, bounce 16MiB/worker,
bulk 512KiB, framework input ceiling 6GiB/BD, SMD1000, compatibility ON,
auto-direct OFF, PS_JF_SHARE_DERIVED=0, PS_GEO_SHARE=0 and frozen constants.
No parameter sweep or batch rescue. The 4BD/1GPU bridge is staging-only.

## Bounded proposal

One exclusive Perlmutter GPU node, four GPUs, 128 logical CPU slots, all node
RAM, lcls_g/premium, 01:10:00. Maximum reservation: 1.167 node-hours or 4.667
allocated GPU-hours before premium accounting conversion, not an Iris charge
prediction. Stop at the first failed gate/timeout; no retry or automatic next job.

| Work | Stop budget |
|---|---:|
| Runtime identity, hardware, 3-/6-rank affinity probes | 300 s |
| 31 native GPU tests plus 10 descriptor guards on each GPU | 4 × 120 s |
| ECC shape/content checks: 1BD/1GPU/batch50/201 and 4BD/4GPU/batch200/1003 | 2 × 300 s |
| Staging: three 100k repeats each of 1BD/1GPU/batch50, 4BD/4GPU/batch200, 4BD/1GPU/batch200 | 9 × 240 s |
| Independent compact-receipt audits | 11 × 20 s |
| Startup/safety reserve | 440 s |
| Total | 4200 s |

Rotate the staging profile order across repeats. All cases share one allocated
node. Case ceilings are unmeasured stop bounds, not performance predictions.
ECC checks include full and tail batches. Physical GPU memory must stay >=4GiB
free, including verification; record framework and sampled physical memory
separately. Keep device tests, lifetime/drain, error publication, exact event/
byte/range, actual-open identity, weakref and single-stream gates.

Use srun --mpi=cray_shasta with 3/6 ranks, 120/tasks CPU slots, --cpu-bind=none,
--gpu-bind=none, numactl --interleave=all. Probe real allowed CPU masks and bind
3,3,7 or 3,3,7,7,7,7 distinct physical cores. Preserve the first BD CPU/GPU
placement across profiles. Helpers see no CUDA; explicit allocated UUIDs are
checked against physical PCI IDs. This is controlled, not optimized placement.

Recondition only qualified owner-controlled input files after setup, require
zero resident pages before processing, retain available Lustre counters, and
label backend traffic unverified. Check concurrent count2 users before submit;
the task's cooperative lock does not constrain old campaigns. Reserve 2GiB for
reports/logs. Refresh Iris balances immediately before authorized submission.

The immutable bootstrap02r2 bundle and command must be presented after review,
with no submit until approval. See SUBMISSION.md. After execution/result review,
size the full same-node S/E/EC/ECC campaign from measured processing/setup/check
times. It must repeat staging on its own node, keep both target configurations
and four workers fixed, and apply this same no-export shape/content policy.
Full-stage throughput remains unmeasured; no numerical-reference claim is made.

Per-rank progress histories add monotonic markers after processing, post-loop
gates, cube verification, gather and shared-memory close. All diagnostic writes
stay outside the unchanged processing timer. The first post-loop marker follows
the existing end barrier and timing allreduce; a stall inside those collectives
cannot be distinguished from processing by that marker alone. The 300 s limit
remains an unmeasured stop bound, not a claim that the old stall is resolved.
