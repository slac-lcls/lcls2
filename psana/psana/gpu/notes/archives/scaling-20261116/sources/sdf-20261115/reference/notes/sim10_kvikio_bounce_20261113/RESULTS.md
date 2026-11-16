# Bounce-buffer result — job 44870463

Recorded November 14, 2026. Job **44870463** completed on **sdfampere018** on
November 13, 2026, **17:14:55–17:49:38 PST** (34m 43s), exit **0:0**.
All **31 native GPU tests, 4 traced pilots, 8 untraced full runs, and the final
audit passed**. Owned staged inputs were removed. Re-running the frozen
auditor reproduced the saved audit JSON byte for byte.

## Throughput and carry-forward recommendation

| BDs | Workers/BD | Batch | Bounce MiB | Mean events/s | Observed min–max |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 50 | 16 | 654.16 | 640.34–667.97 |
| 1 | 8 | 50 | 64 | 638.67 | 628.16–649.18 |
| 4 | 4 | 200 | 16 | 1619.83 | 1605.06–1634.59 |
| 4 | 4 | 200 | 64 | 1640.49 | 1605.25–1675.73 |

Means use two full repeats per configuration. Comparing 64 MiB against its
same-allocation 16 MiB control gives **−2.37% at 1 BD** and **+1.28% at 4 BDs**.
Observed ranges overlap, and the direction differs between profiles. These
small, two-repeat differences do **not establish a meaningful, consistent
benefit or equivalence**. Observed extrema are not confidence intervals.

**Carry forward the 16 MiB bounce-buffer control, with 1 MiB tasks.** There is
no demonstrated reason to promote 64 MiB for this fixture. This is a cautious
screening recommendation, not an independently reviewed optimality claim.

All four pilots show unchanged, unsplit POSIX read sizes of **524724, 629664,
734604 bytes**, below both buffer sizes. Pilots are excluded from throughput
comparisons. Their syscall traces do not directly measure native task counts,
H2D timings, or pinned-buffer allocation counts; unchanged POSIX reads alone
do not prove identical H2D behavior.

## Controls and acceptance

This is one allocation cohort: one **A100-SXM4-40GB**, pinned PR298 runtime
`67c327a148a08b2aaf972e67e1e02e367c2b298f`, and reviewed harness
`de7513219568bc0acee99ffe3d3b61287e088f77`. KvikIO **26.10 CPU fallback is ON**
(not GDS), auto-direct is OFF, task size is 1 MiB, and each BD uses one CUDA
stream and a 6 GiB input budget. Bulk target is 512 KiB; SMD batch is 1000.
Effective settings, installed KvikIO identities, and native compatibility
receipts passed the frozen audit.

Each full run globally partitions the same **100000 events**, reads **500000
datagrams / 335820000000 bytes**, and delivers **2000 callbacks at batch 50**
or **500 at batch 200**. All ranks drain; no pressure drains, unexpected batch
splitting, or psana CPU payload reads were observed. GPU memory headroom and
disjoint physical-core affinity passed. The one-/four-BD cases use 17/34
physical cores respectively; Slurm requested 36 cores in the exclusive
allocation.

After DataSource setup, all cases passed zero initial payload page residency
and physical NIC acceptance. Across the eight full runs, idle-adjusted NIC
RX/payload was **1.07909–1.11767**. This corroborates **network-served Weka FFB
input**, not cold backend storage or exclusive attribution of NIC traffic.

Timing brackets the event loop and final drain/barrier, excluding input
staging, setup, cache conditioning, and subsequent audit. This remains a
**staging-only GpuTask** with descriptor validation and metadata recording:
no application expansion/calibration/cube kernels or per-event publication
D2H. It is not whole-job or full-cube throughput.

## Host-memory evidence

The source-based estimate of one active reusable POSIX bounce buffer per
KvikIO worker gives the following aggregate demand across BDs:

| Profile | 16 MiB buffers | 64 MiB buffers |
|---|---:|---:|
| 1 BD × 8 workers | 128 MiB | 512 MiB |
| 4 BDs × 4 workers | 256 MiB | 1024 MiB |

All host-memory gates passed, including preloop available memory above the
aggregate estimate plus 8 GiB. These are **planning estimates**, not measured
pinned allocation counts or bounds on all host memory. The archive retains
162 per-rank snapshots across startup, before-loop, and after-loop sampling
in [host-memory.csv](results/host-memory.csv), with byte units for VmRSS,
VmHWM, VmLck, VmPin, and host MemAvailable. VmHWM is a process-lifetime peak;
these kernel counters do not isolate KvikIO allocations or account reliably
for all CUDA pinned memory. Do not interpret their differences as exact
bounce-buffer usage or sum rank peaks as a simultaneous host peak.

## Evidence and review status

- [runs.csv](results/runs.csv): all 12 cases, ordered and labeled pilot/full,
  with rates, counts, resource demand, NIC ratios, and raw-file hashes.
- [summary.csv](results/summary.csv): four full-run configurations with two
  repeats each, mean/min/max rates, and pooled throughput.
- [comparisons.json](results/comparisons.json): same-job 64/16 MiB comparisons.
- [validation.json](results/validation.json): byte-identical frozen re-audit,
  auditor/interpreter paths, recording time, and final accounting command.
- [source-manifest.json](results/source-manifest.json): SHA256 identities for
  original result artifacts, identifying which files are preserved locally.
- `results/`: preserved audit, hardware/provenance, four pilot syscall
  summaries, completion/cleanup, GPU-test, and final accounting receipts.

Raw per-case timestamps, traces, and logs remain at
`/sdf/scratch/users/m/monarin/gpu-validation/sim10-kvikio-bounce-20261113/attempt01`.
Hashes establish identity but do not preserve raw files against scratch
purge. The compact archive preserves these conclusions, not every input
needed to repeat the full frozen audit.

Preparation review passed with no findings on the harness SHA above. **No
independent review of these result interpretations has been performed.**
The automated audit and independent preparation review are distinct from
independent result review.

This cohort is linked from the [single-node screening
package](../sim10_single_node_scaling_20261113/README.md), but is not pooled
with earlier allocations or merged into its existing four-cohort figures.
The next planned screen is **1/2/4 GPUs with four total BDs and 100k global
events fixed**, retaining one stream/BD and matched controls. Its harness
adaptation, placement/capacity acceptance, and prelaunch review remain to be
done. Multi-stream/forced-split tuning stays deferred under #308. Recording
this result changes no runtime and submits no new job.
