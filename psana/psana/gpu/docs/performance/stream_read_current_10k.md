# Current-code bulk off/on: 10,000 events, cold and warm

Measured on 2026-09-25, using the current worktree with all three CPU-overhead
optimizations: indexed slot selection, per-file pending ownership counts, and
direct group submission with shared descriptor validation. Bulk on was slower
in both rounds and both cache conditions. Fewer read submissions did not yield
an end-to-end throughput improvement for this workload.

## Results

Each case ran twice on the same GPU and node. Round two reversed both variant
and cache order. Rates below are 10,000 divided by median loop time; setup,
cache preparation, the 200-event correctness warmup, and teardown are excluded.

| Cache | Bulk off events/s | Bulk on events/s | Off median s | On median s | Bulk-on time change |
|---|---:|---:|---:|---:|---:|
| Cold | 134.15 | 127.26 | 74.5416 | 78.5812 | +5.42% |
| Warm | 304.32 | 245.88 | 32.8598 | 40.6703 | +23.77% |

| Bulk | Cache | Round 1 events/s | Round 2 events/s | Reads per sample |
|---|---|---:|---:|---:|
| Off | Cold | 133.02 | 135.31 | 60,000 |
| On | Cold | 126.67 | 127.85 | 50,182 |
| Off | Warm | 299.53 | 309.28 | 60,000 |
| On | Warm | 244.87 | 246.90 | 50,182 |

Both modes read exactly 335,669,114,744 payload bytes per sample. Bulk on removes
9,818 requests (16.36%). The independently counted SMD extents predict 50,000
large Jungfrau requests plus 182 coalesced feespec requests; the count respects
100-event boundaries, file-offset gaps, transition fences, and the 4 MiB target.
The same counting method reproduces the earlier 1,000-event total of 5,019.

This compares bulk on with off in the current runtime. It does not isolate the
individual optimizations or establish the cause of the remaining overhead.
Two rounds give a repeatability check, not a statistical confidence interval.

## Runtime and workload

- Job **39084570**, `COMPLETED`, exit `0:0`, elapsed **27m46s**.
- Node `sdfampere011`; one A100,
  UUID `GPU-154e7adf-a139-1fb1-8ac7-73e6a5362bcc`, PCI bus `0000:01:00.0`.
- Three MPI ranks: one SMD0, one EB, one BD. Identical 48-CPU affinity for all
  samples; 700 GiB host-memory reservation. The node was not exclusive.
- Dataset `mfx101210926/r0387`, physical streams 0, 5–9; Jungfrau plus feespec,
  with the existing benchmark-only exclusive GPU routing of shared stream 0.
- 10,000 measured events, batch size 100, pool depth 1, 8 GiB GPU budget,
  `gpu_d2h_chunk_size=0`, `PS_SMD_N_EVENTS=1000`.
- 4 MiB bulk target and KvikIO task size, eight KvikIO workers;
  `KVIKIO_COMPAT_MODE=ON` (CPU fallback, not GDS).
- CuPy 13.6.0, KvikIO 24.08.02. Controls collect existing reader counters;
  no Python profiling or native-operation tracing in these measurements.
- Source HEAD `c3357e622c4aa4e5d899757b2c20045a518a71f3` plus current dirty
  changes, including direct group submission. Both modes use one frozen runtime.
  Its executable Python sources matched the current checkout and the previously
  device-tested candidate before submission. No production runtime changed for
  this comparison.

Frozen runtime SHA-256:

| File | SHA-256 |
|---|---|
| `gpu_kvikio_read.py` | `9a43193f6720b896e6f3b1c3522891054a4dceac61c2d6c46fafb21f40efa13c` |
| `gpu_read_plan.py` | `72de7118c3c3bf3dbfaaaacb4bcc1dd51e0845d1cbf951c91e2aaae79017a441` |

## Acceptance and cache evidence

All eight controls passed event count, timestamp hash, feespec GPU sum hash,
exact payload and read counts, runtime/GPU identity, and CPU-affinity checks.
Every sample first validated 200 full feespec arrays and three Jungfrau raw
and calibrated reference samples. Full-array D2H is outside measured controls.
Frozen runtime, harness, constants, and reference hashes were checked, and Weka
SSD-only placement passed before and after the campaign.

Cold means evicted node page cache; Weka server caches were not flushed.
The largest pre-run per-file residency was **0.0005301%**, below the 1% gate.
Physical NIC RX was **546.77–549.29 GB** per cold sample, exceeding the required
98% of payload. NIC counters also include filesystem overhead and other traffic;
they are not an exact measure of useful payload bytes.

Warm prefixes were read by a separate `numactl --interleave=all` helper, without
changing the timed workers' NUMA policy. Every file was checked after preparation
and after timing. Minimum residency was **99.99676%**, above the 99% gate.
Warm measured NIC RX was **0.186–0.263 GB** per sample.

Harness changes add `acceptance.py --study current --events 10000`, an eight-case
current-only summary, and NUMA-interleaved preparation for measured prefixes
totaling at least 64 GiB. Existing studies retain their 1,000-event defaults.
All **24 harness tests passed**, including event-count/read-count rejection,
balanced order, incomplete-result rejection, exact prefix reads, and residency
gates. `git diff --check` passed. The identical runtime had already passed the
20 A100 tests documented in `stream_read_direct_group.md`.

## Reproduction and artifacts

Submit the saved launcher:

```sh
sbatch /sdf/scratch/users/m/monarin/gpu-validation/stream-read-current-10k-20260925/run.sbatch
```

Campaign root:
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-current-10k-20260925`

It contains `current/python`, `current.patch`, `builds.json`, the frozen harness,
`run.sbatch`, and `references/` with measured/warmup manifests, CPU references,
and independently predicted request counts. Full runtime file hashes are in
`builds.json`; native dependencies remain in the recorded shared installation.

Accepted results:
`job-39084570/{summary.md,summary.json,results.json,provenance.json}`,
with each sample's log and GPU monitor CSV. The scheduler log is
`job-39084570.log`. The complete summary is written only after final provenance
and storage-placement verification.
