# Cold Weka FFB: uncompressed versus sim10 JF

The matched kernel-free read comparison completed on one A100, sdfampere002, job 43989739. Both inputs were read from **drpsrcf/ffb**, with zero client file-page residency immediately before each measured loop. There was no NVMe input staging and no warm-cache run. Input hashing/copying/eviction was outside timing.

| Input / repeat | Payload GiB | Time s | GiB/s | Events/s | Physical NIC RX GiB |
|---|---:|---:|---:|---:|---:|
| Uncompressed / first | 312.526 | 96.506 | 3.238 | 103.6 | 331.176 |
| Sim10 / first | 31.276 | 7.364 | 4.247 | 1357.9 | 33.160 |
| Sim10 / second | 31.276 | 6.697 | 4.670 | 1493.1 | 33.317 |
| Uncompressed / second | 312.526 | 75.082 | 4.162 | 133.2 | 332.566 |

The compressed cold input again reaches roughly 4 GiB/s per GPU with four KvikIO workers. The raw input reaches 3.24–4.16 GiB/s under the same settings. Two repeats show meaningful variability, especially for raw; they do not establish a stable maximum or an intrinsic benefit from smaller read sizes. This is a standalone I/O ceiling measurement, not a psana pipeline/decompression/cube benchmark. It supports using about 4 GiB/s as an achievable cold-read reference for the compressed input; psana's lower end-to-end rate remains a separate pipeline result.

Each loop completes exactly 10,000 events, 50,000 reads, and either 335,571,760,000 raw bytes or 33,582,000,000 sim10 bytes. All future lengths are checked; each FIFO slot is reused only after completion. Timing covers first submission through all completions and device synchronization. Post-timing GPU/CPU comparisons cover 105 byte windows over all 35 final slot buffers. These check sampled byte transport, not scientific correctness or every pixel.

## Storage and settings

- Raw: existing owned five JF prefixes in `/sdf/data/lcls/drpsrcf/ffb/users/monarin/jf-feespec-bulk-38995226/xtc`, experiment mfx101210926 run 387. Full-file SHA256 and saved SMD/timestamp references verified before timing.
- Sim10: five JF files from `/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sample10_10k`, based on mfx101555026 run 10; copied to `/sdf/data/lcls/drpsrcf/ffb/users/monarin/jf-cold-sim10-43989739`, verified against the reviewed manifest, and removed after completion. The original project-scratch fixture remains.
- These are different source runs with 32 panels/five JF streams and 10,000 events each; this compares storage workloads, not identical underlying events or a real compressor. Ancillary detectors excluded.
- Same four physical CPU cores and one visible A100; KvikIO 26.10, CuPy 14.0.1, forced CPU fallback, 4 workers, 4 MiB tasks, 16 MiB bounce buffer, auto-direct off, ring depth 35. The raw datagrams create larger device buffers and more 4 MiB subtasks; queue depth and configuration are matched, byte footprint/subtask count are not equal.
- No CUDA compute kernels, dense gather, XTC GPU parsing, calibration, expansion or cube. The SMD/manifest read plan is constructed before timing.
- Exclusive node, 128 GiB requested memory. Slurm allocated all four GPUs with the node; the benchmark exposed and used only one GPU. Order raw,sim10,sim10,raw. All five files are fsynced, advised DONTNEED, then measured with mincore; each must have zero resident pages after setup. This does not evict Weka server/storage caches.

The 312.5 GiB raw payload exceeds the 128 GiB memory request, while the 31.3 GiB sim10 payload fits. Raw final page residency was 38.21% and 39.71%, versus 99.9975% for sim10. This is consistent with page-cache reclaim during the raw loops and is an uncontrolled memory-pressure asymmetry, not a matched condition. Cgroup reclaim counters were not collected, so its contribution to throughput was not measured.

Weka server-side cache state was uncontrolled and may have been warmed: all input bytes had been hashed or copied through this client shortly before timing, and the second raw loop followed the first within minutes. Concurrent FFB load from other jobs was not measured. Repeat differences cannot be separated from server caching, changing shared-storage load, or the memory-pressure difference.

Physical NIC `enp225s0np0` reports 331–333 GiB for each 312.5 GiB raw payload and 33.2–33.3 GiB for each 31.3 GiB sim10 payload. Idle receive rate is about 0.93–0.98 MiB/s. Physical counters include protocol/background traffic; these values support network-backed reads but do not calculate an exact remote/cache fraction. NIC endpoints bracket and slightly exceed the loop interval. GiB/s in the table is requested payload divided by measured loop time, not NIC bytes/time.

## Validation and review

[Audit output](validation.json) passes all four cases; [seven tests](tests.log) accept native evidence and reject one cached page, wrong file identity, short count, wrong rate/counter, and incomplete cleanup. Python/shell syntax checks pass. The temporary FFB input was removed (`cleanup.json`). No production source changed.

Native evidence: `/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/jf_cold_ffb_20261109_r1`; logs in sibling `jf_cold_ffb_20261109_r1_logs`. Harness commit 693c25bb3; [submission](submission.txt). Claude round 2 **passed** commit `1a2d4043f1ed9155e3ec7f89270aa5b1426bd5c0`, accepting the clarified memory/cache/load limits with no unresolved blocking findings. See [review](review-02/review.json) and [responses](review-01/response.md). Optional auditor hardening COLD-R1-04 remains deferred with the reviewer-accepted rationale. No measurement or harness changes were made for the clarifications.

The earlier 8-worker psana staging attempt measured 5.617 GiB/s cold on a different node, with parsing/gather kernels, batch 20, 1 MiB tasks, and 1 execution stream. It is not a matched point in this table. Its warm/NVMe scope was superseded at the user's request; no warm result was accepted. See [history](../jf_raw_cache_20261109/README.md).
