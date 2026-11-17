# Perlmutter run-51 input preparation

A private copy of mfx100848724 run 51 completed on November 17, 2026.
Job **61480775** on nid002273 finished with **COMPLETED / 0:0 in 28:34**.
The [metadata and receipt audit](input-audit-02.json) passes. Final independent
review of the audit fixes and resulting evidence is pending.

| Prepared input | Value |
| --- | --- |
| Streams | 003, 005, 007, 008, 009; all 32 Jungfrau segments |
| Full copied files | Five XTC and five SMD files; 602,108,910,258 bytes (560.758 GiB) |
| Event selection | First 10,000 of 17,872 events; native step index 0 |
| Selected event bytes | 335,571,760,000 bytes (312.526 GiB) |
| Exact event ranges | 84 disjoint ranges per stream; 83 intervening 24-byte SlowUpdates excluded |
| Integrity | All ten source/copy full SHA-256 hashes match; selected-range hashes recorded |
| Header checks | 50,000 selected headers match SMD timestamp/service/size; 415 gaps are 24-byte SlowUpdates |
| Private ownership | Owner-only directory; ten read-only data files owned by the user, link count 1 |
| Lustre layout | All ten files: stripe count 1, stripe size 1 MiB; bigdata OST placement recorded |

The usable private root is:

```text
/pscratch/sd/m/monarin/psana2-gpu/datasets/mfx100848724-r0051-pr-regression-20261117-attempt02
```

Use its `dataset-manifest.json`, `event-reference.json`, and
`preparation-complete.json`. Their hashes and all supporting artifact locations
are in [artifact-index.json](artifact-index.json). Large data and reference plans
remain in scratch. This preparation does not establish durable retention or
create an SDF copy.

The independent reference came from a bounded SMD scan using a pinned psana
Dgram decoder, without iterating the benchmark workload. The actual files have
small SlowUpdate gaps, so the native adapter now supports exact disjoint ranges
(v2 manifests) while retaining v1 compatibility. Gap bytes are excluded from the
selected hash, requested-byte denominator and cache-residency byte count.
See [source scan](source-scan.json) and [preparation instructions](../../scripts/performance/regression/PREPARE.md).

The compute job ran the frozen code from
`199b5091c0482d015609ed728efa1aaaab0240ab`, after [Claude round2 PASS](review-02.json).
Full-file hashes and header checks ran on the compute node. The final audit
checks the receipts against the reviewed code, current file metadata, SMD hashes,
private ownership, actual layouts and scheduler records; it does not rehash
large XTC files on the login node. It explicitly refuses optimized Python so
assertions cannot silently disappear. All 55 offline tests pass.

## First attempt and recovery

Job **61478366** reached its 30-minute limit after copying all five large files.
Only streams 3 and 8 completed verification; no completion receipt was written.
The [failed attempt](attempt-01-failure.json) remains preserved and unaccepted.
The fresh retry used a separate directory, a 45-minute limit and unbuffered
random-access header reads. No integrity check was removed. Read-ahead was a
suspected contributor to the first timeout; these two jobs do not isolate or
measure that effect. Logs do not contain per-stream completion timestamps.

Both scripts requested 8 CPUs, one GPU and 32 GiB on premium/lcls_g. Slurm
actually allocated a full 128-CPU/four-GPU node with a 32 GiB memory request.
The [audit](input-audit-02.json) records requested and allocated resources
separately. GPUs were hidden and unused during input preparation.

## Remaining qualification

This completes the copying and reference-generation work, subject to final
independent evidence review. Runtime builds/import pins, compute placement,
cache eviction/residency, native correctness and A/A noise qualification remain
subsequent steps. The shipped profile stays disabled. No performance result or
cold-cache guarantee is inferred from successful preparation.

Only selected L1 headers are matched to SMD identities. Gap timestamp matching
and prefix non-L1 header validation remain explicitly deferred (PREP-002).
This input selection contains one native step and supplies no multi-step or
full EndRun coverage. Source files were not changed or evicted.
