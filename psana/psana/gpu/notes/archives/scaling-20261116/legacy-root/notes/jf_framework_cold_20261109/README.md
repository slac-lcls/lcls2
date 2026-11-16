# Psana2-GPU framework staging from cold Weka FFB

This measures the requested **psana2-GPU framework**, using DataSource/MPI event delivery through GPU staging and final draining. It supersedes the standalone KvikIO comparison as the answer to the user’s framework question. One SMD0, one EB, one BD, one A100 used on sdfampere040; JF only, 10,000 events per full run. Both inputs are on Weka FFB. No NVMe input stage, calibration, expansion, cube, or user compute kernel. Framework XTC parsing is included; dense mode additionally includes the framework raw-gather kernel.

## Matched framework comparison

| Case | GiB/s | Events/s | Loop seconds | Final page residency | NIC RX GiB |
|---|---:|---:|---:|---:|---:|
| 00-raw-field | 3.006 | 96.2 | 103.954 | 85.05% | 336.314 |
| 01-sim10-field | 1.494 | 477.7 | 20.932 | 100.00% | 33.408 |
| 02-raw-dense | 3.144 | 100.6 | 99.405 | 98.36% | 334.717 |
| 03-raw-dense | 3.204 | 102.5 | 97.557 | 99.39% | 335.913 |
| 04-sim10-field | 1.504 | 481.0 | 20.791 | 100.00% | 33.774 |
| 05-raw-field | 3.156 | 101.0 | 99.037 | 100.00% | 332.726 |

With the same settings, raw descriptor staging reaches **3.01–3.16 GiB/s**, versus **1.49–1.50 GiB/s** for sim10 descriptor staging. Raw dense staging reaches **3.14–3.20 GiB/s**. Thus the larger payload achieves about twice the byte throughput in this comparison, while sim10 processes about five times as many events per second. This is consistent with amortizing fixed per-event/per-batch costs over more bytes; it does not identify which stage dominates or prove input size is the only cause. The matched configuration did not itself reach 4–5 GiB/s; the separate historical control below did. The raw-field and raw-dense ranges are close enough that these two repeats do not establish an incremental gather cost.

All six runs delivered exactly 10,000 events in 1,500 callbacks and exercised two execution streams. Each completed 50,000 GPU read requests: **335,571,760,000 bytes (312.526 GiB)** raw, **33,582,000,000 bytes (31.276 GiB)** sim10. Peak outstanding requests were 35 in both modes; peak requested bytes differed (234,900,232 raw versus 23,507,400 sim10). The same number of requests does not mean the same byte footprint or KvikIO subtask count. Individual physical-file datagrams are approximately 5–7 MiB raw versus 0.5–0.7 MiB sim10.

## Historical configuration control

**Yes: the framework reached 5.04 and 4.91 GiB/s** on uncompressed cold FFB input with the historical staging settings, including production XTC parsing and dense raw gathering. Both runs used the same node, GPU, frozen runtime and raw inputs as the matched matrix. This is a separate tuning/configuration control, not a matched raw-versus-sim10 pair, and the several changed settings cannot be attributed individually.

| Repeat | GiB/s | Events/s | Loop seconds | NIC RX GiB |
|---|---:|---:|---:|---:|
| 1 | 5.040 | 161.3 | 62.012 | 336.321 |
| 2 | 4.914 | 157.2 | 63.602 | 340.497 |

The historical control used its own newly created CuPy compiler-cache directory and no preflight: repeat 1 may include in-loop JIT work; repeat 2 can reuse that cache. This differs from the preflight-prepared cache in the matched matrix.

Both read 312.526 GiB in exactly 50,000 requests/10,000 events, began with zero resident pages per file and ended with approximately 100% residency. No CPU payload reads or admission-pressure drains occurred. [Control audit](../jf_framework_legacy_control_20261109/validation.json) and [five acceptance tests](../jf_framework_legacy_control_20261109/tests.log) pass. Both NIC totals support network-backed reads, with the same server-cache/background-traffic caveats as the main matrix.

| Setting | Matched comparison | Historical control |
|---|---|---|
| BD / GPU used | 1 / 1 | 1 / 1 |
| EB batch_size | 7 | 20 |
| SMD event batch | 20 | 1000 |
| Execution stream pool | 2 | 1 |
| KvikIO workers / task | 4 / 4 MiB | 8 / 1 MiB |
| Bounce / fallback / auto-direct | 16 MiB / forced ON / off | same |
| Bulk target | 512 KiB | 1 MiB |
| Framework memory budget | 4 GiB | 6 GiB |
| CPU placement | one core/helper, four physical BD cores | helpers and BD share all allowed physical cores |
| User callback | minimal GpuTask host callback | none; benchmark adapter requests dense gather |
| Host memory / allocation policy | 700 GiB / interleave all NUMA nodes | same request / interleaved launch |

## What is timed and checked

The timer covers the whole MPI loop from before the start barrier through event delivery and final BD GPU synchronization/drain, ending after the completion barrier. It includes framework staging, parsing, callback/metadata preparation, event retirement, and lazy GPU allocation/JIT that occurs inside iteration. Setup, full-file verification, input copying, explicit cache conditioning, NIC endpoint sampling and acceptance checks are outside that timer. For the matched matrix, preflight processes prepare a shared compiler cache; this is client-data-cold, not a cold-JIT test. NIC counters bracket a slightly wider interval than the loop; payload GiB/s uses requested bytes divided by loop time.

Raw-field and sim10-field use public GpuTask tuple selectors and inspect descriptors without a user kernel. Raw-dense uses `GpuTask(inputs=('jungfrau.raw',))`, requesting the production dense gather; the callback inspects shape/type. Only the separate 200-event raw-dense preflight copies arrays to CPU and compares three complete gathered arrays against frozen CPU hashes. Full runs perform no validation D2H and publish no arrays. No admission-pressure drains or CPU BD payload reads occurred; the runner asserted that GPU managers closed with no pending reader futures, active events, or held leases. Those drained-state values were not serialized separately, so offline review relies on the successful in-process assertions. The [audit](validation.json) passes all eight cases; [eight tests](tests.log) accept native evidence and reject corrupted cache, byte/count, delivery, rate and NIC records.

Every full run starts with zero resident pages for each of its five bigdata files, after DataSource setup and file-specific fsync/FADV_DONTNEED plus an independent mincore check. Physical NIC RX is about 333–336 GiB for raw and 33.4–33.8 GiB for sim10; idle RX is 0.90–1.01 MiB/s. These physical-interface totals include protocol/background traffic and support network-backed reads; they are not an exact remote/cache attribution. Server-side Weka caches and concurrent shared FFB load are uncontrolled, and verification/copying reads every input byte before cache eviction.

The 700 GiB request is larger than either payload; all matched ranks report an interleaved allocation policy. That does not guarantee identical cache behavior: raw final residency ranged from 85.05% on the first raw-field case to almost 100% on the last, versus 99.9975% for sim10. Cgroup reclaim was not instrumented in the matched harness; the remaining difference is disclosed rather than attributed to one mechanism. Two repeats are not a steady-state maximum or a statistical confidence interval.

## Inputs, artifacts and review

Raw input is the owned mfx101210926 run387 JF prefix set in `/sdf/data/lcls/drpsrcf/ffb/users/monarin/jf-feespec-bulk-38995226/xtc`. Sim10 is based on mfx101555026 run10 from the accepted project-scratch sample10_10k fixture; only its five JF files and SMD were copied to a unique FFB directory, SHA-verified and removed after the matrix. The sources have different events, schemas and transitions, so this is a workload comparison, not paired codec/scientific validation. The raw shape and both descriptor segment lists verify 32 panels. The legacy control rechecks SMD/owner/size identity and relies on its immediately preceding job’s full raw SHA verification.

Runtime is the same frozen B psana install, CuPy 14.0.1 and KvikIO 26.10 CPU fallback. The runtime warning says “NVMe” generically; findmnt and the actual input paths show Weka FFB. No detector or calibration DB was modified. Exclusive allocations may allocate all four node GPUs; each benchmark exposes and uses only one.

Main artifacts: `/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/jf_framework_cold_20261109_r1` (job43996574). Historical control: sibling `jf_framework_legacy_20261109_r1` (job43998942). Log directories use the same roots with `_logs`. [Submission records](submission.txt). Harness commits: 3c69eed3c (matched), f3e7d9e1d (legacy launcher). The measured main harness remains unchanged. Independent Claude review **passed** code commit `ab76dd53f82e2c3631d8b6ca66fd719d2352c768` with no blocking findings; see [review](review-01/review.json) and [responses](review-01/response.md). Optional future audit/cleanup hardening is recorded there. A supplemental [sim10 identity check](sim10_identity_check.json) independently confirms the native input timestamp list against the SHA-verified accepted generation manifest. No runtime code or measurement changed after review. Both hardware jobs completed successfully; the temporary sim10 FFB copy was removed. MPI shutdown logs contain nonfatal UCX unmatched-tag warnings after timing; exact event/I/O checks and GPU draining passed, and these warnings were not investigated in this benchmark.
