# Sim10 bulk-read target sweep

**Larger bulk targets reduced reader calls but lowered framework staging throughput.**

| Bulk target | GPU reader calls/run | Mean read size | GiB/s, two runs | GPU callbacks/run | Events/GPU subbatch | Peak outstanding reads | Peak outstanding input |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512 KiB | 50,000 | 0.641 MiB | 2.1884, 2.1197 | 100 | 100 | 500 | 320.26 MiB |
| 4 MiB | 9,454 | 3.388 MiB | 1.5626, 1.5864 | 4,356 | 1–5 | 5 | 17.61 MiB |
| 8 MiB | 4,599 | 6.964 MiB | 1.7871, 1.6503 | 2,233 | 1–11 | 5 | 38.43 MiB |

All full cases read **33,582,000,000 bytes (31.275675 GiB)** over 10,000 events
from five JF XTC files. Reader calls are GPU-reader requests to KvikIO, not OS
syscalls or KvikIO worker tasks. Peak outstanding counts mean submitted requests
not yet retired through wait_batch; some may already have finished physical I/O.
They do not show simultaneous storage activity. Both repeats matched the independent predicted
counts and complete read-size histograms exactly.

At 4 MiB the mean rate was 1.574 GiB/s, 26.9% below the 512 KiB mean of 2.154 GiB/s.
At 8 MiB the mean rate was 1.719 GiB/s, 20.2% below the 512 KiB mean of 2.154 GiB/s.

The small-group restriction matters even with **one CUDA execution stream**:
it allows only one live small input group per physical XTC file. Peak pending
reads fell from 500 to 5. Group boundaries also changed execution scheduling:
100 callbacks of 100 events became thousands of callbacks of 1–5 or 1–11 events.
Thus coalescing worked, but the same configuration change also reduced the number of
reads submitted ahead of each wait and increased execution/callback overhead. These observations are
consistent with scheduling costs offsetting larger reads; no section timings
were collected to assign the slowdown to individual components. This does not
show that larger reads alone are slower, or that storage bandwidth is the limit.

The measured choice for this current single-stream pipeline remains **512 KiB**.
An isolated read-size experiment would need to preserve input concurrency and
execution batching while varying physical requests; that change is outside this
sweep. No production framework code was changed.

Actual subbatch histograms (events per callback → number of callbacks); both
repeats at each target were identical:

- 512 KiB: `{"100": 100}`.
- 4 MiB: `{"1": 1574, "2": 1027, "3": 985, "4": 433, "5": 337}`.
- 8 MiB: `{"1": 334, "2": 418, "3": 223, "4": 292, "5": 290, "6": 127, "7": 254, "8": 26, "9": 59, "10": 46, "11": 164}`.

Job **44009313** completed with exit **0:0** in **5m50s** on **sdfampere042**.
The full audit and **12 positive/negative validation tests passed**. Each case
delivered exact ordered timestamps and callback identity, all requested bytes
completed, no CPU payload reads or admission drains occurred, and GPU reader,
event-slot, and budget resources drained. The temporary FFB copy was removed.
The original fixture remains intact. See [validation.json](validation.json),
[tests.log](tests.log), and [jobs.txt](jobs.txt).

All five payload files had zero initial resident pages in every case and
99.9975% final residency. Physical NIC `enp225s0np0` recorded:

| Run | Payload GiB | NIC RX GiB | Events/s |
|---|---:|---:|---:|
| 00-target-524288 | 31.276 | 33.862 | 699.7 |
| 01-target-4194304 | 31.276 | 34.019 | 499.6 |
| 02-target-8388608 | 31.276 | 32.951 | 571.4 |
| 03-target-8388608 | 31.276 | 33.197 | 527.7 |
| 04-target-4194304 | 31.276 | 33.430 | 507.2 |
| 05-target-524288 | 31.276 | 34.802 | 677.7 |

Measured-case idle NIC RX was 0.93–1.03 MiB/s. NIC totals include protocol,
SMD/background traffic and a wider timing bracket; they do not provide
an exact remote-read/cache-hit fraction. The NIC bracket exceeded the loop timer
by about 0.9–1.2 seconds in five runs and 4.3 seconds in run 04 (4 MiB). The table
reports NIC byte totals, not NIC rates; these counters were not used to compute
the payload GiB/s results. Both repeats used the same GPU:
`GPU-1b50deab-6fd5-0cd6-f752-536b4de64f73`, PCI `0000:01:00.0`.

Settings: one BD, one A100-SXM4-40GB, one CUDA execution stream, EB batch 100,
SMD batch 1000, eight KvikIO workers, 1 MiB tasks, 16 MiB bounce buffer, fallback
ON, automatic direct I/O OFF, 6 GiB GPU budget. All MPI ranks shared 56 physical
cores with NUMA interleaving; exclusive node allocation requested 700 GiB.
Runtime: frozen B install, CuPy 14.0.1, KvikIO 26.10.00; module paths/hashes are
recorded in the native results. Inputs are sim10 seed descriptors via public
GpuTask. No dense gather, calibration, expansion, cube, user kernels, or timed
payload D2H. This is framework staging throughput; pixel values were not inspected.

Three 200-event preflights ran at 512 KiB, 4 MiB, and 8 MiB. The six full runs
then ran at 512 KiB, 4 MiB, 8 MiB, 8 MiB, 4 MiB, and 512 KiB. Preflights warmed compiler caches; lazy allocations in
each new process remained timed. Timing covered the MPI event loop and final
GPU drain. Setup, SHA verification/copying, cache eviction, idle samples and
final cache scans were excluded. Only two repeats per setting were measured;
shared storage traffic was uncontrolled, so no statistical confidence claim is made.

Input came from the SHA-verified accepted sample10_10k fixture copied into this
campaign’s unique **Weka FFB** directory (`drpsrcf/ffb`). No NVMe input staging.
Only the five bigdata payload files were evicted after DataSource setup and
mincore-checked. SMD and server caches were uncontrolled. Node-local directories
held temporary/compiler files only. The runtime warning mentioning NVMe is generic.

The independent read-count predictor uses exact per-file offsets and batch100
boundaries without production planner imports; gaps break groups. A separate
offline frozen-runtime planner check matched all six event-count/target cases
before hardware validation (cpu-plan-check.json). The live counters now match too.

Implementation: `f849540905a19cc2c18d620702088ade120740fe`, unchanged since the
code-only review. **Claude results review passed** in round 2 on commit
`f171fdb16c3ec9980a5bbbe60eee50de986645cc`, with no blocking findings. SBT-02 was
confirmed resolved. The two new wording suggestions, SBT-04 and SBT-05, are
addressed in this evidence-only update. The benchmark code is unchanged. See
[review](review-02/review.json) and [responses](review-02/response.md). Optional SBT-01/SBT-03 cleanup remains
deferred; instrument.py is hashed as lineage for bulk_accounting.py, and the
inherited --preflight flag is unused (campaign selects short cases by events=200).

Native output root:
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sim10_bulk_target_20261109_r1`.
Slurm logs use the same path with `_logs` appended. Large evidence copies for
review are under the detached scratch worktree’s `build/bulk-review-evidence`.

Validation commands (Python 3.11 from xpp_drp_gpu_311):

```bash
python notes/sim10_bulk_target_20261109/audit.py /sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sim10_bulk_target_20261109_r1
python notes/sim10_bulk_target_20261109/test_audit.py /sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sim10_bulk_target_20261109_r1
```

The coordinator verified 25 native evidence copies byte-for-byte by SHA-256,
ran the audit and tests, and confirmed benchmark code was unchanged. Claude
reviewed the copied files with read-only tools; it could not independently run
tests or hashes, and inspected long timestamp/callback arrays selectively.
These limitations are retained in the review record.
