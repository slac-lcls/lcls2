# Perlmutter: controlled Sim10 streams and duration

**One CUDA execution stream was faster in every main-matrix comparison.** Its median staging rate was 2.166 GiB/s for 10k events and 2.195 GiB/s for 100k; two streams measured 1.259 and 0.969 GiB/s respectively. This is a within-site Lustre result, not a direct attribution of the earlier SDF/Weka discrepancy.

Native input validation, all ten cases, the independent audit and all **47 audit tests passed**. Job **61065065** completed **0:0** in **35m58s** on **nid001029**. **Independent final code and native-results review passed** on `185087d6efcc64d59113595605ce4cd8234bcdbb` with no blocking findings. Acceptance is recorded for the stated local-page-cache-only scope; [final review and closure](perlmutter-review-03/responses.md).

| CUDA streams | Events | Forward GiB/s | Reverse GiB/s | Median GiB/s | Median events/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 10,000 | 2.1555 | 2.1764 | 2.1660 | 692.5 |
| 1 | 100,000 | 2.1815 | 2.2083 | 2.1949 | 701.8 |
| 2 | 10,000 | 1.2562 | 1.2624 | 1.2593 | 402.7 |
| 2 | 100,000 | 0.9142 | 1.0233 | 0.9688 | 309.7 |

At 10k, the one-stream median was **1.72×** the two-stream median; at 100k it was **2.27×**. In paired forward/reverse comparisons, 2/1-stream rate ratios were **0.583/0.580** at 10k and **0.419/0.463** at 100k. The 100k/10k ratios were **1.012/1.015** for one stream and **0.728/0.811** for two streams. Thus one-stream throughput changed little with duration here; two-stream full cases were about **19–27% slower** than their paired prefixes. The two full two-stream repeats differed by about 12%. These are descriptive observations from two repeats, not confidence intervals or a diagnosed mechanism.

## Individual cases

| Case in execution order | Loop seconds | GiB/s | Final local resident fraction |
|---|---:|---:|---:|
| pilot-s1 | 3.667 | 0.8530 | 1.000% |
| pilot-s2 | 2.719 | 1.1501 | 1.000% |
| 00-s1-n10000 | 14.509 | 2.1555 | 10.000% |
| 01-s2-n10000 | 24.897 | 1.2562 | 10.000% |
| 02-s1-n100000 | 143.366 | 2.1815 | 32.313% |
| 03-s2-n100000 | 342.107 | 0.9142 | 29.699% |
| 04-s2-n100000 | 305.633 | 1.0233 | 26.680% |
| 05-s1-n100000 | 141.627 | 2.2083 | 37.192% |
| 06-s2-n10000 | 24.774 | 1.2624 | 10.000% |
| 07-s1-n10000 | 14.370 | 2.1764 | 10.000% |

First/last timed windows (including pilots): **2026-11-10T02:02:20.110438+00:00** to **2026-11-10T02:24:56.658752+00:00**. Slurm timestamps are saved explicitly in [UTC](native/slurm-accounting-utc.txt). Cache conditioning and setup also consume allocation time outside the loop timer.

## Controlled workload and acceptance

The same transferred 100k JF files at `/pscratch/sd/m/monarin/psana2-gpu/datasets/mfx101555026-r0010-sim10-100k-jf` supplied every case. The 10k case is a prefix of those files, not a different physical file layout. Original `generation.json` remains byte-identical; the transfer manifest explicitly selects physical streams s005–s009 and relocates their paths. All eleven independent SHA256 checks passed, followed by Configure/Names, panel mapping, every SMD timestamp/offset and a full 100k DataSource traversal. No new generation, transfer, calibration database setup or native build was needed.

One SMD0, one EB and one BD ran through Cray MPICH on one selected **NVIDIA A100-SXM4-40GB** (UUID `GPU-38dd3137-64b7-7ca8-74e6-4b01011bb179`, PCI `03:00.0`). CUDA reported 42,404,806,656 total bytes; nvidia-smi reported 40960 MiB. Slurm exclusive allocation reserved four GPUs; each timed step requested one, and only the BD exposed the selected UUID. Helpers were masked empty before MPI/psana/CuPy imports. Original step masks, physical UUID and inventory PCI were checked. All ranks shared the same **24 physical CPU cores** and NUMA interleave policy in every case.

Fixed controls: batch100, SMD1000, bulk524288B, GPU budget6GiB, KvikIO compatibility ON, 8 workers, 1MiB tasks,16MiB bounce, automatic direct I/O OFF, PS_FAKESTEP_FLAG=0, PS_PARALLEL=mpi, one EB and zero SRV. Frozen install-B identities matched all eight recorded Python modules. A separate native probe and the GPU pilots exercised Cray MPICH 9.1.0.794, CuPy 14.0.1, KvikIO 26.10.00 and CUDA runtime12.9. The probe ended during serial validation, before any timed case. Source hashes alone are not native ABI acceptance.

The task consumes field metadata only. Framework reads, XTC walking and descriptor preparation are included; no dense gather, expansion, calibration, cube or payload D2H is performed. This is simulated compression: 52429 uint16 seed words per original 512×1024 panel; the 100k fixture intentionally repeats payloads with unique event timestamps.

Every event completed five reads totaling **3,358,200 datagram bytes**. Each 100k case completed 500,000 requests and **335,820,000,000 bytes (312.756747 GiB)**. Each 10k case completed 50,000 requests and 33,582,000,000 bytes. Exact ordered identities, callback identities and read-size histograms passed. Every callback contained 100 events: 10/100/1000 callbacks for pilot/10k/100k. Requested CUDA stream counts were exercised. Admission capacity was recomputed from the runtime, with a 6GiB budget and 10% margin; all batches fit without pressure drains. CPU payload reads were zero, peer count was one, pending requests/active slots/held budget retired, and all rank cleanup barriers completed.

Before/after metadata checks confirmed all eleven input files retained size, owner, inode and modification time; no temporary XTC copy was created. All job steps exited 0:0. The cleanup JSON process boolean is a code-path assertion, supplemented here by actual Slurm exits, rank retirement receipts and [input metadata readback](native/input-stat-after.json). Memory records are snapshots, not continuous peaks or a claim that all CUDA cache allocations are zero.

## Cache and interpretation limits

Cold means **local page cache on the five bigdata files only**: owner/path/size-guarded fsync/FADV_DONTNEED followed by zero mincore pages after DataSource setup for every case. SMD files and Lustre server caches remain uncontrolled. Validation reads the entire fixture before the pilots; reverse 10k cases follow all full reads. Reversal does not equalize backend caches. Final residency is reported above, not used to infer cache hits or retroactively change the cold label.

The timer includes lazy GPU pipeline startup, allocation/module loading, fill, drain, event delivery and completion barriers. Short cases have a larger fixed-overhead fraction. Submitted-but-not-retired GPU-read peaks were **500 requests / 335,820,000 bytes**, exactly one 100-event batch, for both stream counts. These are bookkeeping peaks, not counts of simultaneous storage operations. Two exercised streams do not establish simultaneous execution. The measurements identify the better tested staging setting on this node; they do not explain the CUDA/runtime mechanism, measure a hardware ceiling, or establish an end-to-end analysis rate. CPU/GPU/platform, filesystem and runtime differences preclude attributing the historical SDF gap directly to these results.

**Remote traffic is unverified; traffic/payload ratios are unavailable.** Timed receipts retain raw Ethernet snapshots but those may miss KFI/Lustre traffic. The initial lctl lookup missed `/usr/sbin`; an absolute-path probe after every timed case found llite stats unavailable (`param_path llite/*/stats` absent and debugfs permission denied). CXI discovery exposed four SS11 devices and codeword/BER/pause rates, not an audited payload-byte counter. That probe ran only after complete.json and cannot provide measurement brackets. No network-served-byte or cache-hit-fraction claim is made. [Probe log](native/counter-probe.log).

Logs contain nonfatal NERSC module-telemetry Lua errors, the frozen reader’s generic “NVMe” CPU-fallback wording, and MPICH communicator-reference warnings at teardown. Native imports, exact payload accounting, cleanup and every Slurm step passed. The actual input mount is Lustre and KvikIO’s intended compatibility/CPU-bounce path was used.

## Provenance, review and artifacts

Execution-source changes ended at `97399922b`; `e39365a1d` added audit tests/reporting/review records without changing the 16 hashed execution/reference sources. Later test refinements are pinned by the SHA256 printed in [test-audit.log](native/test-audit.log). Final review passed on `185087d6e`, including direct access to the raw receipts and independent coordinator hash verification. Subsequent closure commits change documentation/evidence only. Initial implementation review at `cfadd496a` passed with nonblocking suggestions; [response](perlmutter-review-01/responses.md).

Artifact root: `/pscratch/sd/m/monarin/psana2-gpu/sim10-stream-length-20261109/attempt03`. Full per-case JSON/logs and commands are retained in `results/`; compact [audit](native/audit.json), [summary and paired ratios](native/summary.json), [input validation](native/input-validation.json), [commands](native/commands.txt), and [scheduler evidence](native/slurm-accounting-utc.txt) are copied here. Existing checkouts and native installs were not modified. No SDF allocation was resumed.

Failed attempts are preserved: attempt01/job61064869 verified all hashes, then exposed an overly broad Configure Names assertion; attempt02/job61064979 stopped before input reads on an incorrect single-granted-GPU assumption under exclusive Slurm. Neither produced accepted benchmark data. The final job revalidated the complete input and ran all cases in one allocation.
