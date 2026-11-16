# Four-worker Weka NIC/cache profile

Completed on sdfampere036, one A100-SXM4-40GB, job 43878284 (2m17s, exit 0:0). This extends the [read benchmark](../mfx_kvikio_read_20261108/README.md)
with the September 18 Weka methodology, using the same five synthetic JF files.

Three fresh processes each run a full 10k-event cache-cold pass followed by two
warm passes. Each pass completes 50,000 requests / 33,582,000,000 bytes to GPU.
Four KvikIO workers, four logical CPUs, 35 outstanding buffers, 4 MiB tasks,
16 MiB bounce buffers, compatibility ON and auto_direct_io_read=False.
Slurm requests one GPU/16 CPUs per task plus node exclusivity. The exclusive
allocation reserved all 112 allocatable CPUs and four GPUs on the node; the reader
used device 0 only and restricted itself and its workers to four logical CPUs.
The actual submission and allocation are retained in `slurm-job.txt`.

The five files are owner-checked and confined to the accepted fixture directory.
After KvikIO initialization, fsync/FADV_DONTNEED evicts only these files; mincore
must show zero resident pages before each cold pass. No payload is changed and
no global cache flush occurs. There is no staging or backend/server cache reset.
These fixtures can be read by collaborators: eviction changes their local-node
cache residency, not contents; a concurrent reader may refill cache and fail the
cold gate. No such failure is silently relabeled as cold.

Each NIC's physical rx_bytes_phy/tx_bytes_phy counters bracket each timed pass.
Counters must exist and must not decrease; physical ports are reported separately,
without adding netdev or IB duplicates. A one-second sleep is bracketed by counter sampling before conditioning in
each process; the recorded idle envelopes were 2.78–2.88 seconds. This is node-wide traffic, including possible
OS/services traffic even with an exclusive Slurm allocation. Idle subtraction is
not used to claim exact file traffic. NIC intervals include endpoint sampling
overhead; application timers only cover request issue through completion/device
synchronization. Cache queries, report writes and GPU/CPU witnesses are outside
both read timing and the NIC endpoints. Near-zero NIC bytes on a warm read with
resident input pages support local-cache service. Read-ahead, protocol overhead,
and other traffic prevent exact cache-hit fractions from subtraction. The warm
label means repeated reads; actual residency is separately recorded.

Run artifacts: `/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/kvikio_network_20261108_r1`.
Job 43878284. Submission overrides: `--exclusive --time=00:10:00`, with stdout/stderr
under the adjacent `kvikio_network_20261108_r1_logs` directory.

Validation uses the original byte-count/effective-setting/GPU witness audit plus
independent NIC-delta/rate, cache gate and source-hash checks in `summarize.py`.
Historical baseline artifacts remain measurements of their original code commit;
re-auditing them against current source hashes requires that original checkout.

## Measured result

| Pass condition | Completed read rate, median (range), GiB/s | NIC receive / requested bytes | Input pages resident before |
|---|---:|---:|---:|
| Verified node-cache-cold, 3 passes | **4.021 (4.011–4.105)** | **105.32–106.13%** | **0%** |
| Warm repeats, 6 passes | **13.962 (13.493–15.047)** | **0.0122–0.0129%** | **99.9975%** |

All passes read 31.275675 GiB. The active physical port was enp33s0f0np0;
enp33s0f1np1 received zero bytes. Cold passes received 35.37–35.64 GB over the
active port versus 33.582 GB requested. Warm passes received only 4.08–4.32 MB.
This provides direct evidence of network service for the cold passes and strong
evidence that local page cache served the warm reads. It does not reconstruct
the cache fraction in the earlier 33-case sweep, whose NIC traffic was unmeasured.
The 15.62 GiB/s historical result must not be used as a physical Weka bandwidth
measurement. These new cold measurements characterize this four-worker workload,
not a maximum across worker counts, filesystems, nodes, or backend conditions.

The NIC endpoint envelope is **1.78–1.95 seconds longer** than the read timer
because interface discovery/ethtool sampling is outside read timing. Accordingly,
the raw NIC-envelope cold rate is **3.45–3.51 GiB/s**; do not present it as an
instantaneous or read-window network rate. The byte-ratio evidence is the useful
cache distinction here. Idle receive traffic was about 1.04 MB/s, similar to the warm-pass receive
traffic. These endpoint measurements are retained without subtraction, and all host traffic remains included. No high-frequency timeline
was collected; these are per-pass counter brackets matching the historical method.

All nine passes passed exact completion counts. All three processes passed 105
GPU/CPU sampled-byte witnesses each. Nine counter/summary tests and the five
existing buffer-ownership tests pass. `results.json` binds native reports and
executed reader/runtime source hashes; raw reports are retained on project scratch.
`slurm-accounting.txt` and the scheduler logs preserve allocation/completion evidence.

Independent Claude review: **PASS**, no blocking findings, at
`21c7c50dd842a8cf3c0863e7d775a72a93399a6c`. See [review responses](review-01/responses.md). Executed reader code is commit
`5e878baa84893537fd614d235d0ae57192197133`; subsequent commits add audit tests/results.

Reader device selection was logical CUDA device 0. The reader did not record its
CUDA_VISIBLE_DEVICES value or physical GPU UUID, so the batch-step inventory
cannot conclusively map that logical device to a physical UUID. GPU 1 showed
4% utilization in the final batch-step snapshot; its cause is unexplained.
`slurm-job.txt` proves the exclusive allocation, but this does not establish
absence of all system GPU activity. Future reports should capture UUID/mapping.

`input_direct_fd_count=5` is a descriptor-flag census, not evidence of actual
O_DIRECT transfers; see [metric definitions](../mfx_kvikio_read_20261108/metric-definitions.json).
Observed near-complete page-cache residency and near-idle warm NIC traffic support
buffered reads. The identity block's generic fixture cache label predates this
experiment; the report-level cache description and pass-level mincore gates
specify the actual conditioning for these measurements.
