# psana sim10 stripe-screen results

Job **61372107 completed 0:0** on nid001481, lcls_g/gpu_premium, November16,2026 00:01:57–00:53:29UTC (**51m32s**). All21numeric steps and parent/batch/extern completed0:0. All31native GPUtests,30file validation,18cases and the frozen post-job audit passed. Independent results review is pending direct authorization: automatic approval review rejected the external Claude invocation. The completed job and local audit passed.

**Count1 led all three randomized blocks for this fixed single-node staging workload**, with median5.4428GiB/s. This is an observed screen result for these file placements, not a universal OST or multinode optimum. Count1 means one OST per file; the five files occupy five distinct OSTs116,118,119,120,121.

Profile: PR29867c327a;4BDs sharing1A10040GB,1CUDAstream/BD,4KvikIOworkers/BD,batch200,512KiB bulk target,1MiB KvikIOtask,16MiB bounce,compatibilityON,auto-directOFF;1MiB stripe size. 100000events and335820000000payloadbytes per case. Three randomized complete blocks, six counts. No calibration/cube computation.

Throughput is aggregate application payload bytes divided by the slowest rank event-loop time. Setup,cache eviction and teardown are excluded. Ranges are observed min–max across three repeats, not confidence intervals. Paired ratios use matching blocks and count2 as reference.

| Stripes | Median GiB/s | Min–max GiB/s | Median loop seconds | Paired geometric mean vs count2 |
|---:|---:|---:|---:|---:|
| 1 | 5.4428 | 5.4325–5.4443 | 57.463 | +4.97% |
| 2 | 5.1854 | 5.1595–5.2021 | 60.315 | +0.00% |
| 4 | 5.0765 | 5.0404–5.0982 | 61.609 | -2.14% |
| 8 | 4.8803 | 4.8548–4.9080 | 64.086 | -5.81% |
| 16 | 4.7423 | 4.6686–4.7648 | 65.950 | -8.82% |
| 32 | 4.6480 | 4.6157–4.6656 | 67.289 | -10.41% |

All18cases evicted all30payloadfiles, then passed two zero-resident-page mincore scans and another selected-five-file scan. 7 cases needed a second attempt; none relaxed the gate.

**NIC bytes were not collected.** Per-case before/after osc.*.stats queries returned unavailable; backend caches and filesystem contention were uncontrolled. These are verified cold local-page-cache application rates, not measured network or physical-storage rates.

Sampled device peak4.9804GiB and minimum free35.0196GiB across all cases. Tracked per-BD budget maxima sum2.5703GiB, with each limit6GiB. Sampling may miss brief peaks; summed tracked maxima are not simultaneous total-device usage.

The C++ count1 screen observed7.48GiB/s at20threads and11.42GiB/s at40threads. Those are different access patterns, concurrency,CPU placements and nodes, and exclude psana parsing/GPU staging. They do not establish a single-OST ceiling or quantify removable psana overhead.

Phase allocation usage:3747prior +3092screen =6839node-seconds;14761seconds remain of21600. These are node walltime, not premium charged hours. No further job submitted.

Evidence: [audit](evidence/screen01/audit.json), [trials](evidence/screen01/trials.csv), [summary](evidence/screen01/summary.json), [accounting](evidence/screen01/accounting-utc.txt), [cache checks](evidence/screen01/cache-summary.json), [raw artifact hashes](evidence/screen01/artifact-sha256.json). Original raw receipts remain under /pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111/screen01/results. Compact case summaries replace full timestamp/callback arrays; the coordinator audit recomputed full partitions/range digests against raw files.

Reviewed runtime78c0fb0ae6fa30535112f5f88639e9e9bab783c3, immutable bundle-screen01. No runtime changes after prelaunch pass. Results review will assess this evidence and its interpretation separately.
