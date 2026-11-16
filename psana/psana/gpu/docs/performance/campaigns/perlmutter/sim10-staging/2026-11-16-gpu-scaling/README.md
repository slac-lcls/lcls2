# Perlmutter psana GPU scaling at stripe count one

Job 61385304 completed on nid001008 in 30m23s. Each case processes 100k global events (312.76 GiB requested), with three randomized repetitions. Stripe count is **1**, stripe size 1 MiB, five files on five distinct OSTs. Four BDs/GPU, four KvikIO workers/BD, batch 200 and one CUDA stream/BD remain fixed.

| GPUs | BDs | Median GiB/s | Observed range | Paired speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 5.075 | 5.037–5.075 | 1.00× | 100.0% |
| 2 | 8 | 7.558 | 7.451–7.614 | 1.49× | 74.5% |
| 3 | 12 | 9.341 | 9.270–9.342 | 1.84× | 61.4% |
| 4 | 16 | 9.842 | 9.829–9.859 | 1.94× | 48.6% |

Four GPUs nearly double aggregate throughput; the fourth adds 5.37% over the three-GPU median. GPU count, BDs, CPU allocation and worker concurrency increase together. The fixed single SMD0/EB and other shared resources may limit scaling; the result does not isolate the bottleneck.

The scaling CPU policy uses three physical cores with both SMT threads per BD. Its contemporaneous 5.075 GiB/s one-GPU baseline is the speedup denominator; the earlier stripe screen used seven cores/BD and reached 5.443 GiB/s.

The previous pilot, job 61384503, also used stripe count 1 in every case: four 6,403-event GPU-count cases plus two 100k one-GPU CPU-policy controls. Pilot validation overhead and event counts differ; those short cases are excluded from this table.

Earlier attempt 61384230 failed before GPU tests because the CPU-mask intersection had only 53 cores. It is retained in the attempt history and excluded from all measured tables.

All 12 full cases passed cold local-page-cache gates (r2-g1 needed two eviction attempts). All four Cassini NICs were recorded around every timed case: 317.97–321.66 GiB received, including protocol and other traffic. All 124 native GPU tests passed; maximum sampled GPU usage was 4.981 GiB. The raw-evidence audit passed after terminal Slurm success. Implementation review passed; a separate independent results review has not been recorded.

Evidence and scripts: [original report](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_gpu_scaling_20261116/RESULTS.md) · [measurements and audit](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_gpu_scaling_20261116/evidence/screen01-results/audit.json) · [summary](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_gpu_scaling_20261116/evidence/screen01-results/scaling-summary.json) · [diagnostics](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_gpu_scaling_20261116/evidence/screen01-results/summary.json) · [accounting](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_gpu_scaling_20261116/evidence/screen01-results/accounting-utc.txt) · [raw index](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_gpu_scaling_20261116/evidence/screen01-results/raw-receipt-hashes.json) · [qualification](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_gpu_scaling_20261116/evidence/pilot02-results/summary.json) · [prepare script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_gpu_scaling_20261116/prepare.py) · [campaign script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_gpu_scaling_20261116/campaign.py) · [batch script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_gpu_scaling_20261116/run.sbatch) · [report script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_gpu_scaling_20261116/report.py) · [audit script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_gpu_scaling_20261116/audit.py) · [tests](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_gpu_scaling_20261116/test_scaling.py)

[Machine-readable record](manifest.json). Paths inside the record are relative to the repository root.
