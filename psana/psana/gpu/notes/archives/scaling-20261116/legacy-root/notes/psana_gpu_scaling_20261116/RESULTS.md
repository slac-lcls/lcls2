# 100k-event GPU scaling results

Job 61385304 completed on nid001008 in 1,823 seconds (30m23s), with every Slurm step exiting 0:0. The terminal raw-evidence audit passed against reviewed implementation a493c9cacc29b922e48a82f79d1fa63ed548810b.

Three randomized repetitions per configuration; 100,000 total events (312.76 GiB requested) per case. Timed staging includes read, parse, and drain; setup and cache eviction are excluded.

| GPUs | BDs | Median GiB/s | Range GiB/s | Paired speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 5.075 | 5.037–5.075 | 1.00× | 100.0% |
| 2 | 8 | 7.558 | 7.451–7.614 | 1.49× | 74.5% |
| 3 | 12 | 9.341 | 9.270–9.342 | 1.84× | 61.4% |
| 4 | 16 | 9.842 | 9.829–9.859 | 1.94× | 48.6% |

Four GPUs reach 9.842 GiB/s, with a median paired speedup of 1.94× versus one GPU. The fourth GPU adds 5.37% over the three-GPU median. GPU count, BD count, CPU allocation, and reader concurrency increase together; these results do not isolate a GPU or storage bottleneck.

All 12 cases passed local page-cache coldness checks before timing: two zero-residency scans of all 30 files plus the selected five-file check. Case r2-g1 required a second eviction attempt; the other eleven passed on the first attempt. Remote/OST cache coldness is not established.

All four Cassini NICs were monitored around every timed case, receiving 317.97–321.66 GiB per case. These counters include all NIC traffic and protocol overhead and are not storage-only measurements. All 124 native GPU tests passed. Peak sampled GPU usage was 4.981 GiB; minimum free memory was 35.019 GiB.

The baseline uses three physical CPU cores with both SMT threads per BD so four GPUs fit in one node. The earlier 5.443 GiB/s stripe-screen baseline used seven physical cores per BD. Use the contemporaneous 5.075 GiB/s baseline for these speedups. Stripe count remains one per file, with five files on five distinct OSTs.

Evidence: [summary](evidence/screen01-results/summary.json), [audit](evidence/screen01-results/audit.json), [accounting](evidence/screen01-results/accounting-utc.txt). Raw attempt: /pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111/scale-screen01.
