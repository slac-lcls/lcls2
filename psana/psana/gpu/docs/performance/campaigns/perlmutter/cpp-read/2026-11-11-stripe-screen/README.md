# Perlmutter C++ reader stripe screen

Job 61119210 completed all 54 cases on nid001940. These are full-file C++ reads with fixed 1 MiB requests, five files and three randomized blocks. Medians are aggregate node GiB/s.

| Stripe count | 5 threads | 20 threads | 40 threads |
|---:|---:|---:|---:|
| 1 | 4.930 | 7.480 | 11.418 |
| 2 | 5.131 | 8.317 | 12.916 |
| 4 | 5.131 | 8.167 | 10.893 |
| 8 | 5.135 | 7.806 | 10.686 |
| 16 | 5.085 | 7.634 | 9.951 |
| 32 | 4.935 | 7.096 | 10.488 |

Count 2 led at 20 and 40 threads. Count 1 reached 11.42 GiB/s at 40 threads. This is a measured reader rate, not a storage ceiling. Stripe count 1 still spreads the five files across five OSTs. One placement per count was tested.

The 20-thread reader is the nearer concurrency reference to the 16 KvikIO workers in the one-GPU psana profile, but access sizes, CPU placement, node and processing differ. It is not an overhead-isolation control.

All cases passed local page-cache conditioning and receipt audits. NIC/storage traffic was not established; backend cache state was uncontrolled. Independent result review passed.

Evidence and scripts: [original report](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_striping_20261110/reader/RESULTS.md) · [measurements](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_striping_20261110/reader/sweep/trials.csv) · [audit](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_striping_20261110/reader/sweep/audit.json) · [accounting](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_striping_20261110/reader/sweep/accounting-utc.txt) · [raw index](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_striping_20261110/reader/sweep/artifact-index.json) · [figure](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_striping_20261110/reader/screening-throughput.pdf) · [reader source](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_striping_20261110/reader/reader.cpp) · [campaign script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_striping_20261110/reader/campaign.py) · [batch script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_striping_20261110/reader/run.sbatch) · [audit script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_striping_20261110/reader/audit.py) · [report script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_striping_20261110/reader/plot_results.py) · [tests](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_striping_20261110/reader/test_reader.py) · [build script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/sim10_striping_20261110/reader/build.sh)

[Machine-readable record](manifest.json). Paths inside the record are relative to the repository root.
