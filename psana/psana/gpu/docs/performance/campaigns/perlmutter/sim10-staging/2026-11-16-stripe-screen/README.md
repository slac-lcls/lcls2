# Perlmutter psana stripe screen

Job 61372107 completed 18 cases on nid001481. Each processed 100k global events: four BDs sharing one GPU, four workers/BD, batch 200, one CUDA stream/BD, 1 MiB KvikIO task and stripe size, 16 MiB bounce, compatibility ON, direct OFF. Seven physical cores/BD give 34 bound cores including helpers.

| Stripe count | Median GiB/s | Observed range |
|---:|---:|---:|
| 1 | 5.4428 | 5.4325–5.4443 |
| 2 | 5.1854 | 5.1595–5.2021 |
| 4 | 5.0765 | 5.0404–5.0982 |
| 8 | 4.8803 | 4.8548–4.9080 |
| 16 | 4.7423 | 4.6686–4.7648 |
| 32 | 4.6480 | 4.6157–4.6656 |

Count 1 led all three blocks and became the subsequent scaling control. Its five files occupy five distinct OSTs. All 18 cases passed local cache checks; seven needed a second eviction attempt. NIC bytes were not collected. The terminal audit passed; independent result review remains pending in the source record.

Evidence and scripts: [original report](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_striping_pr298_20261111/screen-20261115/RESULTS.md) · [measurements](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_striping_pr298_20261111/screen-20261115/evidence/screen01/trials.csv) · [summary](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_striping_pr298_20261111/screen-20261115/evidence/screen01/summary.json) · [audit](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_striping_pr298_20261111/screen-20261115/evidence/screen01/audit.json) · [accounting](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_striping_pr298_20261111/screen-20261115/evidence/screen01/accounting-utc.txt) · [raw index](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_striping_pr298_20261111/screen-20261115/evidence/screen01/artifact-sha256.json) · [campaign script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_striping_pr298_20261111/screen-20261115/campaign.py) · [runner script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_striping_pr298_20261111/screen-20261115/runner.py) · [batch script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_striping_pr298_20261111/screen-20261115/run.sbatch) · [audit script](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_striping_pr298_20261111/screen-20261115/audit.py) · [tests](../../../../../../notes/archives/scaling-20261116/legacy-root/notes/psana_striping_pr298_20261111/screen-20261115/test_audit.py)

[Machine-readable record](manifest.json). Paths inside the record are relative to the repository root.
