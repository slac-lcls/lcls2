# Prepared PR298 stripe sweep and accepted small pilot

**Update November 15:** the historical 72-case CSV and launcher below are not
the current execution plan. The [revised selection](../selection-20261115/README.md)
proposes 18 cases at four BDs, four workers/BD, batch 200, one stream/BD.
This harness still needs adaptation and qualification for those settings.

The 72-case parameter matrix is in [sweep.csv](sweep.csv). Job 61133510 passed the 14-case pilot and full independent audit; the final review-loop round passed with no blocking findings. Review commit: `0723c08e841dd26b5c42269ba2fe832c555ca1f3`; verdict in [review-03.json](review-03.json). See [RESULTS.md](RESULTS.md) for accepted timings, controls, limitations and the three rejected diagnostic attempts.

Exact PR298 source: `67c327a148a08b2aaf972e67e1e02e367c2b298f`, with an isolated install_psana under the scratch source checkout. Executed immutable harness bundle: `e7c61f05ab31923fadf8579223b2e74b020d5d79`. No production source changed. All current harness files match bundle04 byte for byte; evidence/pilot04/bundle-comparison.json records this.

Validation: 1,023 CPU unit tests, 31 native GPU tests, seven audit regressions, all 14 native cases, every Slurm step at 0:0, and a full audit pass. The CPU-placement guard rejected pilot03. Fixed affinity was then derived from nine probe ranks across the 3/6-rank Slurm layouts and verified in every pilot04 case.

Prepared workload: 100k events across five JF files per case; stripes 1/2/4/8/16/32 × CUDA depth 1/2 × BD readers 1/4 × three randomized blocks. Fixed bulk target 512 KiB, batch 100, KvikIO 8 workers / 1 MiB tasks / 16 MiB bounce buffers, compat ON, direct OFF, and 6 GiB per BD. Metadata views point at existing accepted copies; no new payload copy. The runner measures descriptor staging, excluding calibration, expansion and cube processing.

The full screen and 100k sizing run remain unsubmitted. Peak-memory telemetry and a sized, reviewed launcher remain deferred before full-run submission. Reference arrays stay in scratch with hashes in the validation receipt and summary; the independent audit checked their hashes and every physical-range digest. Required review evidence is inside this checkout; the reviewer needs no external payload or source reads.
