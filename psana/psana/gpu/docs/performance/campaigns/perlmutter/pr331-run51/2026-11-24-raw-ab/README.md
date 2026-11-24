# PR331 Perlmutter raw A/B regression screen — November 24, 2026

[Full report](../../../../../../notes/master_sync_20261117/performance_20261124.md) and its compact evidence are copied byte-for-byte
from PR331 documentation commit `aca2c1c2a58d6fbb741da6184745b22883046e46`.
No runtime feature code is merged into the results branch for this record.

Five cases completed with passing correctness and terminal audits: three loop
comparisons equivalent at ±5%, two inconclusive. Six balanced pairs per case,
10,000 events per measured sample, separate 200-event warmups. Original Jungfrau
mfx100848724 run 51, not Sim10. Baseline `546764e0`, candidate `5bd5c7ba`.

A/A was omitted under the accepted narrower scope. Two execution reviews remain
pending; full component/observer qualification and persistent raw-evidence
retention remain open. Calibrated/task/broader scaling performance is deferred
under #330. This is not full PR acceptance or a merge-ready declaration.

[runs.csv](runs.csv) indexes 70 measured samples, including ten correctness samples;
warmups are excluded. `whole_case_seconds` is the recorded adapter process time,
not the allocation wall time or warmup-inclusive cost. Payload GiB/s uses requested
input bytes divided by 2**30 and loop seconds. Compare matched revisions within a
case; do not pool cases or infer scaling efficiency. Blank GPU-allocation, stripe
and NIC fields mean not exported here, not measured zero. Original resource data
remain linked by hashes in the full report. `workers_per_bd` denotes the configured
KvikIO worker count (four), not additional MPI ranks; CPU rows do not use KvikIO.

Run the portable compact-evidence verifier:

```bash
python -B psana/psana/gpu/notes/master_sync_20261117/performance_20261124/verify.py
```

The catalog's `documentary-v1` check verifies this wrapper and its artifacts;
the separate verifier reproduces all 20 case/metric summaries from saved sample
times. Neither substitutes for pending independent execution reviews.
