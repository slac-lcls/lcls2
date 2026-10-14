# A / B+gather / B+gather with on-demand wrappers

One-allocation warm comparison following the on-demand locator implementation.

- A: preserved `f52e90cc66` legacy-addressing build.
- G: preserved B+gather build measured in job 38564718, with eager wrappers.
- Z: independent copy of G, changing only `gpudgram/parser.py` to create
  configured locator wrappers on first access. B+gather itself creates none.

`builds.json` records origins and source identity. A and G were verified against
all 25 hashes per build recorded by job 38564718. G/Z differ in exactly one file,
and AST comparison limits that difference to `_locate_configured` and `locate`.
Unrelated memory-reporting cleanup is excluded from both measured G/Z builds.
Existing benchmark installations and production sources are untouched.

## Workload and ordering

Six clean repetitions of each variant, using all six permutations:
AGZ, GZA, ZAG, ZGA, GAZ, AZG. Each variant occupies each position twice; each
ordered adjacent pair within a round occurs twice. No phase-timing hooks,
profiling, or user D2H are enabled in the measured samples.

The workload and timer are retained from the four-way comparison: 10,000 events
from mfx101210926/run 387, 32 Jungfrau segments, five streams s005-s009, batch 20,
depth 1, 8 GiB GPU budget, three MPI ranks (one SMD0, one EB, one BD), one A100,
48 requested CPUs, 450 GiB host memory. KvikIO compatibility ON (CPU fallback),
eight reader threads and 1 MiB tasks. No bulk-read integration.

All builds use the same frozen CPU-reference calibration snapshot. Each sample
has a separate 100-event warmup, and the event-loop timer includes detector
setup and final GPU synchronization. Staging and cache preparation are outside
the timer. Cache residency must be >=99% before and after every measured run.

Z runs the 33 locator/gather integration tests inside the allocation. All three
builds must then pass fresh CPU-reference preflights. Hashes, actual import
paths, CPU affinity, GPU identity, runtime versions, memory samples and host
counters are retained in each job directory. All ranks share the allocation
CPU mask, as in the previous comparisons.

```bash
source setup_env.sh
source install_psana/activate.sh
python -m pytest -q validation/lazy-wrapper-perf-20260920/test_phase_timing.py
sbatch validation/lazy-wrapper-perf-20260920/run.sbatch
python validation/lazy-wrapper-perf-20260920/audit.py validation/lazy-wrapper-perf-20260920/job-JOBID-warm-ab
python validation/lazy-wrapper-perf-20260920/summarize.py validation/lazy-wrapper-perf-20260920/job-JOBID-warm-ab
```

Harness scripts are adapted copies of the preserved four-way scripts. Measured
benchmark behavior is unchanged; variant selection, orders, audit expectations,
and the separate Z correctness preflight are adapted for this comparison.

Completed job **38676923**, sdfampere034, **COMPLETED / 0:0**, elapsed **55:56**.
Audit passed for all 18 samples and three CPU-reference preflights; 33 Z
integration tests passed. All 36 accepted before/after cache checks were 100%.
Medians: A **22.464388 s**, G **25.288431 s**, Z **24.824160 s**.
Z median elapsed is **1.84% lower than G**, and **10.50% above A**. Z wins four
of six rounds; ranges overlap, so no stable speedup is established.
Full report: `psana/psana/gpu/docs/performance/lazy_locator_wrappers_sdf.md`. Eight harness tests passed.
The benchmark AST matches the prior harness except for variant choices;
position and within-round adjacent-pair balance were checked.
