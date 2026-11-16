Independent HARNESS READINESS review only, not GPU/performance acceptance.
The user requested reproduction of uncompressed JF raw-read performance with
strict cold/warm gates and NIC counters, and previously requested this session's
Claude review loop. Main task remains incomplete: Slurm job43983799 is pending
resources with no start estimate. There are no new GPU results to review.

Review base319d1c8ce through exact checkout SHA supplied by helper, changes only
notes/jf_raw_cache_20261109. Read scope.md and README.md; inspect bench.py,
common.py, input_adapter.py (unchanged copy of stage1_regression adapter), read.py,
campaign.py, runtime.sh, run.sbatch, analysis/audit.py/test_audit.py. Check cache
semantics, timing/ownership, exact counts, GPU selection, memory/storage scoping,
comparison fairness and failure handling. No production psana/kernel changes.
Review for correctness and execution readiness; do not infer runtime success.

CPU evidence: tests.log seven passed cache/NIC tests, Python AST and bash -n
passed. Coordinator ran requests(SOURCE) with the pinned runtime on login node:
10000 timestamps per five streams, SMD hashes and global timestamp reference
matched; 50000 requests totaling335571760000 bytes. Full335GB source/copy SHA and
200-event raw-pixel acceptance are on-node gates before performance runs.
The references are copied unchanged from historical scratch campaign; they and
historical source reports are present in this checkout. Frozen install is B at
/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-integration-20261026-r1;
exact active checkout hook interfaces exist for inspection. You cannot access
external scratch/live hardware from this restricted review; report limitations.

Historical rates used localNVMe and weaker cold<1%/warm>99% gates. New run adds
Weka and strict zero/>=99.9% per-file gates; runtime/build/node/CPU placement
comparisons must not be claimed as causal regression evidence. New psana
workload is parse+dense gather, no calibration/user kernels. Standalone is kernel
free. Source owned Weka prefixes unchanged except node cache eviction. Private
local copy is generated and removed inside bounded Slurm allocation. No shared
calibration/dataset mutations. One GPU UUID and common physical-core mask; 6GiB
perBD budget. Queued job code has not changed since submission. Scope 20cases,
max60min allocation, no GPU rates yet. Review outcome only readiness, task
completion still requires GPU acceptance and independent result review.

No nested review/delegation/tools beyond Read/Glob/Grep. Do not claim to run
hashes, tests, live scheduler or GPU commands. Stable finding IDs; distinguish
blocking execution defects from optional improvements. Max three rounds.

Execution update after the documentation commit: job43983799 started on sdfampere038 and is copying/hash-checking the inputs. No throughput results yet. Review remains readiness only.
