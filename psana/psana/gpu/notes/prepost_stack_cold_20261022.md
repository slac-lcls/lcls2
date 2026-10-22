# Pre/post-stack cold comparison — October 22, 2026

Status: job **41614567** submitted; waiting for resources at the last check.
No timed results are available yet. This comparison was requested
following the historical cold-rate audit. It compares the pre/post review stack,
before PR187 and the issue159 alignment changes.

| Arm | Revision | Meaning |
| --- | --- | --- |
| A (`A0` in the harness) | `9b32dda19b46807f7a4db62fb2351ba87b359e47` | Pre-stack |
| B (`B0` in the harness) | `410ebba1ce4d8b2a9d8d7fab56f6dc1fa26a6ef8` | Post-stack |

Both commits have the same complete source tree:
`65b8d582a61a96cafd2e91f03c52b5c910e5e455`. The review stack reconstructs the
same content. This comparison tests independently built endpoints under matched
conditions; it cannot attribute a difference to source code that differs between
these two trees. It also does not measure cumulative PR187/issue159 changes.

## Independent builds and provenance

Fresh detached source worktrees, Meson build directories and install prefixes
were created for A and B under the artifact root below. No earlier frozen
installation was overwritten or reused. Both builds used Release, DAQ disabled,
LTO disabled, four compile workers, C++ compiler 14.3.0 and the same dependency
environment. Current MPI is Open MPI 5.0.8 in both arms.

Verification passed for 28 native `.so` files per installation: corresponding
files have distinct resolved paths and inodes. Both source worktrees are clean.
An A → B → A activation test passed after deliberately poisoning inherited
Python/library/compiler paths. Imported psana modules and loaded project
libraries resolve to the selected installation. All six MPI ranks must repeat
provenance checks before staging and in every sample.

The harness's `screen/A0/install` and `screen/B0/install` links select the two
separate **new** A/B installation directories. They do not share native builds.

## Cold protocol and reporting

- One exclusive Ampere node for the whole job, one physical A100 (GPU 0), four
  BD ranks plus EB and SMD0. Reserve 112 CPUs, 700 GiB memory and all four GPUs.
- Stage the same five hash-verified private input copies once to job-owned
  `/lscratch/monarin/jf-issue159-<job>`. Staged size: 335,656,487,644 bytes.
- Before **each timed sample**, validate private-file ownership, fsync and evict
  with `POSIX_FADV_DONTNEED`; require each measured prefix to be ≤1% resident.
  Post-sample residency is observed without eviction or a cold threshold.
- Four balanced AB/BA pairs per bulk setting: 16 timed samples × 10,000 events.
  Four 200-event pixel/address preflights are excluded from timing statistics.
- JF-only dense raw gathering, no calibration/user kernel; batch 20, depth 1,
  6 GiB budget/BD, eight KvikIO workers/BD, 1 MiB task size, CPU fallback.
- Both arms use the same current harness as the Stage 4 cold screen. This
  includes explicit MPI `pml=ob1`, stronger import checks and the benchmark-only
  zero-shift reader wrapper (request N+1 bytes, expose N). Timed address audits
  are disabled. These are differences from earlier historical harnesses, held
  constant in this comparison; this is not an exact replay of September.
- Report each arm's mean ± **sample standard deviation** (`ddof=1`, n=4) for
  events/s and loop seconds, separately for bulk off/on. Retain individual
  samples and paired loop-time percentage changes. Exclude staging/cache
  preparation and external-library/storage inventory from loop timing.

Read-only storage snapshots before/after samples record disk counters, RAID
membership, device models, read-ahead, cgroup I/O controls, NUMA topology and CPU
affinity. GPU and physical-interface monitoring follow the existing harness.
No host I/O controls are changed. The 90-minute allocation limit leaves room for
staging and preflights; estimate 40–50 minutes after allocation, based on the
recent 43-minute screen (queue time is additional).

## External libraries and historical differences

The login-node A/B inventory has identical paths/hashes for 230 mapped external
libraries. The job repeats the comparison on the compute node, including CUDA
runtime/driver information, before and after the screen. Shared dependency files
used during preparation are frozen with the benchmark manifest.

| Dependency | This comparison, both A/B | October 17 record |
| --- | --- | --- |
| Python | 3.11.14 | 3.11.14 |
| KvikIO / libkvikio package | 26.10.0 (import reports 26.10.00) | Same |
| CuPy | 14.0.1 | Same |
| NumPy | 2.3.5 | Same |
| mpi4py | 4.1.1 | Same |
| Meson | 1.9.2 | Same |
| Cython | 3.2.2 | Same |

All four file hashes recorded in the October 17 dependency metadata still
match: `libkvikio.so`, `cache_probe.so`, and that environment's `env.sh` and
`activate.sh`. The historical metadata does not contain hashes for every loaded
external library, so this is **not** proof that all historical binaries are
unchanged. Current A/B library equality is checked directly.

September used **KvikIO 24.08.02 / CuPy 13.6.0**, whereas this run uses
**26.10.00 / 14.0.1**. The fast October 16 post-stack result already used these
newer versions. No package upgrades or local KvikIO patches were made for this
comparison. GPU driver and loaded CUDA library evidence will come from the job.

## Interpretation and artifacts

If both older revisions reproduce the low absolute rate, that shows the
PR187/issue159 changes are not required to reach that rate under this harness
and node conditions. If both recover the higher rate, node/date effects remain
plausible. Neither outcome by itself resolves the cumulative impact of later
alignment changes. Any A/B difference between these identical source trees
requires checking build artifacts, placement and runtime variation first.

Artifact root:
`/sdf/scratch/users/m/monarin/gpu-validation/prepost-stack-cold-20261022-r1`

- `A/` and `B/`: independent source/build/install trees.
- `metadata/`: build logs metadata, isolation checks, dependency inventories,
  Meson dependency/compiler information and reproducible preparation scripts.
- `screen/plan.json`, `screen/hashes.json`: 2,107 frozen comparison inputs;
  balanced schedule and sample-standard-deviation checks passed.
- `screen/job-<job>/`: per-sample results, pairs, summaries, library inventories,
  provenance, cache gates and system snapshots.
- `logs/driver-<job>.log`, `logs/screen-<job>.log`: job progress.

See [historical cold-rate audit](issue159_cold_rate_history_20261022.md) and
[submission evidence](evidence/prepost_stack_cold_20261022.json).
