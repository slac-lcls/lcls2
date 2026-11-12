# Input-group submission and retirement bookkeeping

A batch with 100 events across five physical XTC files can issue 500 separate
input reads. Previously each group submission polled all retained groups and
linearly checked slot occupancy. With two CUDA execution streams, the previous
batch's retained windows were repeatedly visited while submitting the next batch.

The controller now submits the admitted plan through `issue_groups`. The pool
validates group keys, slots and small-stream credits before I/O. A maintained
busy-slot set replaces repeated occupancy scans. Closed windows with no remaining
planned/live references enqueue one retirement notification; polling checks only
these candidates. CUDA-completion checks still gate reclamation, and failures
preserve ownership for retry. Partial submissions remain pool-owned and drainable.
The BD host thread owns pool state for both CUDA streams; consumer threads only
enqueue notifications. This does not make the pool a concurrent host submitter.

## Matched Perlmutter result

The original implementation was validated on the research branch at
`b95b789ca834e6746870548274d119de09ef86ee`, before extraction onto the current
`features/psana2-gpu` target. These numbers describe that historical before/after
comparison. Subsequent measurements of the extracted PR runtime are reported
separately below; they do not repeat the before/after comparison.

Job 61098604 on nid003900 used one BD, one A100-SXM4-40GB, 10,000 Sim10 JF events,
batch 100, execution depth 1 or 2, 512 KiB bulk target and a 6 GiB budget. KvikIO
26.10 used CPU fallback, 8 workers, 1 MiB tasks, 16 MiB bounce buffers and automatic
direct I/O disabled. Each case made 50,000 reads totaling 33,582,000,000 bytes and
100 callbacks of 100 events. Same GPU, CPU affinity and NUMA policy across cases.

Two clean repetitions per combination, forward/reverse order, gave these medians:

| CUDA streams | Before GiB/s | After GiB/s | Before seconds | After seconds |
|---|---:|---:|---:|---:|
| 1 | 2.2302 | 2.2871 | 14.0240 | 13.6756 |
| 2 | 1.2656 | 2.1841 | 24.7121 | 14.3197 |

Two-stream throughput improved 72.6%; its new rate remained 4.5% below one stream.
Separate instrumented runs reduced pool polling from 50,101 to 101 calls and
two-stream polling self time from 9.22 s to 0.18 s. Input submission bookkeeping
fell from 2.71 s to 0.47 s, including both new submission wrappers and excluding
nested reader work. Individual candidate visits were not counted. Physical read
counts/sizes and callback counts/sizes were unchanged; reader pending-list scans
remain outside this change.

All 12 clean/timing cases and 31 native GPU ownership/transition/task tests passed
on the original implementation. The exact extracted PR tree passed 1,023 CPU unit
tests using checkout Python over the existing SDF native extensions. The target
already includes the corrected run-scoped-constants transition assertion.

Cache preparation established zero local page residency immediately before the
loop; backend/network cache state was not measured. Input stayed on Lustre.
The task checked descriptor staging, with no dense gather, blow-up, calibration,
cube or user payload D2H. Timing instrumentation adds overhead; throughput uses
clean runs only. Two repetitions do not establish narrow uncertainty bounds.

## Extracted PR runtime: matched Ada stream comparisons

Both November 10 S3DF jobs used PR #298 at
`67c327a148a08b2aaf972e67e1e02e367c2b298f`, not the research runtime. The jobs
completed successfully on `sdfada016`, using the same physical NVIDIA L40S and
CPU cores 0-23. Each compared one versus two CUDA streams with two full
repetitions per configuration, in order 1, 2, 2, 1; 1,000-event pilots are
excluded from these medians.

| Job | Events per full pass | Requested JF GiB per pass | One stream GiB/s | Two streams GiB/s | Two-stream repeats GiB/s |
|---|---:|---:|---:|---:|---|
| 44185752 | 10,000 | 31.276 | 3.409 | 3.269 | 3.264, 3.275 |
| 44189224 | 100,000 | 312.757 | 3.019 | 2.870 | 2.862, 2.878 |

Controls: one BD/one GPU, batch 100, SMD 1,000, 512 KiB bulk target, 6 GiB GPU
budget, a 24-physical-core allowance and NUMA interleave. KvikIO 26.10 used CPU
fallback, 8 workers, 1 MiB tasks, 16 MiB bounce buffers and automatic direct I/O
disabled. CuPy was 14.0.1 and the CUDA runtime was 12.9. These are descriptor
staging measurements, not dense gather, blow-up, calibration, cube or payload
D2H measurements. The loop timer includes lazy framework setup but excludes
input staging, tests and cache conditioning.

Each 10k pass completed 50,000 reads / 33,582,000,000 bytes / 100 callbacks of
100 events; each 100k pass completed 500,000 reads / 335,820,000,000 bytes /
1,000 callbacks of 100 events. Both jobs passed 31 native GPU tests with zero
failures or skips, the automatic evidence audit and independent results review.
Event identities, complete draining, and absence of CPU payload reads and
pressure drains were checked. Temporary FFB inputs were removed after each job.

PR Python ran over existing SDF native extensions, not a fresh PR build. The
runtime receipts checked 225 native source files against the existing build and
verified loaded module paths/hashes. The sole build-configuration difference
was the packaging-only `gpu/notes` installation exclusion in `psana/meson.build`;
it changed no extension sources, compiler flags or link inputs.

JF files were verified at zero local page residency after setup and immediately
before each loop. Physical NIC RX was approximately 1.056 times requested JF
bytes for 10k and 1.057 times for 100k, corroborating network-served Weka FFB
reads, not a precise cache-miss fraction. SMD files were not cold-gated, backend
cache state was uncontrolled, and no NVMe input staging was used.

These runs establish GPU acceptance and post-fix throughput for the extracted
PR runtime. They have no pre-fix control on Ada and do not independently confirm
the historical 72.6% Perlmutter improvement. Hardware and filesystem differences
also prevent treating the Ada and Perlmutter rates as a before/after pair. Two
repetitions per configuration remain insufficient for narrow uncertainty bounds.

## Reproduction and evidence

After building/installing this checkout and activating its matching runtime:

```bash
PS_PARALLEL=mpi python -m pytest -q psana/psana/tests/gpu/unit
```

Native ownership acceptance requires a CUDA node. The original campaign's
`gpu_tests.py` lists the six integration files; `campaign.py` and `run.sbatch`
record the matched benchmark launch and fail-closed validation. They pin the
historical runtime and are not generic launchers for a newly installed PR tree.

See the [historical implementation, harness, receipts and independent review](https://github.com/monarin/lcls2/tree/3c3799829c659d739d5927511f86cc968a6fae6e/notes/sim10_input_bookkeeping_20261110).
The extracted PR runtime's later reports and receipts are pinned separately:

- [10k Ada results, job 44185752](https://github.com/monarin/lcls2/tree/f888b316e8d940be6cad683d9dc62110938f006f/notes/sim10_ada_baseline_20261110).
- [100k Ada results, job 44189224](https://github.com/monarin/lcls2/tree/f888b316e8d940be6cad683d9dc62110938f006f/notes/sim10_ada_100k_baseline_20261110).

Each Ada directory contains `results.json`, `runtime-evidence.json`,
`gpu-tests.json`, `audit.json`, `artifact-index.json`, cleanup and Slurm receipts,
the pinned campaign harness and independent review. The 10k directory also
contains `build-config-diff.json` for the native-build reuse qualification.
On November 12, all 35 indexed raw artifacts (18 for 10k, 17 for 100k) and all
13 recorded runtime modules per campaign were rehashed successfully. This was
an evidence-integrity check, not a new benchmark or GPU test run.
