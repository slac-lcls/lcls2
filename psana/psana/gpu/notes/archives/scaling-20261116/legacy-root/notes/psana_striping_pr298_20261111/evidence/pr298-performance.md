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
`features/psana2-gpu` target. These numbers describe that measured comparison;
the exact PR-base runtime has not been rebenchmarked.

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

## Reproduction and evidence

After building/installing this checkout and activating its matching runtime:

```bash
PS_PARALLEL=mpi python -m pytest -q psana/psana/tests/gpu/unit
```

Native ownership acceptance requires a CUDA node. The original campaign's
`gpu_tests.py` lists the six integration files; `campaign.py` and `run.sbatch`
record the matched benchmark launch and fail-closed validation. They pin the
historical runtime and are not generic launchers for a newly installed PR tree.

See the [immutable implementation, harness, receipts and independent review](https://github.com/monarin/lcls2/tree/3c3799829c659d739d5927511f86cc968a6fae6e/notes/sim10_input_bookkeeping_20261110).
The receipts include exact runtime hashes, cache checks, event identities,
read/callback counts, cleanup, Slurm exit status and measured phases.
