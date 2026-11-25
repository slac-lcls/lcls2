# Single-node MPI versus threaded POSIX reads

Prepared 2026-11-25 following the independent CXI validation (job 61982555).
That test used one standalone C++ process, 20 or 40 threads, five 8 GiB files,
stripe count 8 and stripe size 1 MiB, all on OSTs 0–7. Cold median throughput
was 8.333 and 13.669 GiB/s; durations were 4.80–4.82 and 2.84–3.01 seconds.
The 40-thread run used approximately 40/40/20/0 percent of RX on cxi0–3.

## Question and controls

Does splitting a fixed forty read workers across more processes improve
single-node throughput? Compare 1 rank × 40 threads, 2 × 20, 4 × 10, and 8 × 5.
MPI provides rank IDs, barriers, and reductions. Payload I/O remains independent
buffered POSIX `pread`, with 1 MiB requests and a private FD per worker.
This is not MPI-IO or collective I/O, and does not use GPU memory or processing.

Every rank reads a disjoint contiguous fraction of each of the five files.
Its threads subdivide that fraction. Each byte is read exactly once globally.
Keep the same forty physical cores, the same file/offset-to-core mapping,
request size, number of outstanding synchronous worker reads, and input files
for every process count. Buffers are allocated and touched on their pinned
worker cores. Record actual CPU placement and per-worker CPU time.
Randomize process-count order separately in each of three repeats.

The original threaded reader already had independent FDs and disjoint ranges;
there is no shared seek offset or Python GIL in its read path. More processes
could change process-level contention or kernel scheduling behavior. They may
also add overhead. They share the node's page cache and Lustre client resources.
Do not assume one MPI rank gets its own storage NIC or bandwidth allocation.
NIC imbalance is a reason to measure, not proof of a process bottleneck.

## Data and expected duration

Two separate within-dataset comparisons, 12 cases each:

| Dataset | Bytes / GiB | At 13.669 GiB/s | At 20 GiB/s |
| --- | ---: | ---: | ---: |
| Original five-file probe | 42,949,672,960 / 40 | 2.93 s | 2.00 s |
| Existing 400k count-8 copy | 1,343,281,253,664 / 1,251.028 | 91.52 s | 62.55 s |

No new payload copy is needed. The larger dataset exceeds single-node RAM and
gives sustained samples even if throughput improves. Both use count 8 and
1 MiB stripes, but the data, file sizes, and OST placements differ. Compare
rank-count speedups within each dataset; their absolute rates do not isolate
dataset-size effects. Client cache eviction requires two zero-residency scans
on every rank before the first MPI barrier. Backend caches remain uncontrolled.
Do not run another benchmark on these same datasets/OSTs concurrently when
making controlled performance comparisons.

## Correct node-wide CXI measurement

One sampler on rank zero measures all four NICs once. All ranks finish cache
checks and enter a barrier. Rank zero emits READY, receives the sampler's GO
after the initial snapshot, and all ranks enter a second barrier before reads.
Every rank joins its workers, then enters a final barrier. Only after the last
rank arrives does rank zero emit DONE for the end snapshot. Reductions and
output follow. Samples stay in memory during the read and are saved afterward.

Report payload / maximum rank read duration, one node-wide RX/payload ratio,
RX / counter-window duration, per-NIC shares, samples, and barrier/sampling
margin. Do not sum one copy of the same node counters per MPI rank. This scope
is supported by the [CXI provider documentation](https://ofiwg.github.io/libfabric/main/man/fi_cxi.7.html),
which describes these telemetry files as per-interface counters.
The reader rejects a launch spanning more than one shared-memory node.

Job 61988765, the prior four-node sweep, failed after 14 seconds during its
two-node smoke, before any measured cases. Rank 1's local handshake pipe broke;
its MPI process aborted. The new reader duplicates its control FD before
MPI_Init, avoiding dependence on MPI preserving standard input. A regression
test explicitly redirects stdin and verifies the preserved control pipe.
The precise MPI redirection mechanism and actual multi-rank behavior still
require validation on compute nodes. Frozen earlier bundles are preserved.

## Validation and launch

Built production MPI, checksum MPI, and non-MPI test binaries with
`-Wall -Wextra -Werror`. Three local tests passed: exact byte/checksum/range
coverage over 15 simulated rank executions, invariant offset-to-CPU mapping
across 1/2/4/8 ranks, preserved handshake under stdin redirection, and invalid
host/core placement rejection. All ten input identities/layouts and source
pins passed metadata preflight. No new performance result is claimed.

Each allocation runs checksum-validated MPI/CXI smokes at 1/2/4/8 ranks on tiny
fixtures before the 24 performance cases. The MPI checksum executable is used
only for fixtures; production reads do not checksum the payload.

The helper defaults to a dry run. An explicit submit requests one exclusive
GPU node in the interactive queue, account lcls_g, for 90 minutes. It checks
Iris, freezes a hashed source/binary bundle, and refuses existing attempts.

```bash
cd /global/u2/m/monarin/lcls2_worktree/features/psana2-gpu/notes/lustre_multinode_20261125/implementation/cxi_single_node_mpi
bash build.sh
READER_TEST_TMP=/pscratch/sd/m/monarin /pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python -m unittest -v test_comparison
/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python submit.py
# Add --submit to launch this comparison.
```

Accept a completed run only with all 24 audited cases, passing real MPI smokes,
complete.json and successful Slurm accounting. Analyze per-dataset medians and
spread, speedup relative to that run's 1-rank baseline, RX/payload agreement,
per-NIC shares and CPU time. A speedup is measured evidence, not guaranteed by
the use of MPI. Current state: prepared and locally tested; not submitted.

The previous multi-node stripe sweep is paused by explicit user instruction.
The scheduler was checked with no active or pending user jobs. This bundle is
prepared only; neither this comparison nor the previous sweep was submitted
as part of the preparation request.
