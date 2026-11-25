# Perlmutter single-node MPI read comparison — November 25, 2026

Splitting forty C++ read workers across 1/2/4/8 MPI processes produced little
throughput change in this allocation. The 400k sim10 dataset reached a best
median of **9.867 GiB/s at four ranks × ten threads**, compared with
**9.700 GiB/s at one rank × forty threads**. The smaller synthetic probe reached
14.172–14.605 GiB/s across the one- and eight-rank medians. These are descriptive
three-repeat results, not an established storage ceiling or proof of a bottleneck.

## MPI ranks × threads

Each cell reports the median and observed minimum–maximum over three repeats.
Both datasets used the same exclusive node, nid001252, in job **61990131**.

| MPI ranks × threads | 40 GiB probe median (range), GiB/s | 1,251 GiB sim10 median (range), GiB/s |
| --- | ---: | ---: |
| 1 × 40 | 14.172 (13.996–14.554) | 9.700 (9.656–9.788) |
| 2 × 20 | 13.890 (13.786–14.686) | 9.746 (9.544–9.839) |
| 4 × 10 | 14.377 (13.704–14.386) | 9.867 (9.823–9.879) |
| 8 × 5 | 14.605 (14.243–15.029) | 9.764 (9.745–9.949) |

On sim10, ratios of medians relative to one rank were +0.47%, +1.73%, and
+0.66% for 2/4/8 ranks. These are **ratios of separately computed medians**,
not paired speedups or multi-node scaling efficiencies. Forty total workers
and one node were fixed. See [all 24 measurements](runs.csv) and
[exact summaries and NIC shares](summary.json).

## Workload, layout and timing

- One exclusive Perlmutter GPU node, 128 allocated logical CPUs; forty read
  workers pinned to distinct physical cores. Four GPUs allocated, zero used.
  CPU/GPU model inventory was not separately captured for this allocation.
- Standalone C++ buffered POSIX `pread`, 1 MiB requests, private FD per worker.
  MPI supplies rank partitioning, barriers and reductions; no MPI-IO,
  psana decoding, detector processing, GPU transfers, KvikIO or GDS.
- Each rank reads a disjoint contiguous fraction of every file; threads
  subdivide that fraction. Every file byte is read exactly once globally.
  File/offset-to-core assignment stays fixed when rank boundaries change.
- Each dataset has five files, stripe count 8 and stripe size 1 MiB.
  The probe is **42,949,672,960 bytes (40 GiB)**, five equal 8 GiB files,
  all using OSTs 0–7. Each file was fully written with its own random 8 MiB
  block repeated 1,024 times, then flushed and fsynced. Contents therefore
  repeat; this is not 40 GiB of independent random data or an XTC fixture.
- Sim10 is **1,343,281,253,664 bytes (1,251.028 GiB)** from the existing count-8
  copy of the 400k five-stream dataset. File sizes are approximately
  195.475 / 234.568 / 273.662 / 273.662 / 273.662 GiB. These files use
  different OST sets from the probe. Exact input identities and per-file
  stripe placements are preserved in the frozen plan.
- Three repetitions per dataset; process-count order is deterministically
  randomized within each repeat. Probe cases precede all large-data cases.
  The datasets are separate cohorts, not a controlled dataset-size comparison.

Every rank evicts client file-cache pages and verifies two zero-residency scans
before the first barrier. Backend/server caches are uncontrolled. Timed reader
throughput is total file bytes / 2**30 / maximum rank duration from thread
release to last worker completion. It excludes MPI startup, cache preparation,
final reductions and result writing. Probe read durations were 2.66–2.92 seconds;
sim10 durations were 125.75–131.08 seconds. Whole-case durations were not recorded
by the harness and are blank in runs.csv; scheduler step durations include setup
and are retained separately.

## CXI observations

One Python sampler on rank zero reads all four CXI interfaces every 250 ms.
The initial snapshot precedes a barrier releasing all ranks; the final snapshot
follows a barrier after all ranks finish their workers. Samples remain in memory
during the read. Counters are sampled once for the node, avoiding multiplication
by the number of local MPI ranks. NIC rate uses its own counter-window duration;
barrier and sampling margins remain recorded separately from C++ read time.

RX/payload was about **1.0173** for the probe and **1.0273–1.0301** for sim10.
RX shares across cxi0–3 stayed approximately **40/40/20/0%** for the probe and
**34/44/19/3%** for sim10. More ranks did not substantially redistribute traffic.
Hardware bytes include protocol/control/background traffic; excess was not
fully attributed and is not a universal correction factor. Network bytes support
network-crossing reads; they cannot distinguish server-cache service from flash.

The lower sim10 rate is not explained by this experiment. File sizes, OST
placement, content pattern and read duration differ; cache/reclaim behavior may
also differ.
Equal workers per unequal-sized file may create a finish-time tail. Neither
that mechanism nor server-cache effects were independently isolated. The small
probe originated as counter-validation data, not a peak-bandwidth qualification.

## Provenance and validation

Slurm reports **COMPLETED**, **0:0**, elapsed **00:41:13** on **nid001252**.
All four 1/2/4/8-rank MPI/CXI checksum smokes and 24 production read audits passed.
Archiving replayed the exact frozen acceptance code against all 28 cases,
including raw per-rank ranges, byte accounting, CPU affinity, cache gates,
MPI reductions and CXI samples, and checked the table arithmetic.
This replay is not an independent results review; independent review remains
**not performed**. Small changes from three repeats remain descriptive.

The harness was untracked source in the implementation worktree. Its exact
executed files and binary hashes, rather than the results-branch commit, identify
the measurement. The compiler receipt records GCC 14.3.0. The Python executable
was the existing Python 3.11 environment at
`/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python`.
A full native runtime/library inventory was not captured; do not infer one from
current login-node modules.

- [Manifest](manifest.json): configuration, status axes and artifact hashes.
- [Frozen source and pin manifest](../../../../../../notes/archives/cpp-single-node-mpi-20261125/bundle/): exact executed text files;
  native binaries are retained at the original launch location, not in Git.
- [Native completion receipt](../../../../../../notes/archives/cpp-single-node-mpi-20261125/complete.json),
  [UTC scheduler accounting](../../../../../../notes/archives/cpp-single-node-mpi-20261125/accounting-utc.txt), and
  [archive replay receipt](../../../../../../notes/archives/cpp-single-node-mpi-20261125/audit.json).
- [Raw artifact index](../../../../../../notes/archives/cpp-single-node-mpi-20261125/raw-index.json): locations, sizes and SHA256s.
- [Probe generation receipt](../../../../../../notes/archives/cpp-single-node-mpi-20261125/probe-dataset.json): repeated-block hashes
  and original stripe output.

Full raw per-rank timelines and worker records remain at
`/pscratch/sd/m/monarin/psana2-gpu/lustre-cxi-single-node-mpi-20261125/attempt01/`.
Scratch may expire; this compact archive preserves the table and source, but
cannot reconstruct missing raw timelines from hashes. The frozen launch bundle
remains under `notes/lustre_multinode_20261125/implementation/cxi_single_node_mpi/launch-attempt01/bundle/`
in the Perlmutter implementation checkout. Archived launchers retain their
original absolute paths and must not be executed as a relocated installation.

The earlier multi-node sweep remains paused. Its failed allocation and the
separate initial counter-validation job are not pooled into this table.
