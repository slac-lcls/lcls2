# KvikIO fallback read baseline for the 10k sample10 fixture

Follow-up: the [physical NIC/cache profile](../mfx_kvikio_network_20261108/README.md) measured **4.02 GiB/s cache-cold** versus **13.96 GiB/s warm**, with network-byte evidence. The 15.62 GiB/s result below is application throughput with uncontrolled cache history, not a demonstrated Weka network rate.

The best confirmed read-only rate is **15.62 GiB/s median** (15.25–15.68) with
four workers, explicit 4 MiB task and 16 MiB bounce-buffer settings, and a bounded
ring of 35 outstanding datagram reads. This is a repeated-read application
throughput ceiling observed on one A100, not physical Weka bandwidth or an
end-to-end psana rate. The first pass of the same configuration was only
4.14 GiB/s; cache/history and external system activity are not controlled.

Job **43866709**, sdfampere012, A100-SXM4-40GB, completed 0:0 in 4m52s. All **33
fresh-process cases** passed read accounting and GPU/CPU byte witnesses. Ten
host tests passed, including ring-buffer ownership, error draining, tail coverage
and rejection of corrupted result metadata. The coordinator independently audited
all native reports, source identity, settings and confirmation arithmetic.

## Data and timing

Inputs are the five JF files s005–s009 in
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sample10_10k/`.
The manifest hash and file sizes are freshly checked against the accepted
fixture receipt. Full-file hashes are historical generation-acceptance hashes;
this campaign does not reread every file merely to hash it before timing.

- Exact event mode: 50,000 requests, 33,582,000,000 bytes = **31.2757 GiB/pass**,
  exactly the GPU input ranges used by the pipeline. The generation manifest
  gives sizes of 524724–734604 bytes each; these extrema are not native report
  fields. See `request-size-evidence.json` for the coordinator's manifest readback.
- Contiguous-file mode: 4/16/64 MiB requests from offset zero through EOF, including
  partial tails, 33,582,860,544 bytes = 31.2765 GiB/pass. The 860544 extra bytes
  are headers/transitions/etc. This is a read-ceiling comparison, not an
  implementation of psana event grouping; it may cross event/transition boundaries.
- CPU ancillary files, SMD reading, parsing, expansion, calibration, cube and
  publications are excluded. There is no psana `n_gpu_streams` setting here;
  KvikIO manages its own worker/copy streams.
- Each case opens read-only handles, preallocates a bounded GPU buffer ring,
  performs five warm reads and joins, then times complete file passes from first
  submission through all futures and a final device synchronization. Python
  issue/wait overhead is included. Setup, warm reads and post-read witnesses are
  outside timing. Full passes repeat until >=5 s measured or three passes.
- Byte-return counts are checked for every request. A buffer is reused only
  after its previous future completes. After timing, start/middle/end 64-byte
  samples from every ring slot's final request match independent CPU `pread`.
  This validates sampled transfer bytes, not every byte of every timed read.

The queue is continuously replenished in FIFO order, round-robin across files;
35 requests represent roughly seven events' JF data. This does **not** replay
the pipeline's parser, callback or admission delays and is not a timing-identical
psana read schedule. GPU buffers are capped at 4 GiB; largest used is 2.5 GiB.
The job allocates 16 CPUs/64 GiB host memory. The control restricts affinity to
four CPUs, other cases use 16; effective thread counts and affinity are recorded.

No data was staged to NVMe/DRAM and no shared cache was flushed. OS and Weka
caches can serve repeated reads. Across the campaign, 2595.92 GiB of repeated
input was measured over 220.40 s of timed windows; no new dataset was written.
The first pass is not a verified cold-cache measurement. Initial GPU snapshot
shows 6373 MiB/100% activity; first-case post-warmup usage is 460.125 MiB. Final
snapshot is 0 MiB/16% (unattributed spot sample). No continuous exclusivity claim.

## Confirmed results

Each row below has three fresh-process confirmation cases in shuffled order,
separate from selection/screening. Each confirmation case completed three full
passes; medians are of complete-case bytes/time, not per-pass peak rates.

| Pattern | Workers | Outstanding requests | Task / bounce | CPU affinity count | Median GiB/s | Range |
|---|---:|---:|---|---:|---:|---|
| Exact datagrams, four-worker control | 4 | 35 | 4 / 16 MiB | 4 | **15.62** | 15.25–15.68 |
| Exact datagrams, screening winner | 8 | 35 | 4 / 16 MiB | 16 | **15.47** | 15.42–15.49 |
| 64 MiB contiguous reads, screening winner | 4 | 10 | 4 / 1 MiB | 16 | **14.61** | 13.36–14.64 |

All confirmed rows use `compat_mode=ON`, `auto_direct_io_read=False`. Settings
are applied through `kvikio.defaults.set` before opening handles and checked via
`defaults.get`, overriding the shared runtime's four-worker environment value
where applicable. Runtime: KvikIO 26.10.00, CuPy 14.0.1, Python 3.11.14, CUDA
runtime 12090. Every case uses the same environment/package/source files.

The four-worker control is already competitive at this read-only ceiling.
The earlier pipeline recorded compatibility mode and four workers but did not
capture its effective task/bounce sizes. The 4/16 MiB values here match the
installed library defaults and are explicitly set/verified in this new test;
they are not retrospective proof of unrecorded historical environment settings.
Neither increasing to eight workers nor combining datagrams into large
reads demonstrated a confirmed gain over the control. The 4- and 8-worker
confirmation ranges overlap; do not treat their ~1% median difference as a
causal advantage. This does not imply bulk coalescing is useless in the real
pipeline, where host overhead and allowed concurrency differ.
The control and eight-worker confirmation differ in both worker count and CPU
affinity count, so this table cannot isolate a worker-count effect. Recorded
affinity values are logical CPU (hardware-thread) IDs; physical-core/sibling
topology was not recorded.

## Screening evidence (single cases, not causal comparisons)

| Workers | Exact datagrams, depth 35 | 4 MiB reads | 16 MiB reads | 64 MiB reads |
|---|---:|---:|---:|---:|
| 1 | 4.78 | — | — | — |
| 4 | 13.53 | 12.79 | 12.83 | 12.91 |
| 8 | 15.13 | 9.72 | 11.27 | 12.30 |
| 16 | 12.84 | 8.46 | 9.08 | 10.83 |

GiB/s, 16 CPUs, task 4 MiB/bounce 16 MiB; contiguous-read depth 10. Screen order
was shuffled with seed 20261108. First control is separate (4 CPUs, 4.14 GiB/s).
Worker count was not monotonically beneficial. Cache/order effects make these
screening values diagnostic; the confirmation table is the primary result.

Other single-case sweeps:

- Exact datagrams/four workers: depth 5/35/140 gave 12.59/13.53/13.81 GiB/s.
- 64 MiB reads/four workers/depth 10/bounce 16 MiB: task 1/4/16 MiB gave
  13.09/12.91/9.06 GiB/s.
- Same 64 MiB base: depth 5/10/40 gave 13.08/12.91/10.68 GiB/s.
- Same base: bounce 1/16/64 MiB gave 13.23/12.91/13.08 GiB/s. These small screening
  differences do not establish an optimal bounce size in isolation.
- Opportunistic POSIX direct reads requested: exact datagrams/eight workers
  gave 5.03 GiB/s, 64 MiB reads/four workers/bounce 1 MiB gave 3.57 GiB/s. Both
  passed validation; neither was selected for confirmation. `O_DIRECT` fds were
  open, but also exist in buffered cases. The effective flag requests the direct
  path; actual direct-vs-buffered byte fractions were not traced. Unaligned
  portions may use buffered I/O. This remains CPU fallback, not GDS, and is not
  a verified cold-storage measurement. No general direct-I/O performance claim.

The legacy report key `input_direct_fd_count` (and `results.json` alias
`direct_fd_count`) means only a **descriptor-flag census**, not actual I/O-path
evidence. It is five in every case. `metric-definitions.json` labels these saved
fields explicitly; native reports are preserved byte-for-byte. Renaming the
keys in the executable/report schema is deferred to a future benchmark version.

This is an adaptive, bounded sweep, not a full factorial or proof of an absolute
hardware maximum. In particular, queue/bounce tuning focused on the selected
large-read case; all combinations of event-mode workers/queue sizes were not tried.

## Implication for the cube benchmark

At 3,358,200 stored JF bytes per event, 15.62 GiB/s is about **4,994 input events/s**
before parsing/expansion/calibration/cube or ancillary work. The 35 kHz target
would require **109.46 GiB/s** of this JF input on the same byte definition.
These are arithmetic equivalents, not measured event-processing throughput.

The earlier ~0.9–1.0 GiB/s figures used all stored XTC/SMD bytes per event;
they are not the same byte denominator as this five-JF-file benchmark. Also,
different nodes, cache history and scheduling prevent calling the ratio a
controlled speedup. The evidence supports investigating pipeline input
submission/concurrency and host preparation before changing KvikIO defaults.
No pipeline settings have been changed by this task.

## Reproduce and evidence

Native results and per-case logs:
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/kvikio_read_20261108_r1/`

Slurm logs: adjacent `kvikio_read_20261108_r1_logs/43866709.{out,err}`.
`results.json` contains all configurations/rates, confirmation arithmetic,
source hash verification and native report SHA256s. Full effective settings,
CPU affinity, memory, per-pass timing and sampled offsets are in each native JSON.

Submit `run.sbatch SOURCE_DIRECTORY NEW_OUTPUT_DIRECTORY` with Slurm stdout/err
directed to an existing scratch log directory. `campaign.py` executes 33 fresh
processes and saves progress after each case. `audit.py NATIVE_ROOT --output
NEW_RESULT_JSON` validates the complete campaign independently. For the ten host
tests, set `KVIKIO_READ_NATIVE_ROOT` and run unittest discovery for `test_*.py`.
Input files are opened read-only. Failed campaigns stop; outputs are not reused.

Independent restricted Claude review passed round 2 on
`869c2912c0ee9bdb5b32e613521ff142bf4575f6`; measured executable code remains
`70359e3b894206f2e9c9f2d2abb4740805a3a87b`. The blocking settings-provenance wording
was corrected without changing measurements. Four findings are closed; the
non-blocking descriptor-census key rename is accepted for a future benchmark
version. See [review closure](review-02/response.md). Reviewer file-only tools
could inspect source/evidence but could not execute, hash or query live storage;
those validations were performed by the coordinator.
