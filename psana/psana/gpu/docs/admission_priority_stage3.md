# Admission priority: Stage 3 validation (SDF, 2026-09-16)

Production revision: `e6affe6bf`, following ranking commit `4a26bc640` and
independent test fixes `599f856ae`. No production source changed in Stage 3.
New/updated diagnostic scripts are source-tree scripts and remain uncommitted.

## Result

Correctness and memory-budget acceptance passed. Small-dgram-first admission
does what was intended: it preserves the frequent small-dgram input even when
that stream has the larger total footprint. It is **not** a minimum-request
algorithm or a demonstrated general throughput improvement. On these contiguous
warm inputs, it reduces reads of the small stream but increases the total read
count in the mixed-rate cases. No policy change was made in response.

GPU job `38395141` on `sdfampere040` passed **all 18 integration cases**, including
eight slow, pixel-exact DataSource cases, in 454.06 seconds. These cover GPU and
hybrid routing, raw/calibration values, ordering, parser locators, input lifetime,
and memory admission. Log: `gpu-priority-accept-38395141.log` at the checkout root.
This is CPU fallback, not verified GDS. The previous CPU validation remains
365 main-suite cases and four byhand MPI cases passed; Stage 3 changed no core
modules or automated test cases.

## Controlled comparison

Script: `../scripts/compare_admission_priority.py`. Job `38395749`,
`sdfampere027`, one process/one A100-SXM4-40GB, two execution slots, CuPy 13.6.0,
CUDA runtime 12.9, KvikIO 24.08.02 in compatibility mode, one KvikIO thread,
4 MiB task size. Log: `gpu-priority-compare-38395749.log` at the checkout root.

Only `_resident_candidates` ordering is replaced temporarily inside the
benchmark process. Both policies use the current production fit checks,
diagnostics, read planner, parser, detector kernel, scheduler, and leases.
There is no public policy selector. All benchmark dgrams are nonempty, so the
comparison does not depend on the old/new treatment of empty-only streams.

Inputs are valid synthetic XTC records based on the xpptut chunking fixture:
1000 small-stream events and ten large-stream events, one every 100 events.
An opaque Data sibling pads the complete dgram to the stated size; the actual
decoded/calibrated field is the small xpptut array. This isolates admission and
input handling; it is **not** a full-sized Jungfrau calibration benchmark.
Files are generated under this checkout on SDF and then read warm. Setup and
file generation are outside timing. Correctness passes for both policies warm
the kernels and files before five alternating-order timed samples, each replaying
ten batches. No host field copies occur in timed loops; end-of-input drains
are included. Full CPU-reference field/calibration comparisons and owner-release
checks run separately before timing. Per-stream byte totals and budget limits
are asserted in every sample.

| Case / dgram sizes | Policy | Resident | Reads per batch: small + large | Execution subbatches per batch | Median time / ten batches |
| --- | --- | --- | --- | --- | --- |
| Mixed: 100 KiB / 7 MiB | Total footprint | large | 4 + 1 | 4 | 1.4814 s |
| Mixed: 100 KiB / 7 MiB | Mean dgram | small | 1 + 5 | 5 | 1.5068 s |
| Compact mixed: 16 KiB / 64 KiB | Total footprint | large | 3 + 1 | 3 | 1.0898 s |
| Compact mixed: 16 KiB / 64 KiB | Mean dgram | small | 1 + 5 | 5 | 1.1335 s |
| Equal rate: 100 events each, 100 KiB / 1 MiB | Both | small | 1 + 50 | 50 | 0.8717 / 0.8708 s |
| Tight budget: mixed inputs above | Both | neither | 30 + 10 | 30 | 1.8220 / 1.7605 s |

For the mixed case, total-footprint order retains 70 MiB of large-stream input
and reads the small stream in four already-coalesced spans. Mean-dgram order
retains 97.656 MiB of small-stream input and reads two large dgrams per execution
in five spans. Thus the new policy does not replace 1000 tiny reads in this
example: the old execution path already coalesces them into four larger reads.

The mixed and compact-mixed median times are about 1.7% and 4.0% longer under
the new policy. Sample ranges overlap (mixed: old 1.4410-1.5026 s, new
1.4806-1.5241 s; compact: old 1.0636-1.1285 s, new 1.1059-1.1534 s). Identical
tight-budget schedules also show timing variation. Treat these as bounded
observations, not a robust filesystem/GDS performance conclusion.

All committed-plus-held ledger peaks stayed within the configured budgets:

| Case | Budget, MiB (including fixed setup) | Old peak, MiB | New peak, MiB |
| --- | --- | --- | --- |
| Mixed | 126.4897 | 126.3455 | 126.4897 |
| Compact mixed | 16.7085 | 16.6941 | 16.7085 |
| Equal rate | 13.8575 | 13.8575 | 13.8575 |
| Tight mixed | 14.2046 | 14.1958 | 14.1958 |

The synthetic harness deliberately uses exact quotas and zero additional margin;
the production DataSource trace still uses its normal 10% headroom. Ledger
peaks include reserve/hold calls, not device-wide CUDA or pinned-host memory.
NIC counters, internal KvikIO task counts, and cold-filesystem throughput were
not measured. The JSON log includes request sizes, execution ranges, reader
wait/issue-to-complete statistics, and installed-module SHA-256 hashes.

An earlier short-sample job `38395704` also passed. Initial job `38395380`
exposed a benchmark-harness omission: execution-only inputs need the normal
end-of-input EventPool drain. The harness was corrected; production was not
changed. The longer job above is the authoritative timing record.

## Compact real-data trace

Updated script: `../scripts/trace_bulk_reads.py`. It separates:

- setup and detector/stream ownership;
- batch-level stream statistics and residency decisions;
- byte-bounded execution subbatch calculations;
- actual per-stream psana read submissions and estimated KvikIO task-size pieces.

At 1 GiB, job `38395705` on `sdfampere027` consumed 30 run-51 events in three
ten-event batches. Setup cost was 640.065 MiB; headroom 102.400 MiB; admission
capacity 281.535 MiB. Epix stream 2 retained 10.483 MiB of input+parser tables.
The remaining allowance was 135.526 MiB per execution, fitting one event with
128.087 MiB of nonresident input+parser+detector work. All five Jungfrau streams
were deferred to execution, not discarded or allocated beyond the quota.

Each batch made one epix pread covering ten dgrams, plus ten single-event reads
for each of five Jungfrau streams: **51 psana preads**. Requested/completed
bytes matched at 346,386,000 per batch. Sampled committed peak was 906.722 MiB.
All 18 candidate fit expressions and three batch byte/read totals were checked.
The log is `trace-bulk-sdf-38395705.log` at the checkout root.

Job `38395770` on `sdfampere033` passed the 1.5 GiB control with 13 events:
one ten-event batch and one three-event final batch. All six streams were
resident in each batch, with six preads per batch and no execution input reads.
Requested/completed bytes were 346,386,000 and 103,915,800 respectively. The
log is `trace-bulk-sdf-38395770.log`. Its sampled second-batch high-water includes
the prior batch's cached capacity before trimming; it is not the new partial
batch's required allocation.

One 10.313 MiB epix pread corresponds to three **nominal** 4 MiB KvikIO pieces.
These are estimates, not observations of the library's internal tasks or OS
syscalls. The trace explicitly prints this distinction and the runtime task
size, thread count, threshold, and fallback/GDS mode.

Reproduce the compact trace from the checkout root:

```bash
sbatch psana/psana/gpu/scripts/run_trace_bulk_reads_sdf.sbatch --memory-gb 1
```

Add `--verbose` for physical ranges and allocation reservations;
`--show-events` restores individual event summaries. The default still reads
and checks all selected host fields, but prints only the first event and final
aggregate values. This trace is diagnostic, not timed benchmark evidence.

On an allocated SDF GPU, after sourcing `setup_env.sh` and
`install_psana/activate.sh`, the comparison command is:

```bash
export PS_PARALLEL=none OMPI_MCA_btl='^smcuda' TMPDIR=/tmp
python -u psana/psana/gpu/scripts/compare_admission_priority.py --repeats 5 --batches 10
```

## Remaining limitations

The requested priority policy is implemented and has passed correctness
acceptance. True-GDS, cold-storage, multi-BD, and production mixed-rate
throughput remain unmeasured. No universal speedup is claimed. Optimizing total
read requests, partial residency, or changing the admission objective would be
separate design work requiring review, not hidden Stage 3 changes.
