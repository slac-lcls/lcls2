# Run the Stage 5 bulk-read trace on Perlmutter

The diagnostic Python driver and SDF/Perlmutter launchers referenced below
remain local, uncommitted artifacts. These commands require those scripts;
they are not reproducible from a clean checkout of this documentation alone.

Stage 5 is committed as `694b9ff2b`. Its validation passed 275 CPU tests and
17 GPU tests (job `58391075`). The first mixed-detector attempt, job `58396614`,
failed during DataSource routing before any GPU reads: the original `ebeamh`
default shares its stream with six other normal detectors, which `gpu_det`
rejects. The corrected defaults below pass real DataSource routing and
Configure field checks. GPU execution with these defaults subsequently passed
on SDF; see the SDF continuation below. A Perlmutter run remains unverified.

From this checkout's root, submit:

```bash
sbatch psana/psana/gpu/scripts/run_trace_bulk_reads_perlmutter.sbatch
```

The launcher activates `codex/psana2-gpu-xtc-parser`, loads CUDA 12.9, and runs
one serial BD on one GPU. Output goes to `trace-bulk-<jobid>.log` in the submit
directory. The Python driver is a source-tree script; it does not need an
installation rebuild. The installed psana modules must contain Stage 5.

Defaults:

```python
DataSource(
    exp='mfx100848724', run=51,
    dir='/pscratch/sd/p/psdatmgr/psdm/mfx/mfx100848724/xtc',
    detectors=['jungfrau', 'epix100_0'],
    gpu_det=['jungfrau', 'epix100_0'],
    batch_size=10, max_events=30,
    gpu_memory_budget_gb=1.5, n_gpu_streams=2,
    gpu_d2h_chunk_size=0,
)
```

`epix100_0` supplies the smaller input; the user loop reads its
`raw.raw` array (704 by 768, uint16). Direct CPU inspection of the first 30 events found
both detectors on the same timestamps at approximately 120 Hz. This dataset
demonstrates small versus large inputs, not a different detector event rate.
The original `s006` stream containing only `epix100_0` is 1,081,424 bytes per
sampled event. Jungfrau has 32 segments across five other physical streams. Stream
IDs printed by psana can be renumbered after the detector filter is applied.

The driver checks actual DataSource stream ownership and selected Configure
fields before creating the GPU manager. To run only this preflight on a login
node, activate the checkout and use:

```bash
source ~/activate_psana_gpu.sh codex/psana2-gpu-xtc-parser
export LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/ws
unset SIT_PSDM_OFFSITE
python psana/psana/gpu/scripts/trace_bulk_reads.py --check-routing
```

This makes no CUDA allocations. `detectors=[...]` selects physical files;
it does not remove other detector owners from a shared file. Therefore
`--fast ebeamh --fast-field ebeamCharge` remains invalid for this exclusive
GPU driver. No production routing restriction has been changed.

Change the budget to observe different admission decisions:

```bash
sbatch psana/psana/gpu/scripts/run_trace_bulk_reads_perlmutter.sbatch --memory-gb 2
```

The compact driver prints, in this order:

1. Detector-to-stream ownership and segment ordering, once at setup.
2. One-time setup cost, total budget, headroom, and admission capacity. The
   one-time cost is not recharged for every batch.
3. Each **batch**: requested versus actual event count; per-stream present and
   nonempty counts; minimum/mean/maximum full-XTC-dgram size; total input and
   parser cost. A dgram's size includes every segment/payload in that stream.
4. Each candidate's recorded `RESIDENT` or `DEFER TO EXECUTION` decision and
   exact fit arithmetic. Deferred means read when a subbatch is admitted, not
   discarded, and not memory allocated outside the budget.
5. **Execution subbatch** calculation:
   `(admission capacity - resident bytes) // admitted depth`. Each subbatch's
   cost is nonresident input + nonresident parser tables + detector working
   memory. Repeated subbatch patterns are grouped, retaining their event ranges
   and per-stream event counts. Resident data is shared across those subbatches.
6. Each batch's actual read summary, per stream: resident reads once for the
   batch versus deferred reads in named execution subbatches; dgrams per read
   window, psana pread counts and sizes, and total requested/completed bytes.
7. Sampled budget-ledger committed and committed-plus-held peaks. These are
   not device-wide VRAM peaks and do not measure allocator/runtime overhead.

**Psana versus KvikIO:** one coalesced psana physical range produces one
KvikIO `pread` submission. KvikIO may partition that submission using its
effective `KVIKIO_TASK_SIZE` (printed at startup). The driver reports
`sum(ceil(range size / task size))` as **nominal pieces**, not measured internal
task, cuFile, or OS syscall counts. Alignment handling, the small-read shortcut,
and the backend can affect the actual operations. One psana pread does not
necessarily mean one filesystem read.

Use `--verbose` for every physical file/chunk/offset/device-offset range and
allocation-growth reservation, and `--show-events` for each delivered event.
By default the first event shows detector shape/dtype/sum; the final line
reports event count, timestamp endpoints, and aggregate detector sums. The
driver still consumes all host copies regardless of printing options.

Planning itself does not allocate; actual growth reservations still guard
I/O and allocations. Cached storage stays committed until trimmed. Event
indices reset within each batch. Internal batch IDs can skip transition-only
envelopes; they are not event numbers or execution-slot IDs.

The resident decision applies to complete **physical streams**. Depending on
the budget, some or all Jungfrau streams may also be resident. A single
Jungfrau execution subbatch can need multiple physical reads. The driver
prints the scheduler's decisions rather than imposing a detector-specific
schedule or a fixed number of subbatches. Transitions and end-of-run can
produce batches shorter than `batch_size`.

The loop consumes `evt.gpu.get('jungfrau.raw').on_cpu` and
`evt.gpu.detector('epix100_0').field('raw', 'raw').on_cpu`. These are
independent host copies consumed before advancing the iterator. There is no user `calib()` call;
the current pipeline nevertheless eagerly computes Jungfrau calibration
and includes that cost in admission. Logging, host copies, and the raw sum
make this an inspection driver rather than a throughput benchmark.

Tracing subclasses the existing manager and wraps the reader's submission
method within this process. Production scheduling and read planning remain
unchanged. No Stage 6 implementation is included.

## SDF continuation (2026-09-16)

The SDF checkout is at `694b9ff2b`. Its `install_psana` prefix initially lacked
the Stage 1–5 bulk-read modules; an incremental `./build_psana.sh -j 8`
completed successfully. Installed GPU Python modules and the changed DataSource
Python modules were checked byte-for-byte against this checkout.

Run-51 bigdata and smalldata files are available under
`/sdf/data/lcls/ds/mfx/mfx100848724/xtc`. Real DataSource routing and Configure
field preflight passed: Jungfrau uses selected stream IDs `[0, 1, 3, 4, 5]`,
and `epix100_0` uses `[2]`; each selected stream has only its requested detector
owner. These are filtered psana stream IDs, not filename stream numbers.

From the checkout root, preflight and submit using:

```bash
source setup_env.sh
source install_psana/activate.sh
export LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/ws
unset SIT_PSDM_OFFSITE
python psana/psana/gpu/scripts/trace_bulk_reads.py \
  --dir /sdf/data/lcls/ds/mfx/mfx100848724/xtc --check-routing
sbatch psana/psana/gpu/scripts/run_trace_bulk_reads_sdf.sbatch
```

The separate SDF launcher requests one A100 on `ampere` with account `lcls`,
uses this checkout's environment/install, preserves Slurm GPU visibility, and
runs one serial BD with `srun --mpi=none`. It does not change the Perlmutter
launcher. Driver arguments after the launcher filename override its defaults.

All 275 CPU-only GPU unit cases passed with `PS_PARALLEL=mpi`. An initial run
with `PS_PARALLEL=none` had one environment-induced failure because the
MPI batch-source test expects `node.MPI`; no production/test fix was needed.
Logs are `/tmp/gpu_stage5_sdf_build.log`, `/tmp/gpu_stage5_sdf_preflight.log`,
and `/tmp/gpu_stage5_sdf_unit_mpi.log` on the SDF login host.

GPU trace job `38379041` completed on `sdfampere033` with exit `0:0` in 88
seconds, using default 30 events, batch size 10, 1.5 GiB budget, and two
execution slots. The log is `trace-bulk-sdf-38379041.log` in the checkout root.
Results:

- KvikIO reported **CPU fallback**, not true GDS.
- All 30 events were delivered in order with unique timestamps. Both detectors
  returned data: Jungfrau `(32, 512, 1024)` uint16 and epix segment 0
  `(704, 768)` uint16, with nonzero sums.
- Each of three ten-event batches selected all six streams as resident,
  including Jungfrau streams. Each made **6 actual KvikIO pread submissions
  for 60 logical dgrams**, followed by five two-event execution subbatches
  with no further reads. Total: 18 submissions for 180 logical dgrams.
- Requested and completed bytes matched: 346,386,000 per batch,
  1,039,158,000 across the run. The logged budget commitment peaked at
  1,355.420 MiB against a 1,536 MiB limit. This is a sampled budget-ledger
  observation, not a measurement of device-wide peak physical memory.

Startup emitted missing-Kerberos warnings and OpenMPI CUDA shared-memory
diagnostics, but the trace completed. The SDF launcher now disables the unused
`smcuda` transport and sets `TMPDIR=/tmp` rather than inheriting a login-host
scratch path. Repeat job `38379161` completed on the same node with exit `0:0`
in 75 seconds, without those OpenMPI shared-memory diagnostics. All 30 event
timestamps, shapes, dtypes, and detector sums matched the first trace. Its log
is `trace-bulk-sdf-38379161.log` in the checkout root. Missing-Kerberos warnings
remain, but did not prevent these public calibration reads and trace runs.

This is execution/trace validation, not pixel-exact CPU comparison, a throughput
benchmark, true-GDS validation, or a demonstration of different detector rates.
No production runtime changes or Stage 6 implementation were needed.

## Residency-priority trace (follow-up Stage 2)

Stage 1 (`4a26bc640`) prioritizes mean present nonempty dgram size rather than
total stream footprint. Stage 2 adds decision records to the plan and prints
them without recomputing policy in the script. Existing read-count and memory
diagnostics are unchanged. The `fit` expression is:

```text
previously admitted resident bytes
+ candidate input and parser bytes
+ trial concurrency * remaining maximum single-event working bytes
<= admission capacity (fixed setup and safety headroom already excluded)
```

This is a minimum-execution fit check, before final subbatch packing. The trace
prints final subbatch allowances and boundaries separately. An all-empty stream
is not a residency candidate; no absent events enter a candidate's mean. Parser
bytes still account for all supplied descriptors.

SDF validation on 2026-09-16 used the same run-51, 30-event/batch-size-10
trace at two budgets. Both jobs completed on `sdfampere036` using CPU fallback:

| Budget | Job / root log | Resident streams per batch | pread submissions per batch | Highest sampled ledger commitment |
| --- | --- | --- | --- | --- |
| 1.5 GiB | `38389330` / `gpu-priority-stage2-38389330.log` | All six | 6 | 1355.420 MiB |
| 1 GiB | `38389359` / `trace-bulk-sdf-38389359.log` | epix stream 2 only | 51 | 906.722 MiB |

At 1 GiB the trace admits epix, then prints an insufficient-capacity decision
for each of the five Jungfrau streams. Each batch reads epix once and uses ten
single-event executions, each reading five Jungfrau dgrams. All 18 candidate
fit expressions in each log were checked for correct arithmetic and outcome.
Both traces requested and completed 1,039,158,000 bytes across three batches;
all 30 delivered timestamps, shapes, dtypes, and detector sums matched.

Job `38389330` also passed all ten fast CUDA integration cases (eight slow
cases deselected), including the synthetic mixed-rate case where frequent small
dgrams have a larger total footprint than the sparse large-dgram stream. That
case checks field values against the CPU reference and resident storage after
execution-slot reuse. These are correctness checks, not matched-policy
throughput measurements or true-GDS validation; those remain later work.
