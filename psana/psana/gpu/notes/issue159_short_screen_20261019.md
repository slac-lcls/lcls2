# Issue 159: short staged 4BD/1GPU screen

Status: retry **41337036** completed successfully on `sdfampere030` in **41m13s**.
All 16 timed samples and eight diagnostic processes passed. The aligned bulk-on
pairs show **+0.98% / +1.30% loop time**, so the performance gate remains open.
First attempt **41336758** on `sdfampere026` failed its initial sanitizer
diagnostic before timing. This screen measures
stage-2 dense gathering before copy-retention or field-view implementation.
The [stage-2 record](issue159_stage2_20261019.md) documents the kernel timings
and outstanding broad-test failures; this campaign does not waive those failures.

## Fixed comparison

- A0: frozen A, `90c05965cf34869388cc526a50e7e822b50a6eba`, aligned input.
- B0: independent stage-2 build, `7f29f0dc70d3f636f465805afd992c945d39da28`, aligned input.
- B1: the same candidate build, input read directly into a one-byte-shifted view.
- Each case reserves N+1 bytes, exposes N bytes, and charges the actual rounded
  allocation through its last alias. No input repacking or additional image
  buffer/copy is introduced. The adapter is benchmark-only; the frozen installs
  remain unchanged.
- Same physical A100 GPU 0, four BDs plus SMD0/EB, common CPU affinity, batch 20,
  depth 1, eight KvikIO workers, 1 MiB tasks, CPU fallback, explicit 6 GiB/BD.
- Full 32-panel Jungfrau, mfx101210926 run 387, five streams, 10,000 events/sample.
  The workload measures read/parse/dense-gather staging without calibration or
  full-image D2H. MPI uses `ob1` consistently across cases to avoid UCX CUDA
  probing during sanitizer diagnostics.

## Schedule and gates

**16 fresh-process timed samples:** warm cache, bulk off/on, two balanced
adjacent pairs per setting for A0/B0 and B0/B1. Each comparison reverses order
on its second pair. Private staged prefixes must match the retained SHA-256
references and exceed 99% residency before and after every timed sample.

Before timing:

1. Verify frozen runtime/dependency/harness hashes (4,424 artifacts in retry 2).
2. Exercise the reader adapter on all three cases: identical capacity rounding,
   growth/reuse, retained-alias accounting, direct KvikIO destination, and final
   budget release. All three checks passed on the allocated GPU.
3. Run B1's two 200-event pixel preflights under Compute Sanitizer on the four
   GPU ranks. The established CPU references compare three full raw arrays.
4. Audit all 10,000 events through B1 in both bulk modes, checking actual
   `owner_base + field_offset` for every gathered row. Require 320,000 odd-address
   rows and 50,000 even-sized/even-offset input dgrams in each audit. Those even
   extents preserve natural payload parity under different BD/group distributions.
5. Run A0/B0's four 200-event pixel/address preflights, verifying actual aligned
   source addresses before the old kernel dereferences them.

Audit metadata downloads, synchronization and sanitizer are confined to
untimed diagnostic processes. Timed cases retain the same allocation adapter
and existing benchmark counters, with the gather-audit hook disabled.

All existing timestamp, request/byte, rank/device/peer, memory-budget and drained
resource gates remain enabled. The two full audits supplement the six preflights;
they do not add throughput samples or an A/A measurement.

## First-attempt diagnostic and retry

Attempt 1, job **41336758**, passed adapter checks and staged/hash-verified all
five files. Its first B1/off preflight returned exit 99: Compute Sanitizer
reported `CUDA_ERROR_INVALID_VALUE` from `cuMemFree_v2` inside
`cuFileDriverGetProperties`, called by the benchmark's GDS capability query.
No throughput samples were collected. The failed logs remain in
`short-screen-20261019-r1/`; its private node-local stage was removed.

Retry 2 moves that capability query to separate unsanitized setup processes on
the same GPU, requires CPU fallback, saves `capabilities.json`, and supplies the
verified result identically to all workers. It preserves full memcheck on the
KvikIO read, parser and gather path; it adds no sanitizer error suppression and
changes no production code or frozen install.

## Provenance and results

Artifact root:
`/sdf/scratch/users/m/monarin/gpu-validation/issue159/short-screen-20261019-r2/`.

- `plan.json`, `hashes.json`, `metadata/`: fixed setup and preparation scripts.
- `alignment_screen.py`, `rank_exec.py`: exact scheduler/diagnostic orchestration.
- `A0/scripts`, `B0/scripts`, `B1/scripts`: verified identical benchmark harnesses.
- `job-41337036/`: runtime provenance, diagnostics, individual samples, pairs,
  summaries and GPU telemetry.
- `logs/ab-41337036.log`: campaign progress and failure tracebacks.

The 19 CPU harness checks and balanced-schedule check passed before submission.
The tracked [alignment adapter](../scripts/jf_scaling/alignment_adapter.py) is
frozen identically in all cases. Per-case installations link to the previously
verified independent builds; source and dependency hash inventories are included.

The [machine-readable evidence](evidence/issue159_short_screen_20261019.json)
contains provenance, all diagnostic/timed samples, individual pairs and summary
statistics. Timestamp lists are represented by counts and SHA-256 hashes; full
logs and timestamp lists remain in the artifact root.

### Paired loop-time results

Positive changes mean slower. Each row has only two pairs; the median is their
midpoint, not a confidence interval. Keep both bulk modes and all samples,
including the slow B0 bulk-on control, visible.

| Comparison | Bulk | Pair 1 | Pair 2 (reversed order) | Median paired change |
|---|---|---:|---:|---:|
| B0/A0: aligned fix versus frozen base | off | -3.05% | -9.98% | -6.52% |
| B0/A0: aligned fix versus frozen base | on | +0.98% | +1.30% | +1.14% |
| B1/B0: shifted versus aligned candidate | off | -0.16% | -10.32% | -5.24% |
| B1/B0: shifted versus aligned candidate | on | -16.88% | -4.59% | -10.74% |

The aligned bulk-on slowdown appears in both orders. Two pairs do not establish
statistical significance or its cause, but this is sufficient to hold the agreed
stage-2 performance gate. Bulk-off shows no regression in this screen; its spread
does not justify claiming a reliable speedup. The shifted-input comparisons are
also variable: the first bulk-on B0 control took 24.531s, versus 20.143s for its
second control. Read wait increased across all four BDs in the slow control.
No sample was dropped. B1/B0 includes shifted KvikIO/parser input as well as
gather, and these measurements do not establish that unaligned loads are faster.

### Completed checks and limits

- Six 200-event pixel/address preflights passed, comparing the established three
  complete raw-array CPU references per preflight. A0/B0 each audited 6,400 aligned
  rows per mode; B1 audited 6,400 odd-address rows per mode. All eight B1 GPU-rank
  sanitizer logs report zero errors.
- Two full B1 audits each checked 320,000 odd-address gather rows, 50,000
  even-sized/even-offset input dgrams and 500 gather calls. These establish
  actual source-address parity for the full staged workload.
- Every timed sample processed the expected 10,000 unique timestamps, 50,000
  reads and 335,571,760,000 payload bytes. All staged prefixes were 100% resident
  before and after timing. Rank/device/peer, budget and drained-resource checks
  passed; diagnostic gather hooks were disabled during timing.
- Peak owned/held memory was at most **1.251803 GiB/BD**, below 6 GiB/BD and
  identical across cases within each bulk mode. Reader allocation counts were
  four with bulk off and 400 with bulk on, summed across four BDs; all cases
  included the same one-byte allocation allowance. Observed active GPU clocks
  were 1410 MHz. These checks do not explain the timing variance.
- All **4,424 frozen artifact hashes** verified after the runs. Slurm reports
  `COMPLETED`, exit `0:0`; the job removed its private node-local stage.

Next: investigate the aligned bulk-on signal with focused gather/read-wait
profiling before copy-retention or field-view changes. Any candidate optimization
needs a same-node paired A/B check against the unchanged frozen A, preserving
these results; no additional A/A campaign is needed. The existing broad-suite
failures remain open. This warm-cache staging screen does not establish cold-cache
performance, calibration/IPC behavior, or completion of issue 159.
