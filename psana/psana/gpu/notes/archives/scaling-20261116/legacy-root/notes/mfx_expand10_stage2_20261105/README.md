# MFX sample10 Stage 2: native GPU correctness

Completed November 5, 2026. Independent Claude review passed, with all findings
closed. Tested implementation: `11f2f804a0f9968d6210d8a6bfed584691b101d3`;
final evidence follow-up: `b0b946f41ef82e3b33fbdf8c64f2a69a565ad1c2`.
See [review closure](review-02/response.md).

The task reads `jungfrau.sim10.seed` through psana's generic GPU field API,
expands all physical panels into contiguous uint16 `(N,32,512,1024)` storage,
then calls the unchanged Jungfrau calibration example. Outputs use normal GPU
result lifetimes and host access. No psana core files changed.

`adapter.py` registers a process-local detector class before DataSource
construction so native detector ownership routes all five JF streams. The CUDA
kernel validates field/locator metadata and bounds, uses byte loads for
unaligned words, and publishes owned raw/error arrays before launching.
Calibration runs on the supplied stream. Borrowed descriptors are not retained.
All 32 panels retain their physical calibration index.

The hardened `fixture.py` and `verify.py` close Stage 1 F-002 and F-004: reject
Data-before-Shapes and require recorded writer/helper/manifest hashes.
Historical Stage 1 files remain unchanged. Trailing uint64 scalars are unaligned
(Stage 1 F-003); the GPU task does not load those fields.

Artifacts are under:
`/sdf/scratch/users/m/monarin/gpu-validation/mfx-expand10/20261105-stage2-r1/`.
`tiny/` contains the first 10 source events, `pilot/` the first 1,000, and
`tail/` the last three (indices 36034–36036). Each has ten BD and ten SMD files,
retaining five JF and five ancillary streams. Tail keeps original closing
transitions; truncated prefixes close using source-derived templates with
ordered timestamps. The source has only one native step.

```bash
notes=notes/mfx_expand10_stage2_20261105
bash "$notes/runtime.sh" "$notes/test_fixture.py"
bash "$notes/runtime.sh" "$notes/verify.py" --root /absolute/fixture --report /new/report.json
# On an allocated GPU, with a new output directory:
bash "$notes/runtime.sh" "$notes/gpu_accept.py" --root /absolute/fixture --output /new/gpu-report --batch-size 7
```

CPU verification compares every seed with original source pixels and checks
stream timelines, source fields, SMD offsets and DataSource traversal. GPU
acceptance checks every expanded raw word and calibrated float32 bit against
CPU v3, with offsets enabled, status masking disabled, and no common mode
correction. It also runs ten direct CUDA descriptor/bounds cases.

Calibration uses frozen, hash-verified `(3,32,512,1024)` arrays. A constructor-
scoped test override supplies constants before RunSerial creates its GPU
manager, then restores the loader immediately. This does not validate live DB
lookup or deploy records. Scratch retains source experiment/run identity.

CuPy 14.0.1 and KvikIO 26.10.0 run with compatibility mode ON. This serial,
one-GPU correctness test copies full results to the CPU for checking each event.
GDS, MPI scaling, cube accumulation and 35 kHz are not tested. Batches are capped
at 16 events with two CUDA streams; the 4 GiB framework budget excludes some
user output/constant allocations. CuPy pool peak is recorded separately and is
not a total device-memory measure.

All final checks passed before independent review:

| Fixture | Events | CPU panel checks | GPU pixels checked | XTC + SMD bytes |
|---|---:|---:|---:|---:|
| Tiny | 10 | 320 | 167,772,160 | 40,280,528 |
| Native tail | 3 | 96 | 50,331,648 | 15,338,533 |
| Pilot | 1,000 | 32,000 | 16,777,216,000 | 3,770,582,982 |

Pilot: 158 callbacks (sizes 1, 2, 5, 6 and 7), two CUDA streams, including one one-event
callback and a six-event final partial callback. Peak CuPy pool allocation was 2,040,194,560 bytes. Every raw
word and calibrated float32 bit matched; all calibration outputs were finite.
Eight focused CPU tests and all ten CUDA guard cases passed.

See `validation-summary.json` for final checks, artifact hashes, file sizes and
batch/memory statistics. `status.json` retains attempts; `review-01/` contains
independent review context, findings and response.
