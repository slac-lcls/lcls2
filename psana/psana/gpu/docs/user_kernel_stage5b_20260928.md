# Stage 5b: external batched calibration and azimuthal integration

**Status:** Implemented and validated. The host suite passed **510 tests**,
the focused A100 suite passed **24 tests**, and all real-data serial/MPI checks passed. Stage 5a is committed as
`d9189bfd9`. No production runtime code changed for Stage 5b.
Stage 5c performance and Stage 6 final lifecycle acceptance remain pending.

## External user example

Copy these three files to a user directory:

- [jungfrau_calibration.py](../examples/jungfrau_calibration.py)
- [jungfrau_azimuthal_integration.py](../examples/jungfrau_azimuthal_integration.py)
- [integrate_jungfrau.py](../examples/integrate_jungfrau.py)

The user modules import NumPy and each other, with no psana imports. The driver
imports public DataSource/GpuTask APIs. CuPy initialization happens only in the
callback on an assigned GPU worker. Constructor work is CPU-only.

```python
analysis = JungfrauAzimuthalIntegration(
    bin_ids, nbins=64, use_offset=True, status_bits=0xffffffffffffffff)
task = GpuTask(analysis, inputs=analysis.inputs, calibconst=analysis.calibconst)
ds = DataSource(..., gpu_det='jungfrau', gpu_fn=task,
                batch_size=5, n_gpu_streams=2)
```

Supply a signed integer `bin_ids` array in original physical segment layout
`(P,H,W)`, with -1 for excluded pixels and 0..nbins-1 for included pixels. The
constructor owns an immutable copy. There is no default tiled geometry or
invented wavelength/distance. The optional `radial_bin_ids(x_mm, y_mm, edges_mm,
center_mm=(x0,y0))` helper builds radius bins from explicit coordinate arrays.
Every interval is `[left,right)`; out-of-range/nonfinite coordinates are
excluded, including coordinates on the last right edge. Empty selections and
empty bins are valid.

For q bins, user code must prepare the map using confirmed beam parameters and
consistent geometry/units. A new geometry or bin policy requires a new callable;
this example does not automatically follow geometry changes during a run.
Calibration constant values are retrieved from the batch on every invocation.

The CLI accepts an explicit `.npz` containing `bin_ids` and increasing finite
`edges` (nbins+1). For a dataset with available offset and pixel_status:

```bash
PS_PARALLEL=none python integrate_jungfrau.py mfx100848724 51 \
  --directory /sdf/data/lcls/ds/prj/public01/xtc --bins radial_bins.npz \
  --events 13 --batch-size 5 --depth 2 \
  --offset --status-bits 0xffffffffffffffff
```

Available `status_extra` can be selected with `--stextra-bits`. Constants remain
explicit requests; missing-constant fallbacks and custom detector masks are not
implemented. The example does not apply common mode, polarization, solid-angle
correction, pixel splitting or background subtraction.

## Numerical and count policy

Calibration retains Stage 5a's CPU-v3 arithmetic. Its new `calibrate()` method
can retain the image as scratch instead of publishing it, and produce a uint8
validity array in the same launch. Standalone `JungfrauCalibration` still
publishes the full float32 image, with the same calibrated-pixel semantics.

A pixel contributes only when its segment is present, its gain code is 00/01/11,
its selected gain is finite and nonzero, its enabled status masks accept it,
and its calibrated value is finite. It must also belong to the explicit bin
map. **Valid zero intensity contributes one to the count.** Masked, absent,
invalid-gain and nonfinite pixels contribute neither intensity nor count.

The output is `(N,3,nbins)` float64: mean intensity, intensity sum, valid-pixel
count. Empty bins have zero in all three rows. Counts accumulate as uint32
and are converted exactly to float64; physical bin maps are bounded to int32
pixel indexing, so counts cannot overflow. Sums accumulate calibrated float32
values in float64. Counts must equal the CPU reference exactly; sums/means use
`rtol=1e-12, atol=1e-9` for the validated fixtures and public data. Calibration
keeps separate exact-pixel tests. A scientifically different dataset may need
an explicitly justified error budget, particularly for cancellation near zero.

## Scheduling, ownership and memory

Every nonempty callback submits two kernels for the whole selected subbatch:

1. Calibration plus validity, over all `(event, segment, row, column)` pixels.
2. Sorted-bin reduction, over all `(event, bin)` pairs, including normalization.

The reduction adapts Amanda Shackelford's `azint_sorted_kernel` from source
`650c76780`: it gathers through sorted indices directly from the calibrated
image, uses dynamic validity for counts, and normalizes in the same block.
There is no separate gather buffer, normalization launch, atomic accumulation,
or per-event launch loop. The public event loop reads the delivered histogram.

Host sort-order/bin-offset preparation and upload happen lazily once per device
and input segment layout. The first callback includes that setup cost; steady
callbacks reuse the immutable tables. An upload-completion CUDA event orders
reuse on other streams. Tables, host upload sources, calibrated scratch and
validity scratch are retained with `batch.keepalive` before dependent work.
Each callback allocates independent scratch and output; output is published
before submission. No user callback calls `.get()` or `asnumpy()`.

For 32 panels of 512x1024, calibration plus validity uses **80 MiB/event** of
user scratch: 400 MiB at batch 5 or 1600 MiB at batch 20, per in-flight subbatch.
This is outside the framework's device-memory admission budget; multiple slots
and BDs multiply the cost. Sorted indices cost at most 64 MiB per cached full
layout on both host and device, plus small offsets. The callable also owns its
64 MiB host bin map. No device scratch pool is introduced.

At 64 bins only **1536 bytes/event** is published (7680 bytes for batch 5),
versus 64 MiB/event for the Stage 5a calibrated image. Psana's normal batched
D2H handles that publication. Reduced output volume is not itself a throughput
measurement; the complete comparison belongs to Stage 5c.

## Validation

Job **39375921** completed on `sdfampere040` in **7m39s**, exit **0:0**.
Frozen source and artifacts:
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage5b-20260928-r1`.

- 510 host tests passed (5.40 seconds), including 14 new bin/configuration and
  external CUDA/psana-free declaration tests.
- 24 focused A100 tests passed (3.34 seconds): 19 calibration tests plus five
  integration cases. They check sparse/reordered physical IDs, missing events
  and panels, all gain codes, zero/nonfinite gains, both status arrays,
  nonfinite calibration values, valid zero intensity, all-excluded/empty bins,
  float32/64 constants, delayed cross-stream table reuse, changed constants,
  short tails, retained outputs and actual two-launch submissions.
- Real-data reference uses `det.raw.calib(evt)` with no overrides and float64
  NumPy `bincount`. Detector coordinates come from the run's actual stored
  geometry constant. Radius is measured about the explicitly stated geometry
  origin (0,0), **not a claimed measured beam center**. The upper range is 90%
  of the largest radius to exercise out-of-range exclusion. This validates the
  radial integration algorithm; no q-space scientific calibration is claimed.
- The actual copied standalone driver passed: 13 events and three callbacks.
- Direct Stage 5a calibration regression passed against its independent CPU
  reference: all 13 full-image output hashes remain identical. Fresh CPU
  calibration and constant hashes also match the earlier Stage 5a reference.
- Serial and four-rank MPI exclusive/hybrid integration each passed all 13
  events, with exact pixel counts and maximum absolute sum/mean error
  **5.960464477539063e-08**, within the stated tolerance. Histogram output
  hashes match across all three modes.
- Each mode measured three callbacks, subbatches **5,5,3**, and **six actual
  RawKernel submissions**: one calibration and one integration per subbatch.
  Each delivered result was exactly **1536 bytes**. Retained host results
  remained valid after cleanup; MPI service ranks did not import CuPy.

[Validation evidence](user_kernel_stage5b_validation_20260928.json) records
scheduler completion, all 919 frozen Python/CUDA source hashes via the source
manifest, relevant source hashes inline, CPU histogram values, geometry/bin
provenance, per-event output hashes, per-rank launch counts and numerical errors.
All 919 source hashes and all four copied user files were checked against the
checkout after acceptance. Environment: CuPy 13.6.0, CUDA runtime 12090, KvikIO
CPU fallback. No production runtime/native build changes were required. These
correctness checks do not establish a performance speedup; Stage 5c is next.


## Pre-commit review and Stage 5c measurement scope

The review found no blocking correctness or ownership issue. Reviewed source
hashes match the accepted job. Calibration and count semantics, physical-segment
mapping, empty bins, stream dependencies, owner registration before launches,
independent scratch/output allocation and retained host results were checked
against the code and recorded CPU/A100/serial/MPI evidence. No algorithm changes
were required by this review, so the existing acceptance remains applicable.

Stage 5c is a performance comparison of the complete calibration-plus-integration
workload. Its primary comparison runs identical numerical work and delivers the
same compact histogram, with user work scheduled per event versus per execution
subbatch. Input staging, constants, bins, masks, precision, output ownership,
cache state and GPU/BD placement must be matched or explicitly accounted for.
Changing only the DataSource batch size to one is not sufficient to isolate
user scheduling, because it also changes upstream batching.

Planned measurements:

- Event-loop throughput and elapsed seconds for a fixed event count, plus total
  run time. Report explicit initialization and first-use costs (bin sorting,
  table uploads, CUDA compilation) separately from warmed execution.
- Host submission time, actual callback/kernel/copy counts, and GPU calibration,
  integration and D2H durations. Distinguish I/O wait from scheduling overhead.
- Peak device and pinned-host memory, including user scratch and table caches;
  measure the batch-size/stream-depth tradeoff before increasing BD/GPU counts.
- Balanced repeated pairs, initially one BD/GPU, then shared-GPU and multi-GPU
  configurations. Keep cold/warm I/O cases separate and verify histogram
  equivalence before accepting timing samples.

For a full 20-event subbatch, the intended launch comparison is 40 user kernel
submissions per-event versus two batched submissions, with the same two
algorithms and the same 20 delivered histograms. A staging-only no-task run may
provide a secondary workload-cost baseline; it is not the same-work reference
for a claimed batching speedup. No Stage 5c speedup or regression conclusion has
been measured yet. Stage 5c has not been launched by this review.
