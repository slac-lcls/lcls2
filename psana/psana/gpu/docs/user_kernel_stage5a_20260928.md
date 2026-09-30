# Stage 5a: external user-owned batched Jungfrau calibration

**Status:** Initial implementation validated. 493 host unit tests, 150 A100 device tests,
and all copied external-driver serial/MPI checks passed. Focused retry
**39372902** completed with exit 0. No new throughput claim.
Runtime baseline is `21a6628bb` (production runtime unchanged from `10df4c6e3`).
CPU-v3 parity follow-up also passed: **496 host tests, 19 focused A100 tests**,
plus byte-for-byte default CPU calibration matches in serial and both MPI modes.
[Stage 5b radial integration](user_kernel_stage5b_20260928.md) is now validated.
[Stage 5c combined performance](user_kernel_stage5c_20260928.md) is complete and accepted.

## External user contract

Copy [jungfrau_calibration.py](../examples/jungfrau_calibration.py) and
[calibrate_jungfrau.py](../examples/calibrate_jungfrau.py) to a user directory.
The kernel module imports only NumPy and Python standard-library code at module
load. It imports CuPy inside the callback and imports no psana implementation
module. The driver uses public `DataSource`, `GpuTask`, batch methods and result
access only. No runtime calibration hook, internal monkeypatch or private
manager access is needed by either user file.

```bash
PS_PARALLEL=none python calibrate_jungfrau.py mfx100848724 51 \
  --directory /sdf/data/lcls/ds/prj/public01/xtc \
  --events 13 --batch-size 5 --depth 2
```

Standard psana MPI launch configuration also applies. The host-only callable is
constructed wherever the driver runs; CuPy and compiled kernels exist only on
workers executing callbacks. The driver uses `closing(run.events())` and consumes
named host results. Every invocation submits one calibration kernel over the
entire selected subbatch, including short tails; there is no event-loop kernel
launch or Python loop launching per event inside the callback.

## Numerical policy

The default requests `pedestals` and `pixel_gain`. `--offset` additionally
requests `pixel_offset`; `--status-bits 0xffff` additionally requests
`pixel_status`; `--stextra-bits` additionally requests `status_extra`. Missing requested constants raise an error; disabled options
neither request nor read those arrays. These are explicit user policies, not
an implicit reproduction of every default of `det.raw.calib()`.

- Raw data are uint16 `(N, S, H, W)` and presence is uint8 `(N, S)`.
- Original pedestal/gain/offset arrays have `(3, P, H, W)` layout and float32
  or float64 dtype. `P` indexes physical segments, including sparse IDs;
  `batch.segment_ids()` maps the `S` input rows to that axis. Source shape and
  dtype are unchanged by psana. Like CPU v3, the example adds pedestal/offset
  and divides for reciprocal gain in the promoted source precision, then rounds
  those derived constants to float32 before applying them to ADC values.
- Gain codes `00`, `01`, `11` select planes 0, 1, 2. Code `10`, missing segments,
  zero gain, and masked pixels with finite constants produce zero.
- Optional status arrays have the same physical layout and unsigned integer
  dtype (8/16/32/64 bits). A pixel is masked when any of its three gain planes
  has a selected status bit in either enabled status array. Unselected bits do not mask it.
- The formula is `(ADC - (pedestal + optional_offset)) * (1 / gain)`, with
  float32 output arithmetic, correctly rounded source-precision division, and
  fused multiply-add disabled to match CPU v3's operation order.
- No common-mode correction, geometry, custom edge/neighbor/user masks,
  or azimuthal integration is included. Nonfinite unmasked constants follow
  ordinary floating-point propagation; like CPU v3, a zero mask or gain factor
  does not suppress NaN from a nonfinite pedestal. This example does not sanitize them.

The gain-code/pedestal algorithm is adapted from Amanda Shackelford's
`jungfrau_calib_pixel` at `d5437f99dafa68c3071e437ae034457735f00888`.
The adaptation reads original constants directly and adds a leading event axis,
physical-segment mapping, presence handling and explicit optional policies.
Its independent CPU reference lives in test support, not in the user module.

## Scheduling and ownership

The callback checks shape, dtype, contiguity and physical segment bounds using
host metadata only. Its compiled kernel embeds the small immutable segment map;
there is no metadata upload, constant conversion kernel or device-to-host access
inside the callback. Modules are cached by device, segment mapping and constant
types. Original constant arrays are fetched on every invocation, so replacement
at a drained transition does not leave stale derived arrays in user state.

Each invocation allocates a fresh float32 `(N,S,H,W)` output and registers it
with `batch.publish()` before launching on the supplied stream. Modules are the
only persistent CUDA objects in the callable. Original constants and raw inputs
remain borrowed framework-owned values. The user module caches no borrowed
input, constant or output array. Concurrent slots cannot overwrite one another's
output. Published host rows remain readable after run cleanup.

Full images intentionally make numerical inspection easy in this stage. They
cost `4*N*S*H*W` bytes of user device memory outside psana's input quota. A
batch-20, 32-panel `(512,1024)` output is **1280 MiB**; large groups exceed the
unchanged 64 MiB default output pinned cap and use ordinary-host fallback.
Stage 5b should keep intermediate calibrated images on the device and publish
compact integration results; it must explicitly track validity for bin counts,
since a zero calibrated value does not distinguish a valid zero from a masked
or absent pixel. No performance claim is made for full-image publication here.

## Initial validation and provenance (before CPU-v3 parity changes)

- 16 new host tests: CUDA/psana-free copied-module declaration, selector options,
  unsupported settings/layouts, sparse segment bounds, and no psana imports.
- Complete GPU host unit suite: **493 passed**.
- Device tests cover float32/64 constants, offsets, selected status bits including
  bit 40, all gain codes, zero gain, missing segments, sparse/reordered physical
  IDs, and non-block-size-multiple inputs. A separate test queues delayed work
  on independent streams, replaces constants between calls, and checks retained
  outputs in reverse completion order.
- Real-data acceptance imports an actual copy of the module from an external
  user directory. It checks every pixel for 13 events against the independent
  CPU policy reference, counts actual RawKernel submissions, requires callback
  sizes **5, 5, 3**, and checks retained host results. Accepted modes are serial
  base/optional policies and four-rank MPI exclusive/hybrid routing.
- Validation-only launch instrumentation wraps CuPy RawKernel; it does not patch
  psana runtime behavior. Numerical host copies are outside the user callback.

Frozen source, 913 source hashes, copied external files, launcher and logs:
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage5a-20260928-r1`.
Job **39372486**, sdfampere038, passed all 150 device tests in 219.47 seconds
and the copied standalone driver (13 events, three callbacks). Its subsequent
real-data reference comparison failed because the validation driver stacked
public `(1,H,W)` segment fields without removing their singleton dimension.
This was a reference-input shape error; the user module/kernel was unchanged.

The corrected driver strips that singleton dimension and rejects unexpected
panel ranks. The reference now rejects non-4-D raw input explicitly. Retry
source and logs are frozen separately under
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage5a-20260928-r2`;
job **39372902**, sdfampere016, completed in **5m07s**, exit **0:0**. All nine
focused device tests passed again (2.60 seconds), followed by all four public
serial/MPI comparisons. Earlier complete device results remain attached to r1.
Every comparison checked all pixels of 13 `(32,512,1024)` images against its
CPU reference, retained results after cleanup, and measured exactly three
RawKernel submissions for callback sizes 5, 5, 3. MPI used two BDs; service
ranks never imported CuPy. Optional-policy output hashes match exactly across
serial, exclusive MPI and hybrid MPI; all configurations have identical timestamp
sets. Both copied user files are identical across attempts and to the checkout.
The retry changes only the validation driver/reference, not the user kernel.

[Compact validation evidence](user_kernel_stage5a_validation_20260928.json)
records job status, source/log hashes, per-rank batches and per-event output
hashes. All 913 frozen source hashes were verified against the checkout after
acceptance. Environment: CuPy 13.6.0, CUDA runtime 12090, KvikIO CPU fallback.
The existing native installation is reused. No psana core/runtime code changed;
core/byhand results from `10df4c6e3` remain the inherited baseline. No Stage 5a throughput result is inferred from these correctness-test timings.


## CPU-v3 parity follow-up

The minimum follow-up keeps the existing explicit constant selectors and one
kernel launch per subbatch. It adds `stextra_bits`, matches CPU source-precision
constant preparation before float32 conversion, and preserves CPU multiplication
semantics for masked nonfinite values. No runtime code or derived-constant cache
is added. This remains a configured example, not a complete drop-in replacement
for all detector calibration options or missing-constant fallbacks.

The real-data validation now first runs a separate CPU-only DataSource and calls
**`det.raw.calib(evt)` with no overrides**. It saves each float32 image keyed by
timestamp, plus raw-panel and constant hashes. GPU comparisons enable available
offsets and all bits of available `pixel_status`/`status_extra` arrays. Every
pixel is compared with `assert_array_equal`, with no tolerance, after verifying
raw-panel identity and mapping physical segment IDs to CPU rows. The public
fixture has all 32 physical panels. Custom masks and common mode are outside
this acceptance scope; CPU's default v3 also does not apply common mode.

Synthetic tests separately call the actual C++ `calib_jungfrau_v3` entry point
with NumPy-prepared constants, covering float32/64 rounding and nonfinite values.
Status-extra bits, sparse/reordered panels, missing panels and partial batches
remain covered by independent array-reference device tests.

For this public run (which has offsets and pixel_status, but no status_extra),
the external driver selects CPU-default numerical behavior with:

```bash
PS_PARALLEL=none python calibrate_jungfrau.py mfx100848724 51 \
  --directory /sdf/data/lcls/ds/prj/public01/xtc --events 13 --batch-size 5 \
  --offset --status-bits 0xffffffffffffffff
```

Acceptance job **39374665** completed on `sdfampere013` in **10m17s**, exit
**0:0**. Results:

- Complete host unit suite: **496 passed** (5.10 seconds).
- Focused A100 device suite: **19 passed** (3.08 seconds), including actual C++
  v3 comparisons, both status arrays, stream ordering and retained outputs.
- Serial, four-rank MPI exclusive and four-rank MPI hybrid each matched default
  `det.raw.calib(evt)` for **13 full 32-panel images**: **218,103,808 pixels per
  mode**. All output SHA-256 hashes also equal their CPU reference hashes, so
  this is byte-for-byte equality, including signed-zero representation.
- Each mode measured exactly **three RawKernel submissions**, with subbatch
  sizes **5, 5, 3**. MPI used two BDs; service ranks did not import CuPy.
- Existing base/optional NumPy-policy serial checks also passed; optional policy
  used zero pinned-output budget. Retained results stayed valid after cleanup.
- The public fixture has float32 pedestals/gains/offsets and uint64 pixel_status;
  it has no status_extra. Status-extra and float64 preparation are covered by
  synthetic device tests, including actual C++ v3 acceptance.

Artifacts are frozen under
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage5a-cpu-parity-20260928-r1`.
This includes the separate CPU-only reference images/manifest, actual copied user
module, source manifest, batch script and all logs. All **913** frozen Python/CUDA
source hashes and both external user-file copies were checked against the
checkout after completion. The native install is unchanged; CuPy 13.6.0, CUDA
runtime 12090, KvikIO CPU fallback. These are correctness results, not timings
for a performance comparison.

[CPU-v3 parity evidence](user_kernel_stage5a_cpu_parity_20260928.json) records
source provenance, scheduler result, constants/raw-panel hashes, per-rank launch
counts and CPU/GPU output hashes. The earlier validation JSON remains a record
of the original Stage 5a policies before these changes.
