# Stage 3A: sample10 fly-scan cube correctness

Completed November 5, 2026. Native and controlled validation passed; independent
Claude review passed on `444a695e25b4141fd41fd4feafd28ad3a5d57ae8` with no blocking
findings. One optional third-stream rejection test is deferred before broadening
stream configuration; see [review response](review-01/response.md).
This is a bounded application-owned prototype on the existing Stage 2 fixtures.
It does not implement the proposed managed GpuTask factory/lifecycle API.

A fresh `Cube` belongs to one Run. It maps the frozen output timestamps to global
output event ordinals, then uses `ordinal % 29` for 30 bins. Bin 29 deliberately
stays empty. This mapping does not use callback-local or BD-local counters.
The tiny/tail windows each have their own output ordinal zero; source indices
remain provenance, not the output bin key.

Each of at most two CUDA stream objects owns separate float32 sum banks and
uint64 event counts. Stream order serializes each bank's updates. One CUDA thread
owns a pixel across all bins and iterates the callback's events, so repeated bin
IDs cannot race. A separate one-thread kernel increments scalar counts once per
event. Any missing/malformed seed panel rejects the whole event; the native
acceptance driver requires zero such errors for these complete fixtures.

The reviewed Stage 2 expansion/calibration runs unchanged through a small
publication adapter. Full images become batch-kept intermediates; only seed-error
flags are published per event. No raw/calibrated image is copied to host during
the event loop. State arrays are owned independently of the batch and are also
registered as CUDA consumers before submission. No borrowed batch, descriptor,
input image or calibration constant is stored persistently.

The application exhausts `run.events()` before explicitly calling `finish()`.
Finalization joins every bank stream, exports one bin at a time and merges the
per-stream float32 sums into host float64. Repeated finish returns the same
result without another export. Incomplete submission refuses export. `close()`
joins streams before releasing banks and also supports abort cleanup. These are
application methods, not automatic BeginRun/EndRun hooks. Source data have only
one native step; step-label retention is tested separately on GPU, without
claiming native multi-step transition acceptance.

Memory: one bank is 30 * (32*512*1024*4 + 8) = 2,013,266,160 bytes;
two banks total 4,026,532,320 bytes, below the explicit 5 GiB persistent-state cap.
This cap is additional to psana's 4 GiB input budget. Expansion/calibration scratch
and constants also consume memory. The merged host cube, CPU reference and sum
of absolute inputs are each about 4.03 GB; validation requests 48 GiB host memory.
Export staging is one float32 bin (~67 MB) at a time. CuPy pool peak is reported
but is not total device-memory usage. Full arrays are saved only for the pilot;
other checks retain reports and per-bin hashes.

CPU reference: expand seeds, apply CPU-v3 calibration with frozen full-shape MFX
pedestal/gain/offset, then accumulate float64 sums and exact uint64 counts.
Every output pixel must satisfy `abs(error) <= gamma_n * sum(abs(input)) + 1e-5`,
where `gamma_n = n*u/(1-n*u)`, `u=2^-24`, and n is that bin's accepted event count.
This bounds float32 addition error without relying on a relative tolerance near
cancellation. Stream partials merge in float64. Every empty bin must be zero.

No threshold/common mode/pixelwise exposure is added. Calibration masking/options
match Stage 2. This is not a scientific cube or performance benchmark. Native
input uses serial psana and KvikIO compatibility mode; MPI/GDS, managed memory
admission, multi-step lifecycle hooks and 35 kHz remain outside this stage.

Validation results (all pixels of each 30-bin output checked):

| Fixture | Events | Stream event counts | Counts | Max error / allowed bound |
|---|---:|---|---|---:|
| Tiny | 10 | 6 / 4 | Exact | 0 |
| Native tail | 3 | 2 / 1 | Exact | 0 |
| Pilot | 1,000 | 501 / 499 | Exact | 0.178131 |

Pilot sums have a maximum absolute difference of 1.1875 from the float64 CPU
reference; every pixel is below its scale-dependent float32 rounding bound.
The comparison includes 503,316,480 cube pixels, including the deliberately
empty bin. Pilot peak CuPy pool allocation: 6,064,670,208 bytes. All controlled
GPU state tests and the two host test groups passed. See validation-summary.json
for per-run hashes, runtime versions, memory and individual reports.

Artifacts:
`/sdf/scratch/users/m/monarin/gpu-validation/mfx-expand10/20261105-stage3a-r1/`.
Final reports: `tiny02/report.json`, `tail01/report.json`, `pilot01/report.json`.
Final arrays: `pilot01/cube.npy` (float64 merged cube, about 4.03 GB) and
`pilot01/counts.npy` (uint64). Input XTC remains the Stage 2 fixture; no new
XTC data were generated. The first tiny attempt passed before harness hardening;
its regenerable full cube was removed after tiny02 passed, retaining its report
and per-bin hashes. Only the main pilot retains a full cube array.

Run host tests with the Stage 2 runtime.sh. Submit run_gpu.sbatch with absolute
arguments: notes-directory, fixture-directory, new-result-directory, batch-size;
append `save` only when retaining the full cube. The script requests one A100
and 48 GiB host memory. Results include code hashes and no performance claim.
