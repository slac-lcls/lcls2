# Calibration and azimuthal integration callback sources

**September 28 status:** the batched `GpuTask` API is implemented; the scientific
`CalibAzint` example remains pending (Stage 5). The dated source survey and
illustrative per-event signatures below are historical. Adapt algorithms to
`function(batch, stream)` using the [current guide](../user_task_results.md).


2026-09-26. Source inspection only; no callback API or adapted kernels have
been implemented or tested here. The first user-kernel example should run
**user calibration followed by azimuthal integration inside one callback**
passed to `DataSource(gpu_fn=GpuTask(...))`. Psana prepares declared inputs,
tracks completion, and delivers published results. The public event loop
does not launch either algorithm.

## Sources located

### Amanda's lcls2 CUDA kernels: preferred initial source

Remote branch heads were verified with `git ls-remote` on September 26:

- `slac-lcls/lcls2:features/psana2-gpu-kernels`,
  `68cffd612e391ceb8d8018ebb1394f62625fde08`.
- `slac-lcls/lcls2:features/psana2-gpu-d2h-kernels`,
  `d5437f99dafa68c3071e437ae034457735f00888`.

Amanda's commits `74e5d1d1c`, `e3192ee3d`, and `28d88d613` introduce fused
calibration/integration, sorted reductions, and registry integration.
`650c76780e7b1e2db4ff1a46b0a68b77e5c58b52` ports the analysis to a downstream
`on_gpu_view` operator. Prefer the latter source for extracting the algorithms:

| File under `psana/psana/gpu/` | Useful implementation |
|---|---|
| [cuda/analysis_kernels.cu](https://github.com/slac-lcls/lcls2/blob/650c76780e7b1e2db4ff1a46b0a68b77e5c58b52/psana/psana/gpu/cuda/analysis_kernels.cu) | `azint_gather_kernel`, `azint_sorted_kernel`, `azint_global_kernel`, `normalize_kernel`; fused raw-to-histogram variants and bank common-mode kernels also present |
| [cuda/fused_calib.cuh](https://github.com/slac-lcls/lcls2/blob/d5437f99dafa68c3071e437ae034457735f00888/psana/psana/gpu/cuda/fused_calib.cuh) | `jungfrau_calib_pixel`: gain-bit selection, pedestal subtraction, multiplication by prepared inverse-gain/mask |
| [gpu_azint.py](https://github.com/slac-lcls/lcls2/blob/650c76780e7b1e2db4ff1a46b0a68b77e5c58b52/psana/psana/gpu/gpu_azint.py) | `JungfrauAzint`: CPU bin-table setup and CuPy launches; output `(3, nbins)` float32, rows `I_avg`, `sum_I`, `sum_N` |
| [test_analysis_kernels.py](https://github.com/slac-lcls/lcls2/blob/650c76780e7b1e2db4ff1a46b0a68b77e5c58b52/psana/psana/gpu/test_analysis_kernels.py) | Standalone synthetic calibration/integration references; optional saved real geometry |
| [test_azint.py](https://github.com/slac-lcls/lcls2/blob/650c76780e7b1e2db4ff1a46b0a68b77e5c58b52/psana/psana/gpu/test_azint.py) | Sorted and atomic integration compared against float64 CPU accumulation |

All these Git objects are available locally. For example:

```bash
git show 650c76780:psana/psana/gpu/cuda/analysis_kernels.cu
git show d5437f99d:psana/psana/gpu/cuda/fused_calib.cuh
```

The existing branch's `gpu_calib.py:fused_calib_gpu` is another calibration
launcher reference. Its prepared two-array interface is not the proposed
original-calibration-dictionary interface.

### Amanda's original standalone project

The CUDA source explicitly traces its algorithms to this readable checkout:

```text
/sdf/data/lcls/ds/mfx/mfx101344525/results/jungfrau_gpu_azint
```

Its recorded remote is `git@github.com:lcls-daw/jungfrau_gpu_azint.git`; fetching
that remote returned `Repository not found` with the current credentials.
Local HEAD is `1c2296a`, authored by Amanda Jean Shackelford. Inspection used
the local files; they are not claimed to match a fresh remote snapshot.

- `analysis/jungfrau_processing.py`: `Calibrate`, `AzimuthalIntegrate`,
  `DirectIntegrate`, and OpenCL kernels.
- `INSIGHTS_AND_ADDITIONS_CUPY.md`: CuPy/CUDA design and kernel sketches.
- `tests/benchmark_direct_integrate.py`: geometry and direct-integration setup.

Use this for provenance and algorithm context. Its OpenCL queue and pyFAI
workflow cannot directly satisfy the proposed supplied-CUDA-stream contract.

### Stefano's smalldata_tools GPU integration

Found on `smarkesini/smalldata_tools:gpu-azimuthal-binning`, verified head
`0dbddb69f9b05f376faec3dfaa1974485aa04585`:

- [azimuthalBinning_gpu.py](https://github.com/smarkesini/smalldata_tools/blob/0dbddb69f9b05f376faec3dfaa1974485aa04585/smalldata_tools/ana_funcs/azimuthalBinning_gpu.py)
- [README](https://github.com/smarkesini/smalldata_tools/blob/0dbddb69f9b05f376faec3dfaa1974485aa04585/smalldata_tools/ana_funcs/azimuthalBinning_gpu_README.md)
- [tests](https://github.com/smarkesini/smalldata_tools/blob/0dbddb69f9b05f376faec3dfaa1974485aa04585/tests/test_azimuthalBinning_gpu.py)

This subclasses the CPU binning implementation and replaces reduction with
CuPy sparse CSR matvec (or batched SpMM). Geometry, q/phi/r bins, polarization,
solid-angle correction, and normalization come from the parent. The numerical
path uses float64. This is a useful second example with richer corrections.

`doCake()` and `doCake_batch()` currently finish with `cp.asnumpy`; extract a
device-returning operation before adapting it to producer callbacks. Prepare
the sparse operator, fixed gather indexes, and device constants before steady
state; audit boolean compaction and library calls for hidden synchronization.
Retain all temporary arrays through completion on the supplied stream.
Its dark/gain image handling is not a replacement for Jungfrau gain-state
calibration. Avoid applying corrections twice to an already calibrated frame.

Stefano's `smarkesini/lcls2` exposes only `master` at `3d7ba3b5d`; the targeted
tree search did not locate these GPU integration kernels there.

## Adaptation for the first callback

Start with user calibration followed by Amanda's sorted gather/reduction and
normalization. Keep the algorithms in the example/user layer. Do not restore
the old registry, scheduler, or automatic post-calibration invocation.
This clear sequence is the minimum viable example, not a claim that per-event
launch overhead is negligible. Count its calibration, gather, reduction, and
normalization submissions separately from psana's batched parser/gather and
output copies. Keep bin-table preparation out of steady-state callbacks;
consider fusion or batch callbacks only after measuring the working example.

Required changes before using the source as an acceptance example:

1. Consume declared original `pedestals`, `pixel_gain`, optional
   `pixel_offset`, and `pixel_status` values. Define the mask and zero-gain
   policy explicitly. Any conversion to prepared pedestal/gain-mask arrays
   belongs to user code, with registered owners and refresh ordering.
2. Align calibration arrays and geometry/bin tables with `evt.segment_ids()`.
   Use actual geometry and beam parameters for scientific validation; the
   wrapper silently falls back to tiled geometry and arbitrary radius units.
3. Replace `JungfrauAzint._sorted_d`, a single mutable instance buffer, with
   fresh registered scratch for the first example. Serial host invocations
   do not serialize GPU work on different execution-slot streams.
4. Incorporate per-event presence into both intensity and count reduction.
   The existing sorted kernel counts every statically selected pixel. Merely
   zeroing missing segments would bias `I_avg` through an incorrect `sum_N`.
   Define separately whether invalid gain codes count; calibration returning
   zero is not automatically a validity policy.
5. Handle no valid pixels and explicit q-range exclusion. The current wrapper
   clips out-of-range pixels into edge bins and can form a zero-block gather
   launch for an empty selection. Cover these cases against the reference.
6. Keep all launches on the supplied stream, register scratch before launch,
   and publish an independently owned histogram before producing it. Let
   psana schedule D2H; no `.get()`/`asnumpy()` in the callback. Use no common
   mode initially: the source bank-mean algorithm is not psana's median.

Proposed user-facing shape (illustrative; `GpuTask` and `CalibAzint` are not
implemented). `CalibAzint` is a user-owned callable whose `__call__(evt, stream)`
launches both stages, keeps temporaries alive, and publishes `jungfrau.azint`.
Its constructor holds host configuration only; CUDA setup occurs on the BD:

```python
analysis = CalibAzint(geometry=geometry, nbins=256)
task = GpuTask(
    function=analysis,
    inputs=["jungfrau.raw"],
    calibconst=[("jungfrau", key) for key in
                ("pedestals", "pixel_gain", "pixel_offset", "pixel_status")],
)
ds = DataSource(..., gpu_det="jungfrau", gpu_fn=task)
for run in ds.runs():
    for evt in run.events():
        try:
            result = evt.gpu.get("jungfrau.azint")
        except KeyError:  # callback may publish nothing for absent input
            continue
        intensity, summed_intensity, counts = result.on_cpu
```

At 256 bins, the float32 `(3, 256)` result is 3072 bytes per event. Publishing
calibrated frames can be an explicit validation option. This example's layout
is not a framework constraint: users can change output shape/dtype and publish
conditionally or every N events. Psana uses each publication's byte extent and
metadata, and its terminal CUDA event governs host readiness. Any accumulation
and device-buffer allocation/reuse remain user responsibilities.
Compare calibration
pixels against the agreed reference and histogram sums/averages with stated
floating-point tolerances; counts should match exactly. Include overlapping
streams, missing segments, empty bins, changing constants, and delayed D2H.
Historical branch tests and performance reports do not validate this adaptation.
