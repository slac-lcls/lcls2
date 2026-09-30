# EventJoiner — General CPU/GPU Result Combining at Checkpoints

**Status:** Design + implementation plan, not yet implemented  
**Supersedes:** `calib_joiner_implementation.md` (specific to calib scatter)

## Goal

A general `EventJoiner` that accumulates any CPU result (numpy) and GPU result
(CuPy) across N events, then at a checkpoint:
1. Transfers the accumulated GPU results to CPU in **one D→H**
2. Returns both to the user as numpy arrays
3. The user applies whatever combination or operation they need

This covers all use cases:
- Scatter GPU + CPU Jungfrau segments into full `(N, 32, 512, 1024)` array
- Sum GPU hit images normalised by CPU beam energy
- Correlate GPU-computed structure factors with CPU motor positions
- Any custom reduction

---

## Design

### Lazy calibration semantics

`ctx.get('jungfrau.calib')` follows the same lazy principle as the CPU path
(`det.raw.calib(evt)`): calibration only runs when the user calls it, not
for every event regardless of whether the result is needed.

When `ctx.get('jungfrau.calib')` is called it does exactly two things:

1. Runs `fused_calib_gpu()` for the GPU-routed segments (streams 6, 8, 9 →
   19 segs, result stays in GPU VRAM)
2. Runs `compute_cpu_calib()` for the CPU-routed segments (streams 0, 7 →
   13 segs, result stays in CPU RAM)

Nothing is transferred between CPU and GPU at this point. The returned
`GPUResult` has segments sitting in two different places:

```
GPU VRAM:  calib_slot_buf[slot][offset:offset+19]  ← streams 6, 8, 9
CPU RAM:   cpu_segs (13, 512, 1024) float32        ← streams 0, 7
```

Transfers only happen when the user explicitly requests a unified view:

```python
calib = ctx.get('jungfrau.calib')

calib.gpu_segs   # (19, 512, 1024) cupy  — no transfer
calib.cpu_segs   # (13, 512, 1024) numpy — no transfer

calib.on_gpu     # (32, 512, 1024) cupy  — H→D for the 13 CPU segs (~1 ms)
calib.on_cpu     # (32, 512, 1024) numpy — D→H for the 19 GPU segs (~13 ms)
```

This means different event loops have different transfer costs:

```python
# No transfers — joiner uses .gpu_segs and .cpu_segs internally
for ctx in run.events():
    result = joiner.add(ctx.get('jungfrau.calib'))

# D→H only — user wants full array on CPU
for ctx in run.events():
    full = ctx.get('jungfrau.calib').on_cpu   # D→H + scatter, ~13 ms/event

# H→D only — user wants full array on GPU for further GPU compute
for ctx in run.events():
    full = ctx.get('jungfrau.calib').on_gpu   # H→D for CPU segs, ~1 ms/event

# Both — GPU compute then conditional save
for ctx in run.events():
    calib = ctx.get('jungfrau.calib')
    peaks = find_peaks_gpu(calib.on_gpu)      # H→D for CPU segs (~1 ms)
    if peaks > threshold:
        save(calib.on_cpu)                    # D→H for GPU segs (~13 ms)
```

If `ctx.get()` is never called (e.g. the user only reads scalar diagnostics),
no calibration runs at all — matching CPU-path behaviour.

### GPUResult carries both sides

`GPUResult` is updated to carry both the GPU-routed segments and the
CPU-routed segments:

```python
result = ctx.get('jungfrau.calib')
result.gpu_segs   # cp.ndarray (19, 512, 1024) — GPU-calibrated, on device
result.cpu_segs   # np.ndarray (13, 512, 1024) — CPU-calibrated, on host
```

`EventJoiner` is created **before** the event loop with just `n_checkpoint`.
On the first `add()` call it reads the seg_ids from the `GPUResult` and
auto-configures the scatter callable — no `ctx` needed at construction time:

```python
joiner = EventJoiner(n_checkpoint=100)   # ← before the loop

for ctx in run.events():
    result = joiner.add(ctx.get('jungfrau.calib'))
    if result is not None:
        full = result.combined   # (100, 32, 512, 1024) numpy
```

`joiner.add()` is consistent regardless of routing: fully GPU-routed detectors
produce a `GPUResult` with `cpu_segs=None`; split-routed detectors carry both;
either way the call is identical.

## API

```python
class GPUResult:
    gpu_segs : cp.ndarray   # GPU-calibrated segments (on device)
    cpu_segs : np.ndarray   # CPU-calibrated segments (on host, None if all GPU-routed)
    on_gpu   : cp.ndarray   # full detector array on GPU (H→D cpu_segs if split-routed)
    on_cpu   : np.ndarray   # full detector array on CPU (D→H gpu_segs + scatter)


class EventJoiner:
    def __init__(self, n_checkpoint, op=None):
        """
        n_checkpoint : int
            Number of events to accumulate before returning a result.

        op : None | 'stack' | 'sum' | 'mean' | callable
            None (default) — auto-configure from the first GPUResult received:
                If GPUResult carries seg_ids, builds a scatter callable that
                returns (n_checkpoint, n_segs_total, nrows, ncols) as
                result.combined.  Works for both split-routed and fully
                GPU-routed detectors.
            'stack' — (n_checkpoint, ...) stacked array per side
            'sum'   — element-wise sum over n_checkpoint events
            'mean'  — element-wise mean over n_checkpoint events
            callable(cpu_arr, gpu_arr) → single array or (cpu, gpu) tuple
                Applied after D→H; both args are numpy.
                Single return  → JoinResult.combined
                Tuple return   → JoinResult.cpu / JoinResult.gpu
        """

    def add(self, result_or_cpu, gpu_result=None):
        """Add one event.

        Parameters
        ----------
        result_or_cpu : GPUResult or scalar or np.ndarray
            Pass a GPUResult from ctx.get() to use the single-argument form.
            Pass a CPU scalar/array as the first arg and a GPUResult (or
            cp.ndarray) as gpu_result for the two-argument form.

        Returns
        -------
        JoinResult or None
        """

    def flush(self):
        """Return any partially accumulated events without waiting for
        n_checkpoint.  Returns None if no events accumulated since last output.
        """


class JoinResult:
    cpu      : np.ndarray   # aggregated CPU-side input (on host)
    gpu      : np.ndarray   # aggregated GPU-side input (on host after D→H)
    combined : np.ndarray   # unified result when op returns a single array
    n_events : int          # events in this result (< n_checkpoint for flush())
```

---

## Usage examples

### Example 1 — Full detector calibration: all segments, all N events on CPU

Some Jungfrau segments are GPU-calibrated (streams 6, 8, 9) and some are
CPU-calibrated (streams 0, 7). The joiner auto-configures from the first
`GPUResult` it receives — no `ctx` or seg_id knowledge needed at construction.

```python
joiner = EventJoiner(n_checkpoint=100)   # before the loop

for ctx in run.events():
    result = joiner.add(ctx.get('jungfrau.calib'))
    if result is not None:
        full = result.combined    # (100, 32, 512, 1024) float32 numpy
        save(full)
```

`result.combined` has all 32 segments in their correct calibconst positions.
On the first call, the joiner reads seg_ids from the `GPUResult` and builds
the scatter callable automatically. Subsequent calls reuse it.

### Example 2 — Sum GPU images normalised by CPU beam energy

```python
joiner = EventJoiner(n_checkpoint=100, op='sum')   # before the loop

for ctx in run.events():
    result = joiner.add(ctx.raw('gmd').energy,     # scalar float (CPU side)
                        ctx.get('jungfrau.calib'))  # GPUResult   (GPU side)
    if result is not None:
        # result.cpu: float  — sum of 100 beam energies
        # result.gpu: (32, 512, 1024) numpy — sum of 100 full detector images
        if result.cpu > 0:
            normalised = result.gpu / result.cpu
            save(normalised)
```

### Example 3 — Custom operation (e.g. XPCS two-time correlation)

```python
def my_reduce(cpu_arr, gpu_arr):
    # cpu_arr: (100,) motor positions, gpu_arr: (100, 32, 512, 1024) numpy
    return compute_g2(gpu_arr) / cpu_arr.mean()   # → result.combined

joiner = EventJoiner(n_checkpoint=100, op=my_reduce)  # before the loop

for ctx in run.events():
    result = joiner.add(ctx.raw('motor').position,
                        ctx.get('jungfrau.calib'))
    if result is not None:
        save_g2(result.combined)
```

### Example 4 — GPU only (no CPU scalar)

```python
joiner = EventJoiner(n_checkpoint=100, op='mean')   # before the loop

for ctx in run.events():
    result = joiner.add(ctx.get('jungfrau.calib'))
    if result is not None:
        save(result.combined)   # (32, 512, 1024) numpy — mean over 100 events
```

### Example 5 — Hit finding: select only bright events

GPU computes the number of bright pixels per event. CPU has the beam energy.
At every 100 events, pull only the hit events to CPU.

```python
import cupy as cp
import numpy as np

HIT_THRESHOLD  = 5.0    # ADU
MIN_BRIGHT_PIX = 1000   # minimum pixels above threshold to count as a hit
MIN_ENERGY     = 0.3    # mJ — skip dark shots

joiner = EventJoiner(n_checkpoint=100, op='stack')

for ctx in run.events():
    calib  = ctx.get('jungfrau.calib').on_gpu     # (19, 512, 1024) cupy
    energy = ctx.raw('gmd').energy                # scalar float

    # GPU hit flag (scalar int on GPU — cheap, no D→H here)
    n_bright = int(cp.sum(calib > HIT_THRESHOLD))
    is_hit   = int(n_bright >= MIN_BRIGHT_PIX and energy > MIN_ENERGY)

    result = joiner.add(np.array([energy, n_bright, is_hit]),  # (3,) numpy
                        ctx.get('jungfrau.calib'))            # GPUResult
    if result is not None:
        # result.cpu: (100, 3) numpy — [energy, n_bright, is_hit] per event
        # result.gpu: (100, 19, 512, 1024) numpy — all 100 images on CPU
        hit_mask = result.cpu[:, 2].astype(bool)
        if hit_mask.any():
            hit_images   = result.gpu[hit_mask]           # only hit events
            hit_energies = result.cpu[hit_mask, 0]
            save_hits(hit_images, hit_energies)
        print(f'hits: {hit_mask.sum()} / 100')
```

### Example 6 — Multiple CPU scalars (GMD energy, IPM position, motor angle)

Combine GMD energy, IPM x/y position, and a motor angle with the GPU image.

```python
import numpy as np

joiner = EventJoiner(n_checkpoint=200, op='stack')

for ctx in run.events():
    gmd    = ctx.raw('gmd').energy              # scalar
    ipm_x  = ctx.raw('ipm').xpos               # scalar
    ipm_y  = ctx.raw('ipm').ypos               # scalar
    angle  = ctx.raw('motor').angle            # scalar

    result = joiner.add(np.array([gmd, ipm_x, ipm_y, angle]),  # (4,) numpy
                        ctx.get('jungfrau.calib'))            # GPUResult
    if result is not None:
        # result.cpu: (200, 4) numpy — all slow diagnostics
        # result.gpu: (200, 19, 512, 1024) numpy — all GPU images
        energies = result.cpu[:, 0]
        xpos     = result.cpu[:, 1]
        ypos     = result.cpu[:, 2]
        angles   = result.cpu[:, 3]
        save_batch(result.gpu, energies, xpos, ypos, angles)
```

### Example 7 — Mean image per scan step

The experiment scans a motor through 10 positions. Accumulate a mean image
at each step by resetting the joiner when the motor position changes.

```python
joiner    = EventJoiner(n_checkpoint=50, op='mean')
last_step = None

for ctx in run.events():
    step = ctx.raw('motor').step_value         # int scan step index

    # Flush and reset when scan step changes mid-checkpoint
    if last_step is not None and step != last_step and joiner.n_accumulated > 0:
        partial = joiner.flush()               # see §flush() note below
        if partial is not None:
            save_step_image(last_step, partial.gpu, joiner.n_accumulated)
        joiner = EventJoiner(n_checkpoint=50, op='mean')

    result = joiner.add(step, ctx.get('jungfrau.calib'))
    if result is not None:
        # result.gpu: (19, 512, 1024) numpy — mean over 50 events at this step
        save_step_image(step, result.gpu, n_events=50)

    last_step = step
```

### Example 8 — Azimuthal integration on GPU + CPU normalisation

The GPU computes `I(q)` (1-D intensity profile) from each Jungfrau image using
a pre-computed `q_bins` lookup on GPU. The CPU provides the beam transmission.

```python
import cupy as cp

# Pre-compute azimuthal bin assignments once (stays on GPU)
q_bins = cp.asarray(load_q_bins())    # (19, 512, 1024) int32 bin index per pixel
n_q    = int(q_bins.max()) + 1

joiner = EventJoiner(n_checkpoint=100, op='sum')

for ctx in run.events():
    calib        = ctx.get('jungfrau.calib').on_gpu        # (19, 512, 1024)
    transmission = ctx.raw('att').transmission             # scalar float 0-1

    # Azimuthal integration on GPU (no D→H here)
    I_q = cp.zeros(n_q, dtype=cp.float32)
    cp.scatter_add(I_q, q_bins.ravel(), calib.ravel())    # (n_q,) sum in each bin

    result = joiner.add(transmission, I_q)   # scalar CPU, (n_q,) cupy GPU
    if result is not None:
        # result.cpu: float — sum of transmissions over 100 events
        # result.gpu: (n_q,) numpy — summed I(q) over 100 events
        if result.cpu > 0:
            mean_Iq = result.gpu / result.cpu    # transmission-normalised I(q)
            save_Iq(mean_Iq)
```

### Example 9 — Flush at end of run (partial checkpoint)

The total number of events may not be a multiple of `n_checkpoint`. Use
`flush()` to retrieve any remaining accumulated events at the end of the run.

```python
joiner = EventJoiner(n_checkpoint=100, op='stack')

for ctx in run.events():
    result = joiner.add(ctx.raw('gmd').energy, ctx.get('jungfrau.calib'))
    if result is not None:
        process(result)

# Retrieve the last partial batch (0–99 events)
final = joiner.flush()
if final is not None:
    # final.n_events: actual number of events in this partial batch
    process(final)
```

`flush()` calls `_assemble()` regardless of whether `n_checkpoint` has been
reached and sets `final.n_events` to the number of events actually accumulated.
Returns `None` if zero events have been accumulated since the last checkpoint.

---

## Implementation

### `GPUResult` update — `gpu/context.py`

Add `cpu_segs` to `GPUResult` so it carries both sides from one `ctx.get()` call.
`_make_context()` computes the CPU-routed segments via `router.compute_cpu_calib()`
(numpy arithmetic, no I/O, no GPU) and stores them in the `GPUResult`:

```python
class GPUResult:
    __slots__ = ('_arr', '_stream', '_cpu_segs',
                 '_gpu_seg_ids', '_cpu_seg_ids', '_n_segs', '_nrows', '_ncols')

    def __init__(self, arr_gpu, stream=None, cpu_segs=None,
                 gpu_seg_ids=None, cpu_seg_ids=None,
                 n_segs=None, nrows=None, ncols=None):
        self._arr         = arr_gpu      # cp.ndarray — GPU-routed segments
        self._stream      = stream
        self._cpu_segs    = cpu_segs     # np.ndarray — CPU-routed segments, or None
        self._gpu_seg_ids = gpu_seg_ids  # list[int] — calibconst row indices
        self._cpu_seg_ids = cpu_seg_ids
        self._n_segs      = n_segs
        self._nrows       = nrows
        self._ncols       = ncols

    @property
    def gpu_segs(self):
        """GPU-calibrated segments as CuPy array. No transfer."""
        return self._arr

    @property
    def cpu_segs(self):
        """CPU-calibrated segments as numpy array. No transfer."""
        return self._cpu_segs

    @property
    def on_gpu(self):
        """Full detector array on GPU. H→D for cpu_segs if split-routed."""
        if self._cpu_segs is None or self._gpu_seg_ids is None:
            return self._arr
        import cupy as cp, numpy as np
        full = cp.empty((self._n_segs, self._nrows, self._ncols), dtype=cp.float32)
        full[self._gpu_seg_ids] = self._arr
        full[self._cpu_seg_ids] = cp.asarray(self._cpu_segs)
        return full

    @property
    def on_cpu(self):
        """Full detector array on CPU. D→H for gpu_segs + scatter."""
        if self._stream is not None:
            self._stream.synchronize()
        gpu_cpu = self._arr.get()
        if self._cpu_segs is None or self._gpu_seg_ids is None:
            return gpu_cpu
        import numpy as np
        full = np.empty((self._n_segs, self._nrows, self._ncols), dtype=np.float32)
        full[self._gpu_seg_ids] = gpu_cpu
        full[self._cpu_seg_ids] = self._cpu_segs
        return full
```

`_apply_full_routing()` is removed from `_make_context()`. Instead, the
`GPUResult` is constructed with `cpu_segs` already populated:

```python
# gpu_events.py — _make_context()
def _make_context(self, evt, gpu_results):
    # Attach cpu_segs to each GPUResult so ctx.get() carries both sides.
    for det_name, (det, gpu_det, _seg_map) in self.gpu_detectors.items():
        key = det_name + '.calib'
        if key not in gpu_results:
            continue
        ri = self.router._full_routing.get(det_name)
        if ri is None or not ri.cpu_seg_ids:
            continue
        cpu_segs = self.router.compute_cpu_calib(det_name, det, evt)
        old = gpu_results[key]
        gpu_results[key] = GPUResult(
            arr_gpu      = old._arr,
            stream       = old._stream,
            cpu_segs     = cpu_segs,
            gpu_seg_ids  = ri.gpu_seg_ids,
            cpu_seg_ids  = ri.cpu_seg_ids,
            n_segs       = ri.calibconst_n_segs,
            nrows        = ri.nrows,
            ncols        = ri.ncols,
        )
    return GpuEventContext(evt, gpu_results, cpu_dets=self.cpu_dets,
                           stream=None, router=self.router,
                           gpu_detectors=self.gpu_detectors)
```

### `EventJoiner` class — new file `gpu/gpu_event_joiner.py`

```python
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class JoinResult:
    """Result returned by EventJoiner.add() or flush() at each checkpoint.

    All attributes are numpy arrays (or scalars) on CPU.
    The D→H transfer for gpu has already been performed.

    For op='stack'/'sum'/'mean':
        result.cpu      — aggregated CPU-side input
        result.gpu      — aggregated GPU-side input (already on CPU after D→H)
        result.combined — None

    For op=callable returning a single array:
        result.cpu      — None
        result.gpu      — None
        result.combined — the single combined numpy array from the callable
                          e.g. (n_checkpoint, 32, 512, 1024) full detector array

    For op=callable returning a (cpu, gpu) tuple:
        result.cpu      — first element of the tuple
        result.gpu      — second element of the tuple
        result.combined — None
    """
    cpu      : Any   # np.ndarray or scalar — aggregated CPU-side results
    gpu      : Any   # np.ndarray — aggregated GPU-side results (on CPU)
    combined : Any   # np.ndarray — unified result when callable returns single value
    n_events : int   # number of events in this result (< n_checkpoint for flush())


class EventJoiner:
    """Accumulates per-event CPU and GPU results, transferring GPU data to
    CPU in one batch every n_checkpoint events.

    Parameters
    ----------
    n_checkpoint : int
        Number of events per output batch.
    op : 'stack' | 'sum' | 'mean' | callable
        Aggregation applied to the n_checkpoint results at each checkpoint.
        'stack'    → (n_checkpoint, ...) stacked array
        'sum'      → element-wise sum over n_checkpoint events
        'mean'     → element-wise mean over n_checkpoint events
        callable   → user function(cpu_stacked, gpu_stacked) → (cpu_out, gpu_out)
                     called after D→H with both arguments as numpy arrays
    """

    def __init__(self, n_checkpoint: int, op='stack'):
        self._n   = n_checkpoint
        self._op  = op
        self._count = 0

        # Deferred allocation — sized on first add() call.
        self._gpu_store = None   # cp.ndarray (n, *gpu_shape) or cp.ndarray (*gpu_shape) for sum
        self._cpu_store = None   # list (stack/mean) or np.ndarray (sum/mean accumulator)
        self._cpu_is_scalar = False

    def add(self, result_or_cpu, gpu_result=None):
        """Add one event. Returns JoinResult or None.

        Parameters
        ----------
        result_or_cpu : GPUResult or np.ndarray or scalar
            Pass a GPUResult from ctx.get() and omit gpu_result — the joiner
            unpacks .gpu_segs and .cpu_segs internally.
            Or pass an explicit CPU value as the first argument and a
            GPUResult / cp.ndarray as gpu_result.

        gpu_result : GPUResult, cp.ndarray, or None
            Only needed when result_or_cpu is an explicit CPU value (scalar,
            numpy array) rather than a GPUResult.

        Examples
        --------
        joiner.add(ctx.get('jungfrau.calib'))               # GPUResult only
        joiner.add(ctx.raw('gmd').energy,                   # scalar + GPUResult
                   ctx.get('jungfrau.calib'))
        joiner.add(np.array([e, x, y]), ctx.get('det'))     # array + GPUResult
        """
        from psana.gpu.context import GPUResult as _GPUResult
        import cupy as cp

        # Unpack GPUResult into (cpu_segs, gpu_arr) components.
        if isinstance(result_or_cpu, _GPUResult):
            cpu_result = result_or_cpu.cpu_segs   # np.ndarray or None
            gpu_arr    = result_or_cpu.gpu_segs   # cp.ndarray
        else:
            cpu_result = result_or_cpu
            if isinstance(gpu_result, _GPUResult):
                gpu_arr = gpu_result.gpu_segs
                # If the GPUResult has cpu_segs, fold them into cpu_result
                if gpu_result.cpu_segs is not None and cpu_result is None:
                    cpu_result = gpu_result.cpu_segs
            else:
                gpu_arr = gpu_result

        idx = self._count % self._n

        # ── First call: auto-configure op and allocate storage ────────────────
        if self._count == 0:
            # Auto-configure scatter callable from GPUResult seg_ids when
            # op=None and the result carries routing info.
            if self._op is None:
                if (hasattr(result_or_cpu, '_gpu_seg_ids') and
                        result_or_cpu._gpu_seg_ids is not None):
                    ri = result_or_cpu  # GPUResult carries seg_ids
                    gpu_ids = ri._gpu_seg_ids
                    cpu_ids = ri._cpu_seg_ids or []
                    n_segs  = ri._n_segs
                    nrows   = ri._nrows
                    ncols   = ri._ncols
                    def _scatter(cpu_arr, gpu_arr):
                        import numpy as np
                        N    = gpu_arr.shape[0]
                        full = np.empty((N, n_segs, nrows, ncols),
                                        dtype=np.float32)
                        full[:, gpu_ids] = gpu_arr
                        if cpu_arr is not None and cpu_ids:
                            full[:, cpu_ids] = cpu_arr
                        return full
                    self._op = _scatter
                else:
                    self._op = 'stack'   # fallback for plain cp.ndarray input
            self._allocate(cpu_result, gpu_arr)

        # ── Accumulate ───────────────────────────────────────────────────────
        if self._op in ('sum', 'mean'):
            if gpu_arr is not None:
                if idx == 0:
                    self._gpu_store[...] = gpu_arr
                else:
                    self._gpu_store      += gpu_arr
            if cpu_result is not None:
                cpu_val = np.asarray(cpu_result)
                if idx == 0:
                    self._cpu_store[...] = cpu_val
                else:
                    self._cpu_store      += cpu_val
        else:
            if gpu_arr is not None:
                self._gpu_store[idx] = gpu_arr      # D→D copy (~1 MB)
            if cpu_result is not None:
                if self._cpu_is_scalar:
                    self._cpu_store[idx] = float(cpu_result)
                else:
                    self._cpu_store[idx] = np.asarray(cpu_result)

        self._count += 1

        # ── Checkpoint ───────────────────────────────────────────────────────
        if self._count % self._n == 0:
            return self._assemble(self._n)
        return None

    def flush(self):
        """Return any partially accumulated events without waiting for a full
        checkpoint.  Resets the accumulator.

        Returns None if zero events have been accumulated since the last
        checkpoint.  Returns JoinResult with n_events < n_checkpoint otherwise.
        """
        n = self._count % self._n
        if n == 0:
            return None
        result = self._assemble(n)
        # Reset count to next full boundary so add() continues correctly
        self._count = (self._count // self._n + 1) * self._n
        return result

    def _allocate(self, cpu_result, gpu_arr):
        import cupy as cp

        if self._op in ('sum', 'mean'):
            if gpu_arr is not None:
                self._gpu_store = cp.zeros_like(gpu_arr)
            if cpu_result is not None:
                self._cpu_is_scalar = np.ndim(cpu_result) == 0
                self._cpu_store = np.zeros_like(np.asarray(cpu_result))
        else:
            if gpu_arr is not None:
                self._gpu_store = cp.empty(
                    (self._n,) + gpu_arr.shape, dtype=gpu_arr.dtype
                )
            if cpu_result is not None:
                self._cpu_is_scalar = np.ndim(cpu_result) == 0
                if self._cpu_is_scalar:
                    self._cpu_store = np.empty(self._n, dtype=np.float64)
                else:
                    arr = np.asarray(cpu_result)
                    self._cpu_store = np.empty(
                        (self._n,) + arr.shape, dtype=arr.dtype
                    )

    def _assemble(self, n_events):
        """One D→H transfer, apply op, return JoinResult."""
        # D→H: transfer GPU store to CPU in one cudaMemcpy
        if self._gpu_store is not None:
            if self._op in ('sum', 'mean'):
                gpu_cpu = self._gpu_store.get()
            else:
                gpu_cpu = self._gpu_store[:n_events].get()
        else:
            gpu_cpu = None

        if self._op in ('sum', 'mean'):
            cpu_out = self._cpu_store.copy()
        else:
            cpu_out = (self._cpu_store[:n_events].copy()
                       if self._cpu_store is not None else None)

        combined = None
        if self._op == 'mean':
            if gpu_cpu is not None:
                gpu_cpu = gpu_cpu / n_events
            if cpu_out is not None:
                cpu_out = cpu_out / n_events
        elif callable(self._op):
            ret = self._op(cpu_out, gpu_cpu)
            if isinstance(ret, tuple):
                cpu_out, gpu_cpu = ret   # caller returned (cpu, gpu) pair
            else:
                combined = ret           # caller returned single unified array
                cpu_out  = None
                gpu_cpu  = None

        return JoinResult(cpu=cpu_out, gpu=gpu_cpu,
                          combined=combined, n_events=n_events)

    @property
    def n_accumulated(self):
        """Events accumulated since last checkpoint."""
        return self._count % self._n
```

---

## Memory usage

| op | GPU VRAM | CPU RAM |
|---|---|---|
| `'stack'` (n=100, 19 segs) | `100 × 19 × 512 × 1024 × 4 B = 40 MB` | `100 × 13 × 512 × 1024 × 4 B = 27 MB` |
| `'sum'` or `'mean'` | `19 × 512 × 1024 × 4 B = 40 MB` | `13 × 512 × 1024 × 4 B = 27 MB` |
| scalar CPU (e.g. GMD energy) | `19 × 512 × 1024 × 4 B = 40 MB` | `100 × 8 B = 0.8 KB` |

`'sum'`/`'mean'` use constant GPU memory regardless of `n_checkpoint` — the
running sum has the same shape as a single event.

D→H transfer at each checkpoint:
- `'stack'` (n=100, 19 segs): 40 MB, ~3 ms
- `'sum'`/`'mean'` (19 segs): 0.4 MB, ~0.03 ms

---

## Code changes required

### New file: `gpu/gpu_event_joiner.py`
`EventJoiner` and `JoinResult` as above.

### `gpu/context.py` — `ctx.make_joiner()` (optional convenience)

`cpu_calib()` is no longer a user-facing method — it is called internally
in `_make_context()` to populate `GPUResult.cpu_segs`.

`EventJoiner(n_checkpoint=100)` auto-configures from the first `GPUResult`
received, so `ctx.make_joiner()` is no longer needed for the common case.
It is retained only if the user wants to pre-specify `op` with the scatter
logic built in (e.g. `ctx.make_joiner('jungfrau', n_checkpoint=100, op='sum')`
where the sum reduction should apply to the full 32-segment array, not the
raw 19-segment GPU array):

```python
def make_joiner(self, det_name: str, n_checkpoint: int = 100, op=None):
    from psana.gpu.gpu_event_joiner import EventJoiner
    import numpy as np

    ri = self._router._full_routing.get(det_name)
    if ri is None:
        raise RuntimeError(f'make_joiner: no full routing for {det_name!r}')

    if op is None:
        gpu_ids = ri.gpu_seg_ids
        cpu_ids = ri.cpu_seg_ids
        n_segs  = ri.calibconst_n_segs
        nrows   = ri.nrows
        ncols   = ri.ncols

        def _scatter(cpu_arr, gpu_arr):
            N    = gpu_arr.shape[0]
            full = np.empty((N, n_segs, nrows, ncols), dtype=np.float32)
            full[:, gpu_ids] = gpu_arr
            full[:, cpu_ids] = cpu_arr
            return full   # single array → result.combined

        op = _scatter

    return EventJoiner(n_checkpoint=n_checkpoint, op=op)
```

### `gpu/gpu_events.py` — remove per-event H→D

```python
# Remove _apply_full_routing() from _make_context().
# Pass gpu_detectors to GpuEventContext.
def _make_context(self, evt, gpu_results):
    return GpuEventContext(evt, gpu_results, cpu_dets=self.cpu_dets,
                           stream=None, router=self.router,
                           gpu_detectors=self.gpu_detectors)
```

### `gpu/__init__.py` — export

```python
from psana.gpu.gpu_event_joiner import EventJoiner, JoinResult
```

---

## Files to create / modify

| File | Change |
|---|---|
| `gpu/gpu_event_joiner.py` | **New** — `EventJoiner`, `JoinResult` |
| `gpu/context.py` | Extend `GPUResult` with `cpu_segs`, `gpu_seg_ids`, etc.; add `ctx.make_joiner()` |
| `gpu/gpu_events.py` | Replace `_apply_full_routing()` with per-event `GPUResult` construction that includes `cpu_segs` |
| `gpu/__init__.py` | Export `EventJoiner`, `JoinResult` |

`detector_router.py`, `gpu_calib.py`, `gpu_kvikio_read.py`, `mpi_ds.py`,
`node.py` — **no changes needed**.
