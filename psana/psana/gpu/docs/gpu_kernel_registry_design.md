# GPU Kernel Registry — Design Document

**Status:** Proposed — for team review
**Scope:** `psana.gpu` user-facing API

---

## Problem statement

The psana2 GPU event loop delivers calibrated and raw detector data as
CuPy arrays.  Users routinely need to run custom computations on that data —
threshold filtering, peak finding, azimuthal integration, hit-finding, feature
extraction — without leaving the GPU.  Today every user repeats the same
boilerplate:

```python
# fragile, verbose, repeated in every analysis script
import cupy as cp
kernel = cp.RawKernel(open("my_kernel.cu").read(), "my_kernel")
for evt in run.events():
    with evt.gpu.get("jungfrau.calib").on_gpu_view(stream) as calib:
        n   = calib.size
        out = cp.empty(n, dtype=cp.float32)
        grid = ((n + 255) // 256,)
        kernel((grid,), (256,), (calib.ravel(), out, np.float32(5.0), np.uint64(n)))
```

This has several problems:

- **Boilerplate repeated everywhere** — grid math, type casts, argument
  packing.
- **No reuse** — the same kernel is defined independently in each script.
- **JIT latency on every run** — no shared compilation cache; first event
  pays the full NVRTC compile cost every time the script starts.
- **Thread-safety not handled** — two threads compiling the same kernel
  simultaneously can produce duplicate or corrupted objects.
- **Python callables have no home** — CuPy reductions, element-wise ops,
  and custom Python GPU functions have no analogous registry.

---

## Proposed solution: `GpuKernelRegistry`

A lightweight named registry that maps string keys to GPU kernel
specifications.  Users register kernels once (at module scope) and invoke
them by name anywhere in the event loop.

### Goals

1. Support both raw CUDA C++ kernels (JIT-compiled via CuPy NVRTC) and
   Python/CuPy callables under a single interface.
2. Compile CUDA kernels lazily — no CUDA dependency at import time.
3. Cache compiled kernels for the process lifetime with a thread-safe
   double-checked lock.
4. Work on any CuPy ndarray — raw ADC samples, calibrated panels, or
   user-allocated buffers — with no coupling to psana internals.
5. Provide a module-level singleton for process-wide sharing and support
   isolated instances for testing.

### Non-goals

- No automatic output buffer allocation (callers own memory).
- No kernel chaining or pipeline scheduling.
- No integration with the calibration dispatch path (`GPUDetector`); that
  remains a separate concern.

---

## Design

### Two kernel kinds

| Kind | Registration | Invocation | Use when |
|---|---|---|---|
| `'cuda'` | `register_cuda(name, source, entry_point)` | `launch_1d` / `launch` | Custom CUDA C++: non-trivial parallelism, warp-level ops, shared memory |
| `'callable'` | `@register_callable(name)` | `run` | CuPy element-wise, reductions, or any Python GPU function |

### Data model

```
GpuKernelRegistry
├── _specs:    dict[str → KernelSpec]    registration metadata
├── _compiled: dict[str → cp.RawKernel] compiled kernel cache
└── _lock:     threading.Lock            guards _compiled writes

KernelSpec  (frozen dataclass)
├── name              str
├── kind              'cuda' | 'callable'
├── source            str | None      CUDA C++ source  (cuda only)
├── entry_point       str | None      __global__ function name  (cuda only)
├── fn                callable | None Python function  (callable only)
├── options           tuple[str]      nvrtc compiler flags  (cuda only)
└── threads_per_block int             default block size for launch_1d
```

### Module-level singleton

```python
# Proposed export from psana.gpu:
from psana.gpu import gpu_kernel_registry   # GpuKernelRegistry instance
from psana.gpu import GpuKernelRegistry     # class for isolated instances
from psana.gpu import KernelSpec            # dataclass for introspection
```

A single `gpu_kernel_registry` instance is shared process-wide.  Tests and
isolated algorithms can create a fresh `GpuKernelRegistry()` for isolation.

---

## API

### Registration

```python
from psana.gpu import gpu_kernel_registry

# ── CUDA C++ kernel ──────────────────────────────────────────────────────────
gpu_kernel_registry.register_cuda(
    name="threshold",
    source=r"""
        extern "C" __global__
        void threshold(const float* __restrict__ in,
                       float* __restrict__ out,
                       float t,
                       unsigned long long n)
        {
            unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) out[i] = (in[i] > t) ? in[i] : 0.0f;
        }
    """,
    entry_point="threshold",
    options=("--std=c++17",),   # default; add -O3, --use_fast_math, etc. as needed
    threads_per_block=256,      # used by launch_1d to compute grid
    overwrite=False,            # raise KeyError on name collision
)

# ── Python / CuPy callable ───────────────────────────────────────────────────
@gpu_kernel_registry.register_callable("panel_mean")
def panel_mean(data_gpu, stream=None):
    import cupy as cp
    with (stream or cp.cuda.Stream.null):
        return data_gpu.mean(axis=(-2, -1))
```

- Both methods raise `KeyError` if the name is already registered and
  `overwrite=False`.
- The decorator returns the original callable unchanged so it can still be
  called directly in tests.

### Compilation

```python
# Compile all CUDA kernels eagerly — surfaces syntax errors at startup
# and eliminates first-event JIT latency.
gpu_kernel_registry.compile_all()

# Compile one kernel and get the cp.RawKernel back.
k = gpu_kernel_registry.compile("threshold")
```

Compilation is **lazy by default** (happens on the first `launch_1d` /
`launch` call) and **cached** for the process lifetime.  `compile_all()` is
an explicit opt-in for production deployments.

### Invocation

```python
import cupy as cp
import numpy as np

stream = cp.cuda.Stream(non_blocking=True)

for evt in run.events():
    with evt.gpu.get("jungfrau.calib").on_gpu_view(stream) as calib:
        n   = calib.size
        out = cp.empty(n, dtype=cp.float32)

        # ── 1-D CUDA launch ──────────────────────────────────────────────────
        # Grid is computed automatically: ceil(n_elements / threads_per_block)
        gpu_kernel_registry.launch_1d(
            "threshold",
            (calib.ravel(), out, np.float32(5.0), np.uint64(n)),
            n_elements=n,
            stream=stream,
        )

        # ── Explicit grid / block (e.g. 2-D image kernel) ────────────────────
        rows, cols = calib.shape[-2], calib.shape[-1]
        gpu_kernel_registry.launch(
            "my_2d_kernel",
            grid=(-(-cols // 16), -(-rows // 16)),  # ceil division
            block=(16, 16),
            args=(calib, out, np.int32(rows), np.int32(cols)),
            stream=stream,
        )

        # ── Python callable ───────────────────────────────────────────────────
        means = gpu_kernel_registry.run("panel_mean", calib, stream=stream)
```

**`launch_1d` grid rule:** `grid = (ceil(n_elements / threads_per_block),)`.
`threads_per_block` comes from the `KernelSpec` set at registration time.

**`run` stream forwarding:** if the callable declares a `stream` parameter
it receives `stream=stream`; otherwise the stream argument is silently
dropped.  This lets callables that ignore the stream work without change.

**Wrong-kind errors:** `run` on a `'cuda'` kernel or `launch_1d/launch` on a
`'callable'` raises `TypeError` with a message pointing to the correct
method.

### Introspection

```python
gpu_kernel_registry.names()                  # → ['threshold', 'panel_mean']
gpu_kernel_registry.is_registered("threshold")  # → True
spec = gpu_kernel_registry.get_spec("threshold")
spec.kind               # 'cuda'
spec.entry_point        # 'threshold'
spec.threads_per_block  # 256
```

---

## Thread-safety

The compilation cache is protected by a `threading.Lock` with double-checked
locking so that two threads calling `launch_1d` on the same uncompiled kernel
simultaneously do not trigger duplicate compilation:

```
Thread A                        Thread B
─────────────────────────────   ─────────────────────────────
cache.get(name) → None          cache.get(name) → None
acquire(lock)                   block on lock
cache.get(name) → None
compile → RawKernel
cache[name] = kernel
release(lock)                   acquire(lock)
                                cache.get(name) → kernel  ← reused
                                release(lock)
```

The `_specs` dict is not lock-protected.  All registrations should happen
at module import time, before the event loop starts.

---

## Integration with the event loop

The registry is **decoupled from psana internals**.  It operates on any
CuPy ndarray — from `on_gpu`, `on_gpu_view`, raw bigdata reads, or a
user-allocated buffer — and has no dependency on `DataSource`, `GPUDetector`,
or `GpuEventManager`.

### Recommended pattern

```python
# ── Module level (once per process) ──────────────────────────────────────────
from psana.gpu import gpu_kernel_registry
import numpy as np

gpu_kernel_registry.register_cuda(
    "hit_finder",
    source=r"""
        extern "C" __global__
        void hit_finder(const float* calib, float* hits,
                        float thresh, unsigned long long n) {
            unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) hits[i] = (calib[i] > thresh) ? 1.0f : 0.0f;
        }
    """,
    entry_point="hit_finder",
)
gpu_kernel_registry.compile_all()   # surface syntax errors at startup

# ── Event loop ────────────────────────────────────────────────────────────────
import cupy as cp
stream    = cp.cuda.Stream(non_blocking=True)
THRESHOLD = np.float32(5.0)

for evt in run.events():
    with evt.gpu.get("jungfrau.calib").on_gpu_view(stream) as calib:
        n    = calib.size
        hits = cp.empty(n, dtype=cp.float32)
        gpu_kernel_registry.launch_1d(
            "hit_finder",
            (calib.ravel(), hits, THRESHOLD, np.uint64(n)),
            n_elements=n,
            stream=stream,
        )
        n_hits = int(hits.sum())
```

### `on_gpu_view` vs `on_gpu`

| Accessor | Semantics | Recommended when |
|---|---|---|
| `.on_gpu_view(stream)` | Zero-copy view; slot stays leased until `__exit__` | Run a kernel in-stream; keeps ordering without extra sync |
| `.on_gpu` | Independent D→D copy; no slot dependency | Retain the array beyond the event, or feed multiple streams |

For most custom kernels `on_gpu_view` is preferred: it avoids the copy
and ensures the user kernel and the psana pipeline share the same stream
order automatically.

---

## Open questions for team review

1. **Raw bigdata access** — should the registry also expose pre-calibration
   data (uint16 ADC values) directly, or is post-calibration float32 via
   `evt.gpu.get` sufficient for the initial version?

2. **Output buffer management** — should `launch_1d` accept an optional
   `out` parameter and allocate a CuPy array when omitted, or should callers
   always manage output buffers explicitly?

3. **Kernel naming conventions** — free-form strings vs. a structured
   namespace (e.g. `"jungfrau/hit_finder"`) to avoid collisions between
   experiments?

4. **Name collision between independent analysis modules** — each MPI rank is
   a separate OS process, so the process-wide singleton is naturally isolated
   between ranks with no cross-rank interference.  The real question is
   intra-process: if two independent analysis modules loaded in the same BD
   rank both try to register a kernel under the same name (e.g. both call
   `register_cuda("threshold", ...)` with different sources), the second
   registration raises `KeyError`.  Should the registry provide a scoped
   or namespaced API (e.g. per-module sub-registries) to prevent accidental
   collisions, or is `overwrite=True` and free-form naming sufficient?



---

## Design decisions

### Why not add this to `GPUDetector`?

`GPUDetector` is tightly coupled to the calibration pipeline: it manages
calibration constants, segment maps, and pre-allocated slot buffers.  Adding
user-kernel dispatch to it would entangle user code with pipeline internals,
make the class harder to test, and prevent the registry from being used
outside the full psana pipeline (e.g. standalone GPU scripts, AMI nodes).

### Why separate `launch_1d` / `launch` / `run`?

A single unified `run()` that auto-detects kernel kind would need to infer
grid dimensions from argument types at call time.  CuPy arrays, NumPy
scalars, and size integers look identical to Python's type system.  Explicit
methods are self-documenting, produce clear error messages when misused, and
keep each call site unambiguous.

### Why lazy compilation?

`register_cuda()` is called at module import time in most analysis scripts.
Triggering NVRTC at import would add 1–2 s to startup on every node,
break CPU-only login and build nodes that have no CUDA runtime, and produce
confusing `ImportError`s when CuPy is absent.  Lazy compilation also makes
`compile_all()` a useful explicit opt-in for production deployments where
first-event latency matters.

### Why `threads_per_block` on the spec rather than at call time?

Block size is a kernel-specific tuning parameter (depends on register
usage, shared memory, and occupancy).  Attaching it to the spec at
registration time lets `launch_1d` compute the grid without the caller
repeating it at every call site, while explicit `launch` still accepts any
grid/block combination for non-1-D kernels.
