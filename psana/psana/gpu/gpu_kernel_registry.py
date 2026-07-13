"""
gpu_kernel_registry.py — Per-detector GPU calibration kernel registry.

This module implements the GPUKernelRegistry described in the psana2 GPU BD
implementation guide §4b ("Custom GPU Kernels / GPUKernelRegistry").

Architecture
------------
GPUKernel
    Abstract base.  Subclass to add GPU calibration support for a new
    detector type.  Every kernel declares:
      - name       : result-key suffix (e.g. 'calib', 'peaks')
      - det_types  : list of psana2 detector-type strings it handles
      - raw_dtype  : expected raw-pixel numpy dtype ('uint16' or 'uint32')

GPUFileKernel  (extends GPUKernel)
    Convenience base for kernels whose CUDA source lives in an external
    .cu / .cuh file.  Set class attributes kernel_file and kernel_func;
    access self.kernel (compiled + cached on first use) in calibrate().

gpu_kernel_from_file()
    Factory function.  Creates a GPUKernel directly from a .cu file whose
    __global__ function uses the standard calibration interface
    (raw, peds, gmask, calib, npixels).  No Python subclassing required.

GPUKernelRegistry
    Maps (det_type, result_name) → GPUKernel.  Looked up by GPUDetector
    to choose the right calibration algorithm for the current detector.

Built-in kernels
----------------
JungfrauCalibKernel     — 3-gain-mode calibration (gain bits 15:14 of uint16)
                          Registered for: 'jungfrau'
SimpleAreaCalibKernel   — Single-gain-mode: (raw − peds) × gmask
                          Registered for: 'epix100', 'epix100a', 'epixhr',
                          'cspad', 'cspad2x2', 'generic_area'

default_registry()
    Returns (and caches) a GPUKernelRegistry pre-loaded with the two
    built-in kernels above.  Used by DataSource(gpu_det=...).

Custom kernel example
---------------------
    from psana.gpu.gpu_kernel_registry import GPUKernel, GPUKernelRegistry

    class MyEpix10kKernel(GPUKernel):
        name      = 'calib'
        det_types = ['epix10k', 'epix10k2M']
        raw_dtype = 'uint16'

        def calibrate(self, raw_gpu, peds_gpu, gmask_gpu, stream=None):
            # epix10k gain encoding is different from Jungfrau: implement here
            ...

    reg = GPUKernelRegistry()
    reg.register(MyEpix10kKernel())

Guide reference: §4b.
"""

from functools import lru_cache


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class GPUKernel:
    """Abstract base class for GPU calibration and analysis kernels.

    Subclass this and register an instance with GPUKernelRegistry to add
    GPU support for a new detector type.

    Class attributes to override
    ----------------------------
    name : str
        Result-key suffix.  ctx.get(f'{det_name}.{name}') dispatches to
        this kernel.  E.g. 'calib', 'peaks'.
    det_types : list[str]
        Psana2 detector-type names this kernel handles.
        E.g. ['epix100', 'epix100a'].
    raw_dtype : str
        Expected raw-pixel numpy dtype: 'uint16' (default) or 'uint32'.
    """

    name:      str  = ''
    det_types: list = []
    raw_dtype: str  = 'uint16'

    def setup(self, det, gpu_detector):
        """Called once per run at BeginRun.

        Override to transfer detector-specific constants (gain thresholds,
        lookup tables, etc.) to GPU before the first event.

        Parameters
        ----------
        det          : psana2 Detector object
        gpu_detector : GPUDetector — has .peds_gpu, .gmask_gpu, .det_shape
        """

    def calibrate(self, raw_gpu, peds_gpu, gmask_gpu, stream=None, out=None):
        """Apply calibration to one event's raw pixels on GPU.

        Parameters
        ----------
        raw_gpu   : cp.ndarray, dtype = self.raw_dtype, shape (n_segs, nrows, ncols)
        peds_gpu  : cp.ndarray float32, flat, length n_modes * npixels
        gmask_gpu : cp.ndarray float32, flat, same length as peds_gpu
        stream    : cp.cuda.Stream or None
        out       : cp.ndarray float32 or None
            Pre-allocated output buffer (same size as raw_gpu, flat).
            When provided the kernel writes directly into this buffer instead
            of allocating a new array, avoiding CuPy pool growth.

        Returns
        -------
        cp.ndarray float32, same shape as raw_gpu
        """
        raise NotImplementedError(
            f'{type(self).__name__}.calibrate() is not implemented'
        )


# ---------------------------------------------------------------------------
# File-based kernel support
# ---------------------------------------------------------------------------

class GPUFileKernel(GPUKernel):
    """Convenience base for kernels whose CUDA source lives in an external file.

    Set the following class attributes on your subclass:

        kernel_file  : str | Path
            Path to the .cu or .cuh file containing the __global__ function.
            Relative paths are resolved relative to the subclass's module.
        kernel_func  : str
            Name of the ``extern "C" __global__`` entry-point function.
        compile_opts : tuple[str]
            NVRTC compile options (default: ``('--std=c++17',)``).

    The kernel is compiled JIT on first access to ``self.kernel`` and cached
    for the process lifetime.  Override ``calibrate()`` to control the
    launch parameters (grid size, block size, argument order).

    Example
    -------
    .. code-block:: python

        # File: my_epix10k_calib.cu
        # extern "C" __global__
        # void epix10k_calib_kernel(const unsigned short* raw,
        #                           const float* peds, const float* gmask,
        #                           float* calib, unsigned long long npixels) { ... }

        class Epix10kKernel(GPUFileKernel):
            name         = 'calib'
            det_types    = ['epix10k', 'epix10k2M']
            kernel_file  = '/path/to/my_epix10k_calib.cu'
            kernel_func  = 'epix10k_calib_kernel'

            def calibrate(self, raw_gpu, peds_gpu, gmask_gpu, stream=None):
                import cupy as cp, numpy as np
                npixels = int(raw_gpu.size)
                calib   = cp.empty(npixels, dtype=cp.float32)
                blocks  = (npixels + 255) // 256
                ctx = stream if stream is not None else cp.cuda.Stream.null
                with ctx:
                    self.kernel(
                        (blocks,), (256,),
                        (raw_gpu.ravel(), peds_gpu, gmask_gpu,
                         calib, np.uint64(npixels)),
                    )
                return calib.reshape(raw_gpu.shape)

        reg = GPUKernelRegistry()
        reg.register(Epix10kKernel())
    """

    kernel_file:  str   = ''                   # path to .cu / .cuh
    kernel_func:  str   = ''                   # __global__ function name
    compile_opts: tuple = ('--std=c++17',)     # NVRTC options

    @property
    def kernel(self):
        """Return the compiled CuPy RawKernel (compiled + cached on first access)."""
        if not hasattr(self, '_kernel_cache') or self._kernel_cache is None:
            self._kernel_cache = self._compile()
        return self._kernel_cache

    def _compile(self):
        import cupy as cp
        from pathlib import Path as _Path
        import inspect

        if not self.kernel_file:
            raise ValueError(
                f'{type(self).__name__}.kernel_file must be set'
            )
        if not self.kernel_func:
            raise ValueError(
                f'{type(self).__name__}.kernel_func must be set'
            )

        path = _Path(self.kernel_file)
        if not path.is_absolute():
            # Relative paths: resolve from the subclass's defining module.
            caller_file = inspect.getfile(type(self))
            path = _Path(caller_file).parent / path

        source = path.read_text()
        return cp.RawKernel(source, self.kernel_func,
                             options=tuple(self.compile_opts))


def gpu_kernel_from_file(cuda_file, func_name, det_types,
                          name='calib', raw_dtype='uint16',
                          threads_per_block=256,
                          compile_opts=('--std=c++17',)):
    """Create a GPUKernel directly from a CUDA source file.

    The CUDA function must use the **standard calibration interface**:

    .. code-block:: c

        extern "C" __global__
        void <func_name>(
            const unsigned short* raw,    // flat raw pixels (uint16)
            const float*          peds,   // flat pedestals  (n_modes × npixels)
            const float*          gmask,  // flat gain×mask  (n_modes × npixels)
            float*                calib,  // output          (npixels)
            unsigned long long    npixels
        )

    This is identical to ``jungfrau_calib_kernel`` in
    ``cuda/fused_calib.cuh``, so any psana2-style calibration kernel
    works without modification.  For ``uint32`` raw data change the first
    argument type in the CUDA file and pass ``raw_dtype='uint32'``.

    No Python subclassing is required — the function returns a ready-to-use
    GPUKernel instance.

    Parameters
    ----------
    cuda_file         : str | Path  — path to the .cu file
    func_name         : str         — name of the __global__ function
    det_types         : list[str]   — detector type names (e.g. ['epix10k'])
    name              : str         — result-key suffix (default 'calib')
    raw_dtype         : str         — 'uint16' (default) or 'uint32'
    threads_per_block : int         — CUDA block size (default 256)
    compile_opts      : tuple[str]  — NVRTC options (default ('--std=c++17',))

    Returns
    -------
    GPUFileKernel instance — register it with GPUKernelRegistry.

    Example
    -------
    .. code-block:: python

        from psana.gpu.gpu_kernel_registry import (
            GPUKernelRegistry,
            gpu_kernel_from_file,
        )

        kernel = gpu_kernel_from_file(
            '/path/to/my_epix10k_calib.cu',
            func_name = 'my_epix10k_calib_kernel',
            det_types = ['epix10k', 'epix10k2M'],
        )
        reg = GPUKernelRegistry()
        reg.register(kernel)
    """
    from pathlib import Path as _Path
    import inspect

    # Resolve relative paths from the caller's file.
    path = _Path(cuda_file)
    if not path.is_absolute():
        caller_frame = inspect.stack()[1]
        path = _Path(caller_frame.filename).parent / path

    _tpb   = int(threads_per_block)
    _dtype = str(raw_dtype)
    _path  = str(path)
    _func  = func_name
    _opts  = tuple(compile_opts)

    class _FileKernel(GPUFileKernel):
        """Auto-generated kernel from gpu_kernel_from_file()."""

    _FileKernel.name         = name
    _FileKernel.det_types    = list(det_types)
    _FileKernel.raw_dtype    = _dtype
    _FileKernel.kernel_file  = _path
    _FileKernel.kernel_func  = _func
    _FileKernel.compile_opts = _opts

    def _calibrate(self, raw_gpu, peds_gpu, gmask_gpu, stream=None):
        import cupy as cp
        import numpy as np
        npixels = int(raw_gpu.size)
        calib   = cp.empty(npixels, dtype=cp.float32)
        blocks  = (npixels + _tpb - 1) // _tpb
        ctx     = stream if stream is not None else cp.cuda.Stream.null
        with ctx:
            self.kernel(
                (blocks,), (_tpb,),
                (raw_gpu.ravel(), peds_gpu.ravel(), gmask_gpu.ravel(),
                 calib, np.uint64(npixels)),
            )
        return calib.reshape(raw_gpu.shape)

    _FileKernel.calibrate    = _calibrate
    _FileKernel.__name__     = f'_FileKernel_{_func}'
    _FileKernel.__qualname__ = _FileKernel.__name__

    instance = _FileKernel()
    instance._kernel_cache = None
    return instance


# ---------------------------------------------------------------------------
# Built-in: Jungfrau (3-mode, gain bits 15:14)
# ---------------------------------------------------------------------------

class JungfrauCalibKernel(GPUKernel):
    """Jungfrau gain-mode calibration (3 modes, gain bits in top 2 bits).

    Implements the formula from cuda/fused_calib.cuh:

        mode  = raw >> 14           # 00→g0, 01→g1, 11→g2, 10→bad(0)
        calib = (raw & 0x3fff − peds[mode]) × gmask[mode]

    peds_gpu and gmask_gpu must have shape (3 * npixels,) in mode-major
    order: [mode0_pix0..N, mode1_pix0..N, mode2_pix0..N].
    """

    name      = 'calib'
    det_types = ['jungfrau']
    raw_dtype = 'uint16'

    def calibrate(self, raw_gpu, peds_gpu, gmask_gpu, stream=None, out=None):
        from psana.gpu.gpu_calib import fused_calib_gpu
        import cupy as cp
        ctx = stream if stream is not None else cp.cuda.Stream.null
        with ctx:
            return fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu, out=out)


# ---------------------------------------------------------------------------
# Built-in: simple single-gain-mode area detector
# ---------------------------------------------------------------------------

class SimpleAreaCalibKernel(GPUKernel):
    """Single-gain-mode area detector calibration.

    Implements:
        calib[i] = (float(raw[i]) − peds[i]) × gmask[i]

    using only gain mode 0 from the calibration constants.  Appropriate for
    detectors without per-pixel gain-bit switching:

      - ePix100a  (16-bit ADC, single gain)
      - ePix HR   (16-bit ADC, single gain)
      - CSPAD     (16-bit ADC, single gain)
      - generic_area (catch-all for testing / unknown detectors)

    peds_gpu and gmask_gpu may have n_modes > 1 (the registry will pass
    whatever prep_calib_constants() produces); only mode 0 is used.
    """

    name      = 'calib'
    det_types = [
        'epix100', 'epix100a',
        'epixhr',
        'cspad', 'cspad2x2',
        'generic_area',   # catch-all for testing / unknown uint16 detectors
    ]
    raw_dtype = 'uint16'

    def calibrate(self, raw_gpu, peds_gpu, gmask_gpu, stream=None, out=None):
        import cupy as cp
        npixels  = int(raw_gpu.size)
        peds_m0  = peds_gpu[:npixels]
        gmask_m0 = gmask_gpu[:npixels]

        ctx = stream if stream is not None else cp.cuda.Stream.null
        with ctx:
            raw_f32 = raw_gpu.astype(cp.float32).ravel()
            result  = (raw_f32 - peds_m0) * gmask_m0
            if out is not None and out.size >= npixels and out.dtype == cp.float32:
                out.ravel()[:npixels] = result
                return out.reshape(raw_gpu.shape)
            return result.reshape(raw_gpu.shape)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class GPUKernelRegistry:
    """Maps (det_type, result_name) → GPUKernel.

    Used by GPUDetector to choose the right calibration algorithm for the
    current detector type without hard-coding detector names in the pipeline.

    Usage
    -----
    Register built-in or custom kernels:

        reg = GPUKernelRegistry()
        reg.register(JungfrauCalibKernel())      # det_type='jungfrau'
        reg.register(SimpleAreaCalibKernel())    # det_type='epix100', ...
        reg.register(MyEpix10kKernel())          # custom

    Look up a kernel:

        kernel = reg.get('jungfrau', 'calib')   # → JungfrauCalibKernel()
        kernel = reg.get('epix100',  'calib')   # → SimpleAreaCalibKernel()
        kernel = reg.get('unknown',  'calib')   # → None

    Guide reference: §4b.
    """

    def __init__(self):
        self._kernels: dict = {}   # {(det_type, name): GPUKernel}

    def register(self, kernel: GPUKernel):
        """Register kernel for each of its declared detector types.

        Parameters
        ----------
        kernel : GPUKernel (instance)
            Must have non-empty .name and .det_types.
        """
        if not kernel.name:
            raise ValueError(f'{type(kernel).__name__}.name must be non-empty')
        if not kernel.det_types:
            raise ValueError(
                f'{type(kernel).__name__}.det_types must be non-empty'
            )
        for det_type in kernel.det_types:
            self._kernels[(det_type, kernel.name)] = kernel

    def get(self, det_type: str, result_name: str = 'calib'):
        """Return the kernel registered for (det_type, result_name), or None.

        Parameters
        ----------
        det_type    : str, e.g. 'jungfrau', 'epix100a'
        result_name : str, default 'calib'
        """
        return self._kernels.get((det_type, result_name))

    def list_registered(self) -> list:
        """Return sorted list of (det_type, result_name) pairs."""
        return sorted(self._kernels.keys())

    def __repr__(self) -> str:
        entries = ', '.join(f'{d}.{n}' for d, n in self.list_registered())
        return f'GPUKernelRegistry([{entries}])'


# ---------------------------------------------------------------------------
# Default registry (singleton)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def default_registry() -> GPUKernelRegistry:
    """Return (and cache) the default GPUKernelRegistry.

    Pre-populated with:
      JungfrauCalibKernel   → det_type 'jungfrau'
      SimpleAreaCalibKernel → det_types 'epix100', 'epix100a', 'epixhr',
                                        'cspad', 'cspad2x2', 'generic_area'

    The registry is a module-level singleton used automatically by
    DataSource(gpu_det=...).
    """
    reg = GPUKernelRegistry()
    reg.register(JungfrauCalibKernel())
    reg.register(SimpleAreaCalibKernel())
    return reg
