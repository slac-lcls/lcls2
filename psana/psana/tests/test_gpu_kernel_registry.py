"""
test_gpu_kernel_registry.py — pytest tests for psana.gpu kernel registry.

Test groups
-----------
CPU-only (always run, no GPU allocation needed):
  test_default_registry_contents  — default_registry() lookup & singleton
  test_custom_kernel_registration — register / get / list_registered
  test_registry_validation_errors — ValueError on empty name/det_types
  test_detector_router_resolve    — DetectorRouter.resolve_key() rules

GPU tests (skipped automatically when no CUDA device is present,
           and marked slow so they are also skipped in normal CI):
  test_jungfrau_kernel_output     — JungfrauCalibKernel matches fused_calib_gpu
  test_simple_area_kernel_formula — SimpleAreaCalibKernel formula on synthetic data
  test_custom_kernel_in_pipeline  — ScaledJungfrauKernel dispatched end-to-end
  test_detector_router_in_context — ctx.get('calib') == ctx.get('jungfrau.calib')

Run
---
    # All tests (CPU-only by default; GPU skipped if no device):
    pytest psana/psana/tests/test_gpu_kernel_registry.py

    # Include slow/GPU tests (on a GPU node):
    pytest psana/psana/tests/test_gpu_kernel_registry.py -m slow
    # or equivalently on a GPU node:
    PS_TEST_GPU_STREAM_IDS=6,8,9,10,11 pytest \\
        psana/psana/tests/test_gpu_kernel_registry.py -m slow
"""

import os

import numpy as np
import pytest

from psana.gpu.gpu_kernel_registry import (
    GPUKernel,
    GPUKernelRegistry,
    JungfrauCalibKernel,
    SimpleAreaCalibKernel,
    default_registry,
)
from psana.gpu.detector_router import DetectorRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIR        = os.path.dirname(os.path.realpath(__file__))
_JUNGFRAU   = os.path.join(_DIR, 'test_data', 'detector',
                            'test_jungfrau05M_calib.xtc2')
_SMD_GLOB = os.environ.get('PSANA_GPU_TEST_SMD_GLOB', '')
_DET_NAME   = 'jungfrau'


def _gpu_available() -> bool:
    """Return True only when a usable CUDA device is present."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


# Decorator: skip GPU test if no device, and mark as slow for CI.
requires_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason='no CUDA device available',
)


# ---------------------------------------------------------------------------
# CPU-only tests
# ---------------------------------------------------------------------------

def test_default_registry_contents():
    """default_registry() is a singleton pre-loaded with the built-in kernels."""
    reg  = default_registry()
    reg2 = default_registry()

    # Singleton.
    assert reg is reg2, 'default_registry() must return the same object each call'

    # Jungfrau → JungfrauCalibKernel.
    k = reg.get('jungfrau', 'calib')
    assert k is not None, 'jungfrau.calib must be registered'
    assert isinstance(k, JungfrauCalibKernel), (
        f'expected JungfrauCalibKernel, got {type(k).__name__}'
    )

    # Simple area detectors → SimpleAreaCalibKernel.
    for det in ('epix100', 'epix100a', 'epixhr', 'cspad', 'cspad2x2'):
        k = reg.get(det, 'calib')
        assert k is not None, f'{det}.calib must be registered'
        assert isinstance(k, SimpleAreaCalibKernel), (
            f'{det}: expected SimpleAreaCalibKernel, got {type(k).__name__}'
        )

    # Unregistered entries return None.
    assert reg.get('unknown_detector', 'calib') is None
    assert reg.get('jungfrau', 'peaks')         is None   # future result type

    # repr() shows all registered (det, name) pairs.
    r = repr(reg)
    assert 'jungfrau.calib' in r, f'repr missing jungfrau.calib: {r}'
    assert 'epix100.calib'  in r, f'repr missing epix100.calib: {r}'


def test_custom_kernel_registration():
    """Users can register custom kernels; they are isolated per registry."""

    class MyKernel(GPUKernel):
        name      = 'calib'
        det_types = ['my_detector', 'my_detector_v2']

        def calibrate(self, raw_gpu, peds_gpu, gmask_gpu, stream=None):
            pass   # not called in this test

    reg = GPUKernelRegistry()
    reg.register(MyKernel())

    # Found for both declared det_types.
    for det in ('my_detector', 'my_detector_v2'):
        k = reg.get(det, 'calib')
        assert k is not None, f'{det}.calib should be registered'
        assert type(k).__name__ == 'MyKernel', (
            f'expected MyKernel for {det}, got {type(k).__name__}'
        )

    # Custom registry is independent — does not inherit the default registry.
    assert reg.get('jungfrau', 'calib') is None, (
        'custom registry must not inherit jungfrau from default_registry()'
    )

    # list_registered() reflects exactly what was added.
    listed = reg.list_registered()
    assert ('my_detector', 'calib')    in listed
    assert ('my_detector_v2', 'calib') in listed
    assert len(listed) == 2, f'expected 2 entries, got {len(listed)}: {listed}'


def test_registry_validation_errors():
    """Registering a kernel with empty name or det_types raises ValueError."""
    reg = GPUKernelRegistry()

    class EmptyName(GPUKernel):
        name      = ''
        det_types = ['det']

    class EmptyDetTypes(GPUKernel):
        name      = 'calib'
        det_types = []

    with pytest.raises(ValueError, match='name'):
        reg.register(EmptyName())

    with pytest.raises(ValueError, match='det_types'):
        reg.register(EmptyDetTypes())


def test_detector_router_resolve():
    """DetectorRouter.resolve_key() expands unqualified keys correctly."""
    router = DetectorRouter()
    router.register_gpu('jungfrau')
    router.register_cpu('gmd')
    router.register_cpu('ipm')

    # Unqualified → prefixed with the first registered GPU detector.
    assert router.resolve_key('calib') == 'jungfrau.calib'
    assert router.resolve_key('raw')   == 'jungfrau.raw'
    assert router.resolve_key('image') == 'jungfrau.image'

    # Qualified → returned unchanged (backward compatibility).
    assert router.resolve_key('jungfrau.calib') == 'jungfrau.calib'
    assert router.resolve_key('epix.calib')     == 'epix.calib'

    # default_gpu_det is the first registered GPU detector.
    assert router.default_gpu_det == 'jungfrau'

    # gpu_det_names / cpu_det_names reflect registration order.
    assert router.gpu_det_names == ['jungfrau']
    assert 'gmd' in router.cpu_det_names
    assert 'ipm' in router.cpu_det_names

    # Empty router: unqualified key returned unchanged (no default).
    empty = DetectorRouter()
    assert empty.default_gpu_det is None
    assert empty.resolve_key('calib') == 'calib'


# ---------------------------------------------------------------------------
# GPU tests  (marked slow + skipped when no CUDA device)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@requires_gpu
def test_jungfrau_kernel_output():
    """JungfrauCalibKernel.calibrate() matches fused_calib_gpu() and CPU calib.

    Uses test_jungfrau05M_calib.xtc2 where pixel mask is all-zeros, so
    calibrated output is all-zeros regardless of pedestals.
    """
    import cupy as cp
    import psana
    from psana.gpu.gpu_calib import fused_calib_gpu, prep_calib_constants

    ds  = psana.DataSource(files=_JUNGFRAU)
    run = next(ds.runs())
    det = run.Detector(_DET_NAME)
    peds_gpu, gmask_gpu = prep_calib_constants(det)

    kernel = JungfrauCalibKernel()

    for i, evt in enumerate(run.events()):
        if i >= 2:
            break
        raw_cpu  = det.raw.raw(evt)
        calib_cpu = det.raw.calib(evt)  # all-zeros for this fixture

        raw_gpu = cp.asarray(raw_cpu)

        # Kernel must equal fused_calib_gpu directly.
        calib_k = kernel.calibrate(raw_gpu, peds_gpu, gmask_gpu)
        calib_f = fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu)
        assert cp.allclose(calib_k, calib_f, atol=1e-6), (
            f'evt={i}: JungfrauCalibKernel differs from fused_calib_gpu'
        )

        # Kernel must match CPU reference (all-zeros fixture).
        assert np.allclose(calib_k.get(), calib_cpu, atol=1e-5), (
            f'evt={i}: GPU calib does not match CPU calib (max diff '
            f'{np.abs(calib_k.get() - calib_cpu).max():.2e})'
        )


@pytest.mark.slow
@requires_gpu
def test_simple_area_kernel_formula():
    """SimpleAreaCalibKernel computes (raw − peds[mode0]) × gmask[mode0].

    Tested on synthetic arrays so no real detector data is required.
    """
    import cupy as cp

    kernel = SimpleAreaCalibKernel()
    rng = np.random.default_rng(0)

    n_segs, nrows, ncols = 2, 32, 32
    n_pix  = n_segs * nrows * ncols

    raw_np    = rng.integers(100, 3000, (n_segs, nrows, ncols),
                              dtype=np.uint16)
    peds_np   = rng.uniform(0, 2800, n_pix).astype(np.float32)
    gmask_np  = rng.uniform(0, 1,    n_pix).astype(np.float32)

    raw_gpu   = cp.asarray(raw_np)
    peds_gpu  = cp.asarray(peds_np)
    gmask_gpu = cp.asarray(gmask_np)

    calib_gpu = kernel.calibrate(raw_gpu, peds_gpu, gmask_gpu)
    calib_np  = calib_gpu.get()

    expected = (raw_np.ravel().astype(np.float32) - peds_np) * gmask_np

    assert np.allclose(calib_np.ravel(), expected, atol=1e-4), (
        f'SimpleAreaCalibKernel formula wrong; '
        f'max diff = {np.abs(calib_np.ravel() - expected).max():.2e}'
    )

    # With 3-mode calibconst the kernel still uses only mode 0.
    peds_3  = cp.asarray(np.tile(peds_np,  3).astype(np.float32))
    gmask_3 = cp.asarray(np.tile(gmask_np, 3).astype(np.float32))
    calib_3 = kernel.calibrate(raw_gpu, peds_3, gmask_3)

    assert cp.allclose(calib_3, calib_gpu, atol=1e-4), (
        'SimpleAreaCalibKernel: 3-mode calibconst should give same result as 1-mode'
    )


@pytest.mark.slow
@requires_gpu
def test_custom_kernel_in_pipeline():
    """A user-registered kernel is correctly dispatched through gpu_events().

    Registers ScaledKernel (returns 2× the Jungfrau output) and verifies the
    pipeline result is exactly 2× the default pipeline result.
    """
    import glob
    import cupy as cp
    from psana.gpu import gpu_events, GPUKernelRegistry
    from psana.gpu.gpu_calib import fused_calib_gpu

    SCALE = 2.0

    class ScaledKernel(GPUKernel):
        name      = 'calib'
        det_types = ['jungfrau']
        raw_dtype = 'uint16'

        def calibrate(self, raw_gpu, peds_gpu, gmask_gpu, stream=None):
            return fused_calib_gpu(raw_gpu, peds_gpu, gmask_gpu) * SCALE

    custom_reg = GPUKernelRegistry()
    custom_reg.register(ScaledKernel())

    smd_files = sorted(glob.glob(_SMD_GLOB))
    if not smd_files:
        pytest.skip(f'test data not found (set PSANA_GPU_TEST_SMD_GLOB): {_SMD_GLOB}')
    smd_files = list(dict.fromkeys(smd_files))

    default_results = []
    scaled_results  = []

    for ctx in gpu_events(smd_files, 'jungfrau', batch_size=5, max_events=3):
        default_results.append(ctx.get('calib').on_cpu)

    for ctx in gpu_events(smd_files, 'jungfrau', batch_size=5, max_events=3,
                          registry=custom_reg):
        scaled_results.append(ctx.get('calib').on_cpu)

    assert len(default_results) == len(scaled_results) > 0, (
        'both pipelines must produce the same number of events'
    )

    for i, (d, s) in enumerate(zip(default_results, scaled_results)):
        nonzero = d != 0
        if nonzero.any():
            ratio = s[nonzero] / d[nonzero]
            assert np.allclose(ratio, SCALE, atol=0.01), (
                f'evt={i}: scaled/default should be {SCALE} '
                f'(mean ratio={ratio.mean():.4f})'
            )


@pytest.mark.slow
@requires_gpu
def test_detector_router_in_context():
    """ctx.get('calib') and ctx.get('jungfrau.calib') return identical arrays.

    Verifies that DetectorRouter is wired into GpuEventContext and resolves
    unqualified keys correctly end-to-end through the DataSource integration.
    """
    import glob
    import cupy as cp
    from psana import DataSource

    smd_files = sorted(glob.glob(_SMD_GLOB))
    if not smd_files:
        pytest.skip(f'test data not found (set PSANA_GPU_TEST_SMD_GLOB): {_SMD_GLOB}')
    smd_files = list(dict.fromkeys(smd_files))

    ds = DataSource(files=smd_files, gpu_det='jungfrau', batch_size=5,
                    max_events=3)
    n_checked = 0
    for run in ds.runs():
        for ctx in run.events():
            unq = ctx.get('calib').on_gpu           # unqualified
            qua = ctx.get('jungfrau.calib').on_gpu  # qualified

            assert cp.array_equal(unq, qua), (
                'ctx.get("calib") and ctx.get("jungfrau.calib") must be identical'
            )

            # raw also resolves through the router.
            raw_unq = ctx.get('raw').on_gpu
            raw_qua = ctx.get('jungfrau.raw').on_gpu
            assert cp.array_equal(raw_unq, raw_qua), (
                'ctx.get("raw") and ctx.get("jungfrau.raw") must be identical'
            )

            n_checked += 1

    assert n_checked > 0, 'no events processed'


if __name__ == '__main__':
    # Allow running directly: python test_gpu_kernel_registry.py
    # This runs all tests including slow/GPU ones.
    import sys
    pytest.main([__file__, '-v', '-m', 'slow or not slow'])
