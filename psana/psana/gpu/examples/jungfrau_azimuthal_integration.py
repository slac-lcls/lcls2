"""External user calibration + batched sorted-bin reduction, with no psana imports.

Supply an explicit physical-segment bin map. No geometry or beam parameters are
invented. Each callback submits two kernels for the entire subbatch and publishes
float64 (N, 3, nbins): mean intensity, intensity sum, valid-pixel count.
"""
from numbers import Integral
import numpy as np

if __package__:
    from .jungfrau_calibration import JungfrauCalibration
else:
    from jungfrau_calibration import JungfrauCalibration


def radial_bin_ids(x_mm, y_mm, edges_mm, *, center_mm):
    """Build physical-segment bins on the CPU; intervals are [left, right).

    Coordinates must be real, matching (physical segments, rows, columns) arrays
    in millimeters. The center is explicit. Nonfinite/out-of-range coordinates
    are excluded with -1. For q integration callers can supply a q-derived bin
    map instead, with their own confirmed wavelength and geometry.
    """
    x, y = np.asarray(x_mm, dtype=np.float64), np.asarray(y_mm, dtype=np.float64)
    edges = np.asarray(edges_mm, dtype=np.float64)
    center = np.asarray(center_mm, dtype=np.float64)
    if x.ndim != 3 or x.shape != y.shape or any(n <= 0 for n in x.shape):
        raise ValueError('coordinates must have matching nonempty (P,H,W) shapes')
    if (edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or
            not np.all(np.diff(edges) > 0) or len(edges) - 1 > np.iinfo(np.int32).max):
        raise ValueError('edges must be finite and strictly increasing')
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError('center_mm must contain two finite coordinates')
    radius = np.hypot(x - center[0], y - center[1])
    ids = np.searchsorted(edges, radius, side='right') - 1
    good = np.isfinite(radius) & (radius >= edges[0]) & (radius < edges[-1])
    return np.where(good, ids, -1).astype(np.int32)


class JungfrauAzimuthalIntegration:
    """Host-only declaration; device tables are initialized on assigned workers.

    bin_ids has original physical segment layout (P,H,W), integer entries -1
    (excluded) or 0..nbins-1. It is copied at construction and stays fixed for
    this callable. Construct a new callable for a new geometry/binning policy.
    Calibration constants are retrieved anew for every subbatch.
    """
    def __init__(self, bin_ids, nbins, detector='jungfrau', *, output='jungfrau.azint',
                 use_offset=False, status_bits=0, stextra_bits=0):
        if isinstance(nbins, bool) or not isinstance(nbins, Integral) or not 0 < nbins <= 65535:
            raise ValueError('nbins must be an integer in [1,65535]')
        bins = np.asarray(bin_ids)
        if (bins.ndim != 3 or any(n <= 0 for n in bins.shape) or
                not np.issubdtype(bins.dtype, np.signedinteger) or
                bins.size > np.iinfo(np.int32).max or np.any(bins < -1) or np.any(bins >= nbins)):
            raise ValueError('bin_ids must be signed integer (P,H,W) in [-1,nbins) and fit int32 indexing')
        self.bin_ids = np.array(bins, dtype=np.int32, order='C', copy=True)
        self.bin_ids.flags.writeable = False
        self.nbins = int(nbins)
        self.calibration = JungfrauCalibration(detector, output=output, use_offset=use_offset,
                                              status_bits=status_bits, stextra_bits=stextra_bits)
        self.inputs, self.calibconst = self.calibration.inputs, self.calibration.calibconst
        self.detector, self.output = detector, output
        self.calls = self.events = 0
        self._tables = {}  # Owned immutable arrays plus upload-completion events.
        self._reductions = {}

    def _host_table(self, segments, panel_shape):
        if (tuple(panel_shape) != self.bin_ids.shape[1:] or not segments or
                len(set(segments)) != len(segments) or any(
                    isinstance(s, bool) or not isinstance(s, Integral) or
                    s < 0 or s >= self.bin_ids.shape[0] for s in segments)):
            raise ValueError('input physical segments/panel shape do not match the bin map')
        bins = self.bin_ids[list(segments)].ravel()
        selected = np.flatnonzero(bins >= 0)
        order = selected[np.argsort(bins[selected], kind='stable')].astype(np.int32)
        offsets = np.zeros(self.nbins + 1, np.int32)
        offsets[1:] = np.cumsum(np.bincount(bins[selected], minlength=self.nbins))
        return order, offsets

    def __call__(self, batch, stream):
        import cupy as cp
        raw = batch.input(self.inputs[0])
        segments = tuple(batch.segment_ids(self.detector))
        device = raw.device.id
        key = (device, segments, raw.shape[2:])
        with stream:
            tables = self._tables.get(key)
            if tables is None:
                order, offsets = self._host_table(segments, raw.shape[2:])
                order_d, offsets_d = cp.empty(order.shape, cp.int32), cp.empty(offsets.shape, cp.int32)
                batch.keepalive(order, offsets, order_d, offsets_d)
                if order.size:
                    order_d.set(order, stream=stream)
                offsets_d.set(offsets, stream=stream)
                ready = cp.cuda.Event(disable_timing=True)
                ready.record(stream)
                tables = (order_d, offsets_d, ready, order, offsets)
                self._tables[key] = tables
            stream.wait_event(tables[2])  # A later slot may use a different stream.
            batch.keepalive(tables)
            kernel = self._reductions.get(device)
            if kernel is None:
                kernel = cp.RawKernel(_REDUCE, 'integrate', options=('--std=c++17', '--fmad=false'))
                self._reductions[device] = kernel
            image, valid = self.calibration.calibrate(batch, stream, with_validity=True)
            out = cp.empty((batch.size, 3, self.nbins), dtype=cp.float64)
            batch.publish(self.output, out)
            kernel((min(batch.size * self.nbins, 65535),), (256,),
                   (image, valid, tables[0], tables[1], out, np.uint64(raw[0].size),
                    np.uint64(batch.size * self.nbins), np.int32(self.nbins)), stream=stream)
        self.calls += 1
        self.events += batch.size


# Adapted from Amanda Shackelford's azint_sorted_kernel (650c76780). Gather
# directly from the calibrated image, count dynamic validity, and normalize in
# the same block. One grid covers all (event, bin) pairs. No atomic operations.
_REDUCE = r'''
extern "C" __global__ void integrate(
    const float* image, const unsigned char* valid, const int* order,
    const int* offsets, double* out, unsigned long long pixels,
    unsigned long long pairs, int nbins)
{
    __shared__ double sums[256];
    __shared__ unsigned int counts[256];
    for (unsigned long long pair = blockIdx.x; pair < pairs; pair += gridDim.x) {
        unsigned long long event = pair / nbins;
        int bin = pair % nbins;
        double sum = 0;
        unsigned int count = 0;
        for (unsigned long long j = (unsigned long long)offsets[bin] + threadIdx.x;
             j < (unsigned long long)offsets[bin+1]; j += blockDim.x) {
            unsigned long long p = event * pixels + (unsigned int)order[j];
            if (valid[p]) { sum += (double)image[p]; ++count; }
        }
        sums[threadIdx.x] = sum;
        counts[threadIdx.x] = count;
        __syncthreads();
        for (int stride = 128; stride; stride >>= 1) {
            if (threadIdx.x < stride) {
                sums[threadIdx.x] += sums[threadIdx.x + stride];
                counts[threadIdx.x] += counts[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            unsigned long long k = event * 3 * nbins + bin;
            out[k] = counts[0] ? sums[0] / counts[0] : 0;
            out[k + nbins] = sums[0];
            out[k + 2 * nbins] = counts[0];
        }
        __syncthreads();
    }
}
'''
