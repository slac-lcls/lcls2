"""User-owned batched Jungfrau calibration; copy this file outside psana.

Only NumPy is imported during declaration. The callback uses public batch APIs;
CUDA initialization/compilation happens lazily on the assigned worker. Constants
retain their original (gain mode, physical segment, row, column) layout.

Formula: float32(ADC - float32(pedestal + optional_offset)) * float32(1/gain).
Gain bits 00/01/11 select planes 0/1/2; 10, missing segments, zero gain and
masked pixels with finite constants produce zero. Optional pixel_status and
status_extra masking ORs selected bits across all three gain planes. Constant
addition/division use source precision before rounding to float32, like CPU v3. No common mode, geometry, edge or neighbor mask is applied.
"""
from numbers import Integral

import numpy as np


_FLOAT_TYPES = {np.dtype('float32'): 'float', np.dtype('float64'): 'double'}
_STATUS_TYPES = {np.dtype(k): v for k, v in (
    ('uint8', 'unsigned char'), ('uint16', 'unsigned short'),
    ('uint32', 'unsigned int'), ('uint64', 'unsigned long long'))}


class JungfrauCalibration:
    """Callable for GpuTask; constructor and constant declarations are host-only.

    ``use_offset``, ``status_bits`` and ``stextra_bits`` request additional constants explicitly.
    A requested key missing from the run is an error, never silently ignored.
    This first example publishes full float32 images, so large batches may use
    the framework's ordinary-host fallback under its default pinned-byte cap.
    User output allocation is outside the framework device-memory budget.
    """
    def __init__(self, detector='jungfrau', *, output='jungfrau.calib',
                 use_offset=False, status_bits=0, stextra_bits=0):
        if not isinstance(detector, str) or not detector or detector.strip() != detector:
            raise ValueError('detector must be a nonempty name without surrounding whitespace')
        if not isinstance(output, str) or not output or output.strip() != output:
            raise ValueError('output must be a nonempty name without surrounding whitespace')
        if output == detector + '.raw':
            raise ValueError('output cannot use the reserved dense input name')
        if not isinstance(use_offset, bool):
            raise TypeError('use_offset must be bool')
        for name, bits in (('status_bits', status_bits), ('stextra_bits', stextra_bits)):
            if isinstance(bits, bool) or not isinstance(bits, Integral):
                raise TypeError(f'{name} must be an integer bit mask')
            if not 0 <= bits <= np.iinfo(np.uint64).max:
                raise ValueError(f'{name} must fit uint64')
        self.detector, self.output = detector, output
        self.use_offset, self.status_bits = use_offset, int(status_bits)
        self.stextra_bits = int(stextra_bits)
        self.calls = self.events = 0
        self._kernels = {}  # Modules only; no borrowed inputs/constants cached here.

    @property
    def inputs(self):
        return (self.detector + '.raw',)

    @property
    def calibconst(self):
        keys = ['pedestals', 'pixel_gain']
        if self.use_offset:
            keys.append('pixel_offset')
        if self.status_bits:
            keys.append('pixel_status')
        if self.stextra_bits:
            keys.append('status_extra')
        return tuple((self.detector, key) for key in keys)

    def _layout(self, raw, present, segments, constants, size):
        """Host metadata checks only: never copy device values for validation."""
        if raw.dtype != np.uint16 or raw.ndim != 4 or not raw.flags.c_contiguous:
            raise ValueError('raw must be C-contiguous uint16 (events, segments, rows, columns)')
        if raw.shape[0] != size or any(n <= 0 for n in raw.shape):
            raise ValueError('raw must contain the nonempty selected batch')
        if present.dtype != np.uint8 or present.shape != raw.shape[:2] or not present.flags.c_contiguous:
            raise ValueError('presence must be C-contiguous uint8 (events, segments)')
        if (len(segments) != raw.shape[1] or len(set(segments)) != len(segments) or
                any(isinstance(s, bool) or not isinstance(s, Integral) or s < 0 for s in segments)):
            raise ValueError('segment IDs must be unique nonnegative physical indices')
        peds = constants['pedestals']
        if (peds.ndim != 4 or peds.shape[0] != 3 or peds.shape[2:] != raw.shape[2:] or
                max(segments) >= peds.shape[1]):
            raise ValueError('pedestals must have shape (3, physical segments, rows, columns)')
        ctypes = {}
        for key, value in constants.items():
            types = _STATUS_TYPES if key in ('pixel_status', 'status_extra') else _FLOAT_TYPES
            if value.shape != peds.shape or not value.flags.c_contiguous:
                raise ValueError(f'{key} must have the same contiguous physical layout as pedestals')
            if value.dtype not in types:
                raise TypeError(f'{key}: unsupported dtype {value.dtype}')
            ctypes[key] = types[value.dtype]
        return ctypes

    def __call__(self, batch, stream):
        import cupy as cp
        raw = batch.input(self.inputs[0])
        present = batch.present(self.inputs[0])
        segments = tuple(batch.segment_ids(self.detector))
        constants = {key: batch.calibconst(det, key) for det, key in self.calibconst}
        ctypes = self._layout(raw, present, segments, constants, batch.size)
        peds, gain = constants['pedestals'], constants['pixel_gain']
        offset = constants.get('pixel_offset', peds)  # Unused pointer when disabled.
        status = constants.get('pixel_status', raw)
        extra = constants.get('status_extra', raw)
        # Layout-specialized modules embed the tiny host segment map. There is
        # no per-event launch, constant conversion, segment-map upload or D2H.
        key = (raw.device.id, segments, tuple(ctypes.items()))
        with stream:
            kernel = self._kernels.get(key)
            if kernel is None:
                kernel = cp.RawKernel(self._source(segments, ctypes), 'calibrate',
                                      options=('--std=c++17', '--fmad=false'))
                self._kernels[key] = kernel
            out = cp.empty(raw.shape, dtype=cp.float32)
            batch.publish(self.output, out)  # Register ownership before submission.
            kernel((min(65535, (raw.size + 255) // 256),), (256,),
                   (raw, present, peds, gain, offset, status, out,
                    np.uint64(raw.size), np.uint64(raw.shape[2] * raw.shape[3]),
                    np.uint64(peds.shape[1]), np.uint64(self.status_bits),
                    extra, np.uint64(self.stextra_bits)), stream=stream)
        self.calls += 1
        self.events += batch.size

    def _source(self, segments, types):
        # Adapted from Amanda Shackelford's jungfrau_calib_pixel at
        # d5437f99dafa68c3071e437ae034457735f00888 (cuda/fused_calib.cuh).
        # This variant reads original constants and spans the event axis.
        source = r'''
extern "C" __global__ void calibrate(
    const unsigned short* raw, const unsigned char* present,
    const @PEDS@* peds, const @GAIN@* gain, const @OFFSET@* offset,
    const @STATUS@* status, float* out, unsigned long long total,
    unsigned long long panel_pixels, unsigned long long physical_segments,
    unsigned long long status_bits, const @EXTRA@* extra,
    unsigned long long stextra_bits)
{
    const unsigned long long ids[] = {@SEGMENTS@};
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned long long row = i / panel_pixels;
        unsigned long long s = row % @NSEG@;
        unsigned long long pixel = ids[s] * panel_pixels + i % panel_pixels;
        unsigned long long plane = physical_segments * panel_pixels;
        unsigned int bits = raw[i] >> 14;
        if (!present[row] || bits == 2) { out[i] = 0.0f; continue; }
        bool masked = (@USE_STATUS@ && (((unsigned long long)status[pixel] |
                           (unsigned long long)status[plane + pixel] |
                           (unsigned long long)status[2 * plane + pixel]) & status_bits)) ||
                      (@USE_EXTRA@ && (((unsigned long long)extra[pixel] |
                           (unsigned long long)extra[plane + pixel] |
                           (unsigned long long)extra[2 * plane + pixel]) & stextra_bits));
        unsigned int mode = bits == 3 ? 2 : bits;
        unsigned long long k = mode * plane + pixel;
        // CPU v3 prepares poff/gfac in NumPy source precision, then packs
        // float32 constants. Casting each input first changes float64 results.
        float ped = @USE_OFFSET@ ? (float)(peds[k] + offset[k]) : (float)peds[k];
        @DIVTYPE@ inverse_source = gain[k] == 0 ? 0 : @DIV@(1, gain[k]);
        float inverse = (float)(inverse_source * (masked ? 0 : 1));
        out[i] = ((float)(raw[i] & 0x3fff) - ped) * inverse;
    }
}
'''
        values = dict(PEDS=types['pedestals'], GAIN=types['pixel_gain'],
                      OFFSET=types.get('pixel_offset', types['pedestals']),
                      STATUS=types.get('pixel_status', 'unsigned char'),
                      EXTRA=types.get('status_extra', 'unsigned char'),
                      DIVTYPE='double' if 'double' in (types['pedestals'], types['pixel_gain']) else 'float',
                      DIV='__ddiv_rn' if 'double' in (types['pedestals'], types['pixel_gain']) else '__fdiv_rn',
                      SEGMENTS=','.join(str(int(s)) for s in segments), NSEG=str(len(segments)),
                      USE_OFFSET='true' if self.use_offset else 'false',
                      USE_STATUS='true' if self.status_bits else 'false',
                      USE_EXTRA='true' if self.stextra_bits else 'false')
        for name, value in values.items():
            source = source.replace('@' + name + '@', value)
        return source
