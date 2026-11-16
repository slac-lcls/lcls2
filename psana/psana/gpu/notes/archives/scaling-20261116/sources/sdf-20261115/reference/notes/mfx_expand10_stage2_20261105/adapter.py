"""sample10-v1 user GpuTask: descriptor-based expansion, then full JF calibration.

No core hooks or per-event host payload copies. Input leases belong to psana;
all owned arrays are registered before submission and consumed on its stream.
"""
import importlib.util
from pathlib import Path

import numpy as np

from psana.gpu.gpu_task_batch import (
    FIELD_WORDS, FIELD_RAW_PTR, FIELD_RAW_NBYTES, FIELD_LOCATOR_PTR, FIELD_ROW,
    FIELD_TYPE, FIELD_RANK, FIELD_ELEMENT_SIZE, FIELD_SOURCE_PRESENT)
from psana.gpu.gpudgram.batch import (
    LOC_NCOLS, LOC_TYPE, LOC_RANK, LOC_DIM0, LOC_OFFSET, LOC_NBYTES, LOC_STATUS)
from psana.gpu.gpudgram.parser import STATUS_FOUND

PIXELS, WORDS = 512*1024, 52429
CALIBRATION_FILE = Path(__file__).resolve().parents[2] / 'psana/psana/gpu/examples/jungfrau_calibration.py'
spec = importlib.util.spec_from_file_location('sample10_jungfrau_calibration', CALIBRATION_FILE)
calibration_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration_module)


from psana.detector.detector_impl import DetectorImpl


class jungfrau_sim10_1_0_0(DetectorImpl):
    """Process-local reader adapter; seed arrays stay indexed by physical panel."""
    def seed(self, evt) -> dict:
        segments = self._segments(evt)
        if segments is None:
            return None
        return {panel: value.seed for panel, value in segments.items()}


def register_reader():
    """Register before DataSource constructs the normal-detector routing table.

    This affects only the current process, never the installed psana package.
    Do not reinterpret the synthetic schema as jungfrau.raw.
    """
    from psana.detector import detectors
    name = 'jungfrau_sim10_1_0_0'
    existing = getattr(detectors, name, None)
    if existing is not None and existing is not jungfrau_sim10_1_0_0:
        raise RuntimeError('A different sim10 reader is already registered')
    setattr(detectors, name, jungfrau_sim10_1_0_0)


def kernel_source():
    source = r'''
extern "C" __global__ void expand10(const unsigned long long* fields,
        unsigned short* output, unsigned char* present, unsigned int* errors,
        unsigned long long nrows) {
    const unsigned long long tiles = (@PIXELS@+255)/256;
    unsigned long long row = blockIdx.x / tiles;
    unsigned long long pixel = (blockIdx.x % tiles)*256 + threadIdx.x;
    if (row >= nrows || pixel >= @PIXELS@) return;
    const unsigned long long* f = fields + row*@FIELD_WORDS@;
    unsigned int error = 0;
    unsigned long long offset = 0;
    if (!f[@FIELD_SOURCE_PRESENT@]) error = 1;
    else if (!f[@FIELD_RAW_PTR@] || !f[@FIELD_LOCATOR_PTR@] ||
             f[@FIELD_TYPE@] != 1 || f[@FIELD_RANK@] != 1 ||
             f[@FIELD_ELEMENT_SIZE@] != 2) error = 2;
    else {
        // Framework validates the descriptor row against its locator capacity.
        const unsigned long long* loc = (const unsigned long long*)f[@FIELD_LOCATOR_PTR@]
                                      + f[@FIELD_ROW@]*@LOC_NCOLS@;
        offset = loc[@LOC_OFFSET@];
        if (loc[@LOC_STATUS@] != @STATUS_FOUND@ || loc[@LOC_TYPE@] != 1 ||
            loc[@LOC_RANK@] != 1 || loc[@LOC_DIM0@] != @WORDS@ ||
            loc[@LOC_NBYTES@] != @WORDS@*2 || offset > f[@FIELD_RAW_NBYTES@] ||
            @WORDS@*2 > f[@FIELD_RAW_NBYTES@]-offset) error = 3;
    }
    if (pixel == 0) { errors[row] = error; present[row] = error == 0; }
    unsigned short value = 0;
    if (!error) {
        // XTC scalar/array addresses need not have native C alignment.
        // UINT16 is little endian on wire: use byte loads even for odd offsets.
        const unsigned char* p = (const unsigned char*)f[@FIELD_RAW_PTR@] + offset + (pixel/10)*2;
        value = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    }
    output[row*@PIXELS@ + pixel] = value;
}
'''
    for name in ('PIXELS', 'WORDS', 'FIELD_WORDS', 'FIELD_RAW_PTR', 'FIELD_RAW_NBYTES',
                 'FIELD_LOCATOR_PTR', 'FIELD_ROW', 'FIELD_TYPE', 'FIELD_RANK',
                 'FIELD_ELEMENT_SIZE', 'FIELD_SOURCE_PRESENT', 'LOC_NCOLS', 'LOC_TYPE',
                 'LOC_RANK', 'LOC_DIM0', 'LOC_OFFSET', 'LOC_NBYTES', 'LOC_STATUS', 'STATUS_FOUND'):
        source = source.replace('@'+name+'@', str(globals()[name]))
    if '@' in source:
        raise ValueError('Unbound kernel ABI token')
    return source


class ExpandedBatch:
    """Supply materialized raw/presence to the unchanged calibration example."""
    def __init__(self, original, raw, present):
        self.original, self.raw, self.presence = original, raw, present
        self.size = original.size

    def input(self, name):
        if name != 'jungfrau.raw':
            raise KeyError(name)
        return self.raw

    def present(self, name):
        if name != 'jungfrau.raw':
            raise KeyError(name)
        return self.presence

    def segment_ids(self, name): return self.original.segment_ids(name)
    def calibconst(self, det, key): return self.original.calibconst(det, key)
    def keepalive(self, value): return self.original.keepalive(value)
    def publish(self, name, value): return self.original.publish(name, value)


class Sample10:
    inputs = (('jungfrau', 'sim10', 'seed'),)

    def __init__(self):
        self.calibration = calibration_module.JungfrauCalibration(use_offset=True)
        self.calibconst = self.calibration.calibconst
        self.kernel = None
        self.calls = 0
        self.batch_sizes = []
        self.streams = set()
        self.pool_peak_bytes = 0

    def __call__(self, batch, stream):
        import cupy as cp
        if not 1 <= batch.size <= 16:
            raise ValueError('This correctness driver bounds task subbatches to 16 events')
        segments = tuple(batch.segment_ids('jungfrau'))
        field = batch.field('jungfrau', 'sim10', 'seed')
        if segments != tuple(range(32)) or tuple(field.segment_ids) != segments:
            raise ValueError('Expected all 32 canonical physical panels')
        if field.rows.dtype != cp.uint64 or field.rows.shape != (batch.size, 32, FIELD_WORDS):
            raise ValueError('Unexpected descriptor ABI')
        with stream:
            if self.kernel is None:
                self.kernel = cp.RawKernel(kernel_source(), 'expand10', options=('--std=c++17',))
            raw = cp.empty((batch.size, 32, 512, 1024), cp.uint16)
            present = cp.empty((batch.size, 32), cp.uint8)
            errors = cp.empty((batch.size, 32), cp.uint32)
            # Publication/keepalive must precede all submitted operations.
            batch.publish('sim10.expanded', raw)
            batch.publish('sim10.errors', errors)
            batch.keepalive(present)
            self.kernel((batch.size*32*((PIXELS+255)//256),), (256,),
                        (field.rows, raw, present, errors, np.uint64(batch.size*32)), stream=stream)
            view = ExpandedBatch(batch, raw, present)
            self.calibration.calibrate(view, stream, publish=True)
        self.calls += 1
        self.batch_sizes.append(batch.size)
        self.streams.add(int(stream.ptr))
        self.pool_peak_bytes = max(self.pool_peak_bytes, cp.get_default_memory_pool().total_bytes())
