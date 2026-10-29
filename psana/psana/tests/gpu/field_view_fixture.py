"""Byte-packed fields with independently shifted bases and locator offsets."""
from types import SimpleNamespace as NS

import numpy as np

from psana.gpu.gpu_budget import _GpuBudget
from psana.gpu.gpu_input import GpuDetectorBinding, GpuDetectorEvent, InputSlotLease
from psana.gpu.gpu_input_window import InputWindow
from psana.gpu.gpu_kvikio_read import DESC_NCOLS
from psana.gpu.gpudgram.batch import (
    LOC_DIM0, LOC_NBYTES, LOC_NCOLS, LOC_OFFSET, LOC_RANK, LOC_STATUS, LOC_TYPE,
)
from psana.gpu.gpudgram.config import GpuFieldHandle
from psana.gpu.gpudgram.parser import STATUS_FOUND, STATUS_NOT_PRESENT

DTYPES = ('u1', 'u2', 'u4', 'u8', 'i1', 'i2', 'i4', 'i8', 'f4', 'f8')


def field_case(cp, *, type_id=1, shapes=((3, 5),), base_shift=0, offset=1,
               absent=(), limit=1024**3, lease_budget=True, budget=None,
               defer_retirement=True):
    dtype = np.dtype(DTYPES[type_id])
    segments = (9, 4, 17)[:len(shapes)]
    data, expected, rows, handles = {}, {}, {}, {}
    for index, (segment, shape) in enumerate(zip(segments, shapes)):
        nbytes = int(np.prod(shape)) * dtype.itemsize
        # Exact end bound; no spare bytes available to an over-wide read.
        host = np.arange(base_shift + offset + nbytes, dtype=np.uint8)
        source = cp.asarray(host)[base_shift:]
        handle = GpuFieldHandle(index, index, index, index, 0,
                                type_id, dtype.itemsize, len(shape), 0)
        row = np.zeros((1, LOC_NCOLS), dtype=np.uint64)
        row[0, LOC_STATUS] = STATUS_NOT_PRESENT if segment in absent else STATUS_FOUND
        row[0, LOC_TYPE] = type_id
        row[0, LOC_RANK] = len(shape)
        row[0, LOC_DIM0:LOC_DIM0 + len(shape)] = shape
        row[0, LOC_OFFSET] = offset
        row[0, LOC_NBYTES] = nbytes
        rows[segment] = row
        handles[segment] = handle
        # Host locator rows isolate field access from parser kernel timing.
        data[index] = NS(data_gpu=source, dgram_index=0,
                         locate=lambda handle, row=row: NS(rows_gpu=row, ready=None))
        expected[segment] = host[base_shift + offset:].view(dtype).reshape(shape)
    ready = cp.cuda.Event(disable_timing=True)
    cp.cuda.get_current_stream().record(ready)
    owner = InputWindow(0, 0, NS(n_dgrams=0, data_gpu=data, walk_done=ready),
                        np.zeros((0, DESC_NCOLS), np.uint64),
                        defer_retirement=defer_retirement)
    budget = _GpuBudget(limit) if budget is None else budget
    lease = InputSlotLease(ready, (owner,), **({'budget': budget} if lease_budget else {}))
    binding = GpuDetectorBinding('det', canonical_segment_ids=segments,
        field_handles_by_segment={}, field_handles_by_name={('raw', 'value'): handles})
    detector = GpuDetectorEvent(binding, data, lease)
    return NS(detector=detector, result=detector.field('raw', 'value'), lease=lease,
              owner=owner, budget=budget, sources=data, rows=rows, expected=expected)


def retire(case):
    case.owner.close()
    case.lease.wait_until_safe_to_reuse()
    case.owner.drain()
