"""Mixed-rate GPUBAT1 fixture retaining physical stream offsets."""
import struct

import pytest

from psana.gpu.gpu_batch import (
    GPU_BATCH_MAGIC, GPU_BATCH_VERSION, GPU_HEADER_NBYTES,
    GPU_EVENT_NBYTES, GPU_DESC_NBYTES,
)


def _mixed_packet(n_events=1000, fast_size=1, slow_size=1000, interval=100,
                 missing_fast=(), timestamp_base=1000, offsets=(0, 0)):
    events, descs = [], []
    offsets = list(offsets)
    for i in range(n_events):
        first, mask = len(descs), 0
        for stream, size, present in ((0, fast_size, i not in missing_fast),
                                      (1, slow_size, i % interval == interval - 1)):
            if present:
                descs.append((i, stream, offsets[stream], size, 0, 1, 0))
                offsets[stream] += size
                mask |= 1 << stream
        events.append((i, timestamp_base + i, first, len(descs) - first, mask))
    event_offset = GPU_HEADER_NBYTES
    desc_offset = event_offset + len(events) * GPU_EVENT_NBYTES
    total = desc_offset + len(descs) * GPU_DESC_NBYTES
    return (struct.pack('<11Q', GPU_BATCH_MAGIC, GPU_BATCH_VERSION, GPU_HEADER_NBYTES,
                        GPU_EVENT_NBYTES, GPU_DESC_NBYTES, len(events), len(descs),
                        3, event_offset, desc_offset, total)
            + b''.join(struct.pack('<5Q', *row) for row in events)
            + b''.join(struct.pack('<7Q', *row) for row in descs))


@pytest.fixture
def mixed_packet():
    return _mixed_packet
