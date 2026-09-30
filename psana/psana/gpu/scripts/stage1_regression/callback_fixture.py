"""Synthetic full-frame Jungfrau-shaped input, without file I/O or calibration."""
import struct
from types import SimpleNamespace as NS

import numpy as np


def jungfrau_fixture(cp, budget, n_events):
    from psana.gpu.gpu_detector import DenseInputPreparer
    from psana.gpu.gpu_input import GpuDetectorBinding, GpuEventDgrams
    from psana.gpu.gpudgram.batch import GpuXtcBatchPool
    from psana.gpu.gpudgram.config import GpuStreamConfigTable
    from psana.gpu.gpu_kvikio_read import (DESC_NCOLS, DESC_EVENT_INDEX,
        DESC_STREAM_ID, DESC_DEVICE_OFFSET, DESC_READ_SIZE, DESC_TIMESTAMP)
    from test_batched_gather import _xtc
    segments = tuple(range(32))
    entries = [dict(det_name='camera', det_type='test', det_id='camera', segment=s,
                   alg_name='raw', alg_version=(1,0,0), names_id_value=10+s,
                   fields=[dict(name='pixels',type=1,element_size=2,rank=2,
                                field_index=0,shape_index=0)]) for s in segments]
    configs = GpuStreamConfigTable({0:entries})
    binding = GpuDetectorBinding('camera', canonical_segment_ids=segments,
        field_handles_by_segment={s:configs.resolve('camera',s,'raw','pixels') for s in segments})
    parser = GpuXtcBatchPool(configs, field_handles=configs.field_handles(), n_slots=1,budget=budget)
    initial = DenseInputPreparer((32,512,1024),binding,n_slots=1,budget=budget)
    initial.configure_gather(parser.handle_indices)
    expected = np.empty((n_events,32,512,1024),np.uint16)
    descriptors = np.zeros((n_events,DESC_NCOLS),np.uint64)
    pieces, specs, offset = [], [], 0
    base = np.arange(512*1024,dtype=np.uint16).reshape(512,1024)
    shape = _xtc(2, struct.pack('<5I',512,1024,0,0,0))
    for i in range(n_events):
        nodes=[]
        for s in segments:
            expected[i,s] = base + np.uint16(i*400+s)
            nodes.append(_xtc(1,shape+_xtc(3,expected[i,s].tobytes()),10+s))
        payload=struct.pack('<QI',100+i,12<<24)+_xtc(0,b''.join(nodes))
        descriptors[i,[DESC_EVENT_INDEX,DESC_STREAM_ID,DESC_DEVICE_OFFSET,DESC_READ_SIZE,DESC_TIMESTAMP]]=(7+3*i,0,offset,len(payload),100+i)
        specs.append(NS(first_desc=i,n_desc=1,timestamp=100+i,batch_event_index=7+3*i))
        pieces.append(payload);offset+=len(payload)
    host=np.frombuffer(b''.join(pieces),np.uint8)
    del pieces, nodes, payload
    producer=cp.cuda.Stream(non_blocking=True)
    with producer: data=cp.asarray(host)
    producer.synchronize()
    batch=parser.parse(0,data,descriptors,producer)
    batch._test_descriptors=descriptors
    producer.synchronize()
    events=tuple(GpuEventDgrams(e,batch) for e in specs)
    return parser,initial,np.uint16,batch,events,expected
