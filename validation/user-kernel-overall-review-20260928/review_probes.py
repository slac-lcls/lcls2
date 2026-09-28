"""Review probes: real Python control flow; simulated CUDA/MPI completion."""
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS, MethodType

import numpy as np
from psana.detector.areadetector import AreaDetector
from psana.detector.shared_geo_cache import SharedGeoCache
from psana.gpu.gpu_events import GpuEventManager
from psana.gpu.gpu_stream import EventPool, _EventSlot
from psana.psexp.run import Run

ROOT = Path('/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline')


class Comm:
    isolated = False
    def bcast(self, value, root=0):
        if self.isolated:
            raise RuntimeError('BD entered a shared-memory collective after setup')
        return value
    def Barrier(self):
        pass


class SharedMemory:
    is_leader = True
    def __init__(self):
        self.shm_comm = Comm()
        self.arrays = {}
    def has_array(self, name):
        return name in self.arrays
    def get_array(self, name):
        return self.arrays[name]
    def allocate_array(self, name, shape, dtype, **kwargs):
        self.arrays[name] = np.empty(shape, dtype)
        return self.arrays[name]


def geometry_probe(source):
    tree = ast.parse(source)
    setup = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef)
                 and node.name == '_setup_jungfrau_shared_caches')
    # Extract the exact warmup variants in the changed startup method.
    variants = [dict((kw.arg, ast.literal_eval(kw.value)) for kw in node.keywords)
                for node in ast.walk(setup) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == '_pixel_coord_indexes']
    memory = SharedMemory()
    cache = SharedGeoCache(memory)
    geo = NS(get_pixel_coord_indexes=lambda **kwargs:
             (np.arange(8).reshape(2, 2, 2), np.arange(8).reshape(2, 2, 2)))
    detector = NS(_shared_geo_cache=cache, _calibconst={'geometry': ('fake', {})},
                  _path_geo_default=None, _segment_numbers=[0],
                  _det_name='jungfrau', _drp_class_name='raw', _det_geo=lambda: geo)
    detector._arr_for_daq_segments = MethodType(AreaDetector._arr_for_daq_segments, detector)
    method = MethodType(AreaDetector._pixel_coord_indexes, detector)
    for kwargs in variants:
        method(**kwargs)
    memory.shm_comm.isolated = True
    try:
        method(all_segs=True)
        outcome = 'cache hit; no collective'
    except RuntimeError as exc:
        outcome = str(exc)
    return variants, outcome


path = 'psana/psana/psexp/mpi_ds.py'
before = subprocess.check_output(['git', 'show', '480f7074c:' + path], cwd=ROOT, text=True)
print('GEOMETRY baseline:', geometry_probe(before))
print('GEOMETRY current: ', geometry_probe((ROOT/path).read_text()))


def manager_fixture():
    log = []
    pool = EventPool.__new__(EventPool)
    pool._n, pool._write_idx, pool._retiring = 2, 2, None
    pool._slots = [
        _EventSlot(i, {}, [], NS(synchronize=lambda: log.append('producer drained')), [], {})
        for i in range(2)
    ]
    manager = GpuEventManager.__new__(GpuEventManager)
    manager._done = manager._closed = False
    manager._iter = None
    manager.event_pool = pool
    manager.gpu_reader = None
    manager._next_batch = lambda: next(iter(()))
    manager._yield_ready = lambda record: iter([NS(dgrams=[], slot=record.slot_id)])
    manager._drain_pending_gpu_read = lambda: log.append('reads drained')
    manager._output_d2h = NS(close=lambda: log.append('output closed'))
    manager._task_constants = NS(close=lambda: log.append('constants closed'))
    return manager, log


manager, log = manager_fixture()
run = Run.__new__(Run)
run._evt_iter = manager
run._handle_transition = lambda dgrams: False
run._materialize_event = lambda envelope: envelope
events = run.events()
next(events)
events.close()
print('SERIAL public iterator close:', dict(closed=manager._closed,
      active_slots=manager.event_pool.active_count, actions=list(log)))
manager.close()
print('SERIAL explicit private manager close:', dict(closed=manager._closed,
      active_slots=manager.event_pool.active_count, actions=list(log)))

manager, log = manager_fixture()
events = manager._events()
next(events)
events.close()
print('SERIAL internal iterator close during final flush:', dict(closed=manager._closed,
      active_slots=manager.event_pool.active_count, actions=list(log)))
manager.close()
