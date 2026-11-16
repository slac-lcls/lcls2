"""Small native ABI and publication-audit tests; never initializes CUDA."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / 'sim10_kvikio_bounce_20261113'))
os.environ['PS_PARALLEL'] = 'none'
from runtime import activate, identity
activate()
from reference import CpuCalibration, SeedReader, SHAPE, require, digest
from support import fixture
from stages import CopyAudit, StageTask, adapter
from contracts import admission_from_budget, memory_demand
from psana.gpu.gpu_budget import _GpuBudget
from psana.psexp.calib_xtc import CalibXtcConverter, load_calib_xtc_from_buffer

budget = _GpuBudget(6 << 30)
charge = budget.reserve_allocation(1024, requested=1000, category='host-interface-check')
charge.commit()
for stage in ('S', 'E', 'EC', 'ECC'):
    for batch in (50, 200):
        needed = sum(memory_demand(stage, batch).values()) - 1024
        ledger = admission_from_budget(stage, batch, needed, budget)
        require(ledger['framework_resident'] == 1024 and ledger['outstanding'] == needed,
                'Pinned native budget method/ledger interface')


class Manager:
    _gpu_budget = budget


manager = Manager()
pool = SimpleNamespace(used_bytes=lambda: 1024, total_bytes=lambda: 1024)
fake_cupy = SimpleNamespace(empty=np.empty, uint8=np.uint8, uint64=np.uint64, __version__='host-mock',
    cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (40 << 30, 40 << 30))),
    get_default_memory_pool=lambda: pool)
with patch.dict(sys.modules, cupy=fake_cupy):
    task = StageTask('S', 50, [])
    task.prepare(manager, {})
    require(task.initialized and task.ledger['framework_resident'] == 1024
            and task.weakref_probe['released'], 'StageTask.prepare pinned budget integration with host CUDA mock')
    require(len(task.memory) == 2, 'Setup memory observations')
    field = SimpleNamespace(segment_ids=tuple(range(32)),
                            rows=np.empty((1, 32, adapter.FIELD_WORDS), np.uint64))
    callback = SimpleNamespace(size=1, timestamps=[1], step_generation=0,
        field=lambda *args: field, segment_ids=lambda detector: tuple(range(32)))
    task(callback, SimpleNamespace(ptr=7))
    require([row['label'] for row in task.memory[2:]] == ['callback_entry', 'after_submission'],
            'Staging-only callback must have both in-loop memory samples')
charge.release()
require(budget.committed() == 0 and not budget.allocation_snapshot(), 'Budget charge release')

pedestal = np.arange(3 * 32 * 2 * 2, dtype=np.float64).reshape(3, 32, 2, 2) / 7
gain = np.full_like(pedestal, 3)
offset = np.full_like(pedestal, 0.1)
gain[:, 0, 0, 0] = 0
constants = {key: (value, {'ctype': key}) for key, value in
             [('pedestals', pedestal), ('pixel_gain', gain), ('pixel_offset', offset)]}
reference = CpuCalibration(constants)
raw = (np.arange(32 * 2 * 2, dtype=np.uint16).reshape(32, 2, 2) * 127) & 0x3fff
raw |= (np.indices(raw.shape)[0] % 4).astype(np.uint16) << 14
observed = reference(raw)
expected = np.zeros_like(observed)
for panel in range(32):
    encoded = panel % 4
    if encoded == 2:
        continue
    plane = 2 if encoded == 3 else encoded
    corrected = (raw[panel] & 0x3fff).astype(np.float32) - (pedestal[plane, panel] + offset[plane, panel]).astype(np.float32)
    factor = np.divide(np.ones_like(gain[plane, panel]), gain[plane, panel],
                       out=np.zeros_like(gain[plane, panel]), where=gain[plane, panel] != 0).astype(np.float32)
    expected[panel] = corrected * factor
require(np.array_equal(observed.view(np.uint32), expected.view(np.uint32)), 'CPU-v3 native bitwise reference')
converter = CalibXtcConverter({'jungfrau': 'test-jungfrau-frozen'})
buffer, config_bytes, data_bytes = converter.convert_to_buffer({'jungfrau': constants})
restored, owner = load_calib_xtc_from_buffer(buffer)
for key, (array, metadata) in constants.items():
    require(np.array_equal(restored['jungfrau'][key][0], array), 'Calibration XTC roundtrip')


class Copier:
    pinned_bytes = 4096
    def _destination(self, shape, dtype, nbytes):
        return np.empty(shape, dtype), object()
    def enqueue(self, record):
        for item in record.publication_batches:
            self._destination(item.shape, item.dtype, item.nbytes)


copier = Copier()
audit = CopyAudit(copier)
publication = SimpleNamespace(name='sim10.errors', dtype=np.dtype('uint32'), shape=(200, 32), nbytes=25600)
copier.enqueue(SimpleNamespace(publication_batches=[publication]))
require(audit.bytes == 25600 and audit.groups == 1 and audit.fallbacks == 0, 'Copy decision accounting')
rejected = []
for bad in (SimpleNamespace(name='image', dtype=np.dtype('uint32'), shape=(200, 32), nbytes=25600),
            SimpleNamespace(name='sim10.errors', dtype=np.dtype('uint32'), shape=(1000, 32), nbytes=128000)):
    try:
        copier.enqueue(SimpleNamespace(publication_batches=[bad]))
    except ValueError:
        rejected.append(True)
require(len(rejected) == 2, 'Image publication and oversized metadata must be rejected')
fallback = Copier()
fallback._destination = lambda shape, dtype, nbytes: (np.empty(shape, dtype), None)
fallback_audit = CopyAudit(fallback)
try:
    fallback.enqueue(SimpleNamespace(publication_batches=[publication]))
except ValueError:
    require(fallback_audit.fallbacks == 1, 'Fallback must be observed before failure')
else:
    raise RuntimeError('Ordinary-host fallback was not rejected')
manifest, streams, timestamps = fixture()
reader = SeedReader([stream['files']['bd']['path'] for stream in streams], timestamps[:3])
try:
    first = reader.raw(timestamps[0])
    second = reader.raw(timestamps[1])
    require(first.shape == second.shape == SHAPE and first.dtype == second.dtype == np.uint16,
            'Native CPU seed read/physical panel layout')
    require(digest(first) == digest(reader.raw(timestamps[0])), 'Native indexed read must be repeatable')
finally:
    reader.close()
print(json.dumps(dict(status='pass', checks=['native_cpu_v3_bitwise', 'calib_xtc_roundtrip',
    'copy_byte_accounting', 'image_and_cap_rejection', 'ordinary_host_fallback_rejection',
    'native_indexed_five_stream_seed_reader', 'pinned_budget_admission_methods',
    'stage_prepare_real_budget_host_cuda_mock'],
    config_bytes=config_bytes, data_bytes=data_bytes,
    runtime=identity(('psana.dgram', 'psana.utilsdetector_ext', 'psana.pycalgos.utilsdetector', 'psana.psexp.calib_xtc')),
    limitation='Host ABI smoke checks, not CUDA/MPI or full-shape acceptance'), indent=2))
