import os,gc
from types import SimpleNamespace
import cupy as cp
import kvikio
import numpy as np
from alignment_adapter import install
from psana.gpu.gpu_budget import _GpuBudget
from psana.gpu.gpu_kvikio_read import KvikioGpuReader as Reader
from pathlib import Path
shift=int(os.environ['BENCH_ALIGNMENT_SHIFT']);report=install(shift)
budget=_GpuBudget(8192)
reader=SimpleNamespace(cp=cp,_slot_bufs=[None],_budget=budget)
assert Reader.allocation_requirements(reader,512,0)==[(1024,0)]
Reader._ensure_slot_buffer(reader,0,512)
first=reader._slot_bufs[0];assert first.nbytes==512 and first.data.ptr%2==shift
assert budget.committed()==1024
assert Reader.allocation_requirements(reader,512,0)==[(0,0)]
Reader._ensure_slot_buffer(reader,0,256);assert reader._slot_bufs[0] is first
assert Reader.allocation_requirements(reader,1024,0)==[(1536,0)]
Reader._ensure_slot_buffer(reader,0,1024);assert budget.committed()==2560
first=None;gc.collect();assert budget.committed()==1536
path=Path(os.environ['TMPDIR'])/'adapter-input.bin';pattern=np.arange(1024,dtype=np.uint8);path.write_bytes(pattern.tobytes())
with kvikio.CuFile(str(path),'r') as f:
 assert f.pread(reader._slot_bufs[0],size=1024,file_offset=0).get()==1024
np.testing.assert_array_equal(reader._slot_bufs[0].get(),pattern)
reader._slot_bufs[0]=None;gc.collect();assert budget.committed()==0
path.unlink()
assert report['reader_allocations']==2
print('ADAPTER_CHECK_PASS',shift,report,flush=True)

# Query capability outside memcheck; libcuFile's probe itself reports an
# invalid cuMemFree API argument under sanitizer on this node.
import json,kvikio.cufile_driver
available=bool(kvikio.cufile_driver.get('is_gds_available'))
assert not available, 'This campaign requires the verified CPU-fallback setup'
print('GDS_PROBE '+json.dumps({'gds_available':available,'pci_bus':cp.cuda.Device().pci_bus_id}),flush=True)
