"""Six-rank clean runtime/device probe; no timed event workload."""
import os,json
rank=int(os.environ['SLURM_PROCID'])
os.environ['CUDA_VISIBLE_DEVICES']='0' if rank>=2 else ''
os.environ['SLURM_GPUS_ON_NODE']='1'
expected=set(range(24))|set(range(64,88))
os.sched_setaffinity(0,expected);assert set(os.sched_getaffinity(0))==expected
from mpi4py import MPI
from runtime_audit import capture
assert MPI.COMM_WORLD.size==6 and MPI.COMM_WORLD.rank==rank
if rank>=2:
    import cupy as cp,kvikio,kvikio.defaults
    assert cp.cuda.runtime.getDeviceCount()==1
    assert cp.__version__=='14.0.1' and kvikio.__version__=='26.10.00'
    assert kvikio.defaults.is_compat_mode_preferred()
    assert float(cp.arange(16).sum().get())==120
    name=cp.cuda.runtime.getDeviceProperties(0)['name'];bus=cp.cuda.runtime.deviceGetPCIBusId(0)
    device=dict(name=name.decode() if isinstance(name,bytes) else name,bus=bus.decode() if isinstance(bus,bytes) else bus,bytes=cp.cuda.Device().mem_info[1])
else:device=None
row=dict(rank=rank,affinity=sorted(os.sched_getaffinity(0)),device=device,provenance=capture(),mpi=MPI.Get_library_version())
rows=MPI.COMM_WORLD.gather(row,root=0)
if rank==0:
    assert all(set(x['affinity'])==expected for x in rows)
    print('GPU_SWITCH_PROBE '+json.dumps(rows),flush=True)
