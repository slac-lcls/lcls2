"""Small native MPI/GPU import probe; no payload reads."""
import importlib,json,os,sys
from pathlib import Path
rank=int(os.environ['SLURM_PROCID'])
visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
os.environ['CUDA_VISIBLE_DEVICES']=visible[0] if rank==2 else ''
from mpi4py import MPI
import psana
from support import require,sha,write,HERE
require(MPI.COMM_WORLD.rank==rank and MPI.COMM_WORLD.size==3,'Rank identity')
reference=json.loads((HERE/'perlmutter-runtime.json').read_text())['modules']
actual={k:dict(path=importlib.import_module(k).__file__,sha256=sha(importlib.import_module(k).__file__)) for k in reference}
require(actual==reference,'Frozen runtime')
mpi=MPI.Get_library_version();require('CRAY MPICH' in mpi.upper(),'Cray MPI')
row=dict(rank=rank,mask=os.environ['CUDA_VISIBLE_DEVICES'],mpi=mpi,runtime_files=actual)
if rank==2:
    import cupy as cp,kvikio
    require(cp.cuda.runtime.getDeviceCount()==1,'One GPU')
    cp.cuda.Device().synchronize()
    row.update(cupy=cp.__version__,kvikio=kvikio.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),memory=cp.cuda.runtime.memGetInfo(),pci=cp.cuda.runtime.deviceGetPCIBusId(0))
rows=MPI.COMM_WORLD.gather(row,root=0)
if rank==0:write(sys.argv[1],dict(status='pass',job=os.environ['SLURM_JOB_ID'],rows=rows))
