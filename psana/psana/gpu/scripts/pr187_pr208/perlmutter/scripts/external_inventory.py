"""Read-only package and loaded-library fingerprints, outside benchmark timing."""
from pathlib import Path
import hashlib, importlib.metadata as md, json, os, subprocess, sys
import numpy, cupy, kvikio
from mpi4py import MPI
import psana, psana.dgram, psana.eventbuilder, psana.smdreader

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()

packages={n:md.version(n) for n in ('kvikio-cu12','libkvikio-cu12','cupy-cuda12x','numpy','mpi4py','meson','Cython')}
result=dict(python=sys.version,executable=sys.executable,packages=packages,
            imports={m.__name__:str(Path(m.__file__).resolve()) for m in (numpy,cupy,kvikio,MPI,psana)},
            mpi_library=MPI.Get_library_version(),host=os.uname().nodename)
if '--gpu' in sys.argv:
    result['cuda']=dict(driver=cupy.cuda.runtime.driverGetVersion(),runtime=cupy.cuda.runtime.runtimeGetVersion())
    cupy.cuda.runtime.free(0)
    result['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,pci.bus_id','--format=csv,noheader'],text=True)
prefix=Path(os.environ['BENCH_INSTALL']).resolve()
libs=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/' in line.split()[-1] and '.so' in line.split()[-1]})
result['project_libraries']={p:sha(p) for p in libs if Path(p).is_relative_to(prefix)}
result['external_libraries']={p:sha(p) for p in libs if not Path(p).is_relative_to(prefix)}
print(json.dumps(result,indent=2))
