"""Actual step placement and granted hardware; no CUDA or MPI initialization."""
import json,os,socket,subprocess,sys
from pathlib import Path
from support import write,normalize_pci
cpus=sorted(os.sched_getaffinity(0));topology={}
for cpu in cpus:
    p=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu)
    topology[str(cpu)]=[(p/'physical_package_id').read_text().strip(),(p/'core_id').read_text().strip()]
raw=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
    '--query-gpu=uuid,pci.bus_id','--format=csv,noheader'],text=True)
gpus=[]
for line in raw.strip().splitlines():
    uuid,pci=map(str.strip,line.split(','));gpus.append(dict(uuid=uuid,pci=normalize_pci(pci)))
write(Path(sys.argv[1])/('probe-r%s.json'%os.environ['SLURM_PROCID']),
      dict(rank=int(os.environ['SLURM_PROCID']),host=socket.gethostname(),
           cpus=cpus,topology=topology,gpus=gpus))
