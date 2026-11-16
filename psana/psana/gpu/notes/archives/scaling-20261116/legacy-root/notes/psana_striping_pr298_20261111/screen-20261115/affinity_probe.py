"""Record the actual Slurm step mask without importing MPI, psana, or CUDA."""
import os,sys,platform
from pathlib import Path
from support import write
cpus=sorted(os.sched_getaffinity(0));topology={}
for cpu in cpus:
    p=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu)
    topology[str(cpu)]=[(p/'physical_package_id').read_text().strip(),(p/'core_id').read_text().strip()]
write(Path(sys.argv[1])/('affinity-%s-r%s.json'%(sys.argv[2],os.environ['SLURM_PROCID'])),{'rank':int(os.environ['SLURM_PROCID']),'host':platform.node(),'cpus':cpus,'topology':topology})
