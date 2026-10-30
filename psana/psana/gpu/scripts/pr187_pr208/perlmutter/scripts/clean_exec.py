"""Enter a fresh environment, preserving only identity and scheduler transport."""
import os
from pathlib import Path
import sys
assert len(sys.argv)>=4, 'clean_exec.py ROOT VARIANT|none COMMAND...'
root=Path(sys.argv[1]).resolve();variant=sys.argv[2]
assert variant in ('A','B','C','none')
env={k:v for k,v in os.environ.items() if k in ('HOME','USER','LOGNAME') or k.startswith(('SLURM_','PMI_','PMIX_','PALS_','ALPS_'))}
env.update(PATH='/usr/bin:/bin',SHELL='/bin/bash',LANG='C.UTF-8',TZ='UTC',TERM='dumb',HOSTNAME=os.uname().nodename)
if 'CUDA_VISIBLE_DEVICES' in os.environ:env['CUDA_VISIBLE_DEVICES']=os.environ['CUDA_VISIBLE_DEVICES']
os.execve('/bin/bash',['bash','--noprofile','--norc',str(root/'scripts/bootstrap.sh'),str(root),variant,*sys.argv[3:]],env)
