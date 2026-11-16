"""Discard inherited workstation settings, preserving scheduler transport."""
import os,sys
from pathlib import Path
env={k:v for k,v in os.environ.items() if k in ('HOME','USER','LOGNAME','CUDA_VISIBLE_DEVICES') or k.startswith(('SLURM_','PMI_','PMIX_','PALS_','ALPS_'))}
env.update(PATH='/usr/bin:/bin',SHELL='/bin/bash',LANG='C.UTF-8',TZ='UTC',TERM='dumb')
os.execve('/bin/bash',['bash','--noprofile','--norc',str(Path(__file__).with_name('runtime.sh')),*sys.argv[1:]],env)
