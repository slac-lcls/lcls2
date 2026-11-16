"""Launch a separate per-rank Nsight capture after clean-environment setup."""
import os
from pathlib import Path
import sys

rank = int(os.environ['SLURM_PROCID'])
os.environ['CUDA_VISIBLE_DEVICES'] = '0' if rank >= 2 else ''
nsys, destination, *command = sys.argv[1:]
args = [nsys, 'profile', '--trace=cuda,nvtx,osrt,mpi', '--mpi-impl=mpich',
        '--sample=none', '--cpuctxsw=none', '--cuda-memory-usage=true',
        '--force-overwrite=false', '--output='+str(Path(destination)/f'rank-{rank}'), *command]
os.execv(nsys, args)
