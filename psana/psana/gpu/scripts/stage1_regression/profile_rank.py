"""Profile BD ranks only; SMD0/EB launch normally and never initialize CUDA."""
import os
from pathlib import Path
import sys

rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
if rank < 2:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
nsys = os.environ['BENCH_NSYS']
output = str(Path(os.environ['BENCH_TRACE']) / f'rank-{rank}')
os.execv(nsys, [nsys, 'profile', '--trace=cuda,nvtx', '--sample=none',
               '--cpuctxsw=none', '--force-overwrite=true',
               '--output='+output, sys.executable, *sys.argv[1:]])
