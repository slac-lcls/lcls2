"""Profile only the GPU/BD rank; SMD0 and EB execute normally."""
import argparse
import os

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--nsys', required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command[0] == '--' else a.command
if int(os.environ['OMPI_COMM_WORLD_RANK']) == 2:
    command = [a.nsys, 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
               '--cuda-event-trace=false', '--capture-range=cudaProfilerApi',
               '--capture-range-end=stop', '--export=sqlite', '--output=' + a.output] + command
os.execvp(command[0], command)
