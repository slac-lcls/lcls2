
"""copy of ~cpo/junk11.py on 2026-02-11
   mpirun -n 5 python ./ex-mpi-event-loop-times.py
"""
from time import time
from psana import DataSource
import numpy as np
import os

import psutil
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
cpu_num = psutil.Process().cpu_num()
s_rsc = 'rank:%03d/%03d-cpu:%03d' % (rank, size, cpu_num)

def filter_callback(run):
    t0_sec_run = time()
    for i_step, step in enumerate(run.steps()):
        print('%s i_step %d' % (s_rsc, i_step))
        t0_sec_step = time()
        for i_evt, evt in enumerate(step.events()):
            yield evt
        print('%s i_step %d end of %d event loop dt, sec = %.3f' % (s_rsc, i_step, i_evt, time()-t0_sec_step))
    print('%s filter_callback(run) t, sec = %.3f' % (s_rsc, time()-t0_sec_run))

print('%s start' % s_rsc)
t0_sec = time()

os.environ['PS_EB_NODES']='1'
os.environ['PS_SRV_NODES']='1'
ds = DataSource(exp='mfx100848724', run=49, smd_callback=filter_callback, batch_size=1)
#ds = DataSource(exp='mfx100848724', run=49, batch_size=1)
myrun = next(ds.runs())
print('%s DataSource startup time, sec = %.3f' % (s_rsc, time()-t0_sec))

t0_sec = time()
for nstep,step in enumerate(myrun.steps()):
    for nevt,evt in enumerate(step.events()):
        if nevt%100 == 0: print('* %s step:%d evt:%d' % (s_rsc, nstep, nevt))
print('%s run t, sec = %.3f' % (s_rsc, time()-t0_sec))

# EOF
