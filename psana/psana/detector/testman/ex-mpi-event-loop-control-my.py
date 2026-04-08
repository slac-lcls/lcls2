
"""copy of ~cpo/junk11.py on 2026-02-11
   mpirun -n 5 python ./lcls2/psana/psana/detector/testman/ex-mpi-event-loop-control-my.py
   mpirun -n 5 python ./ex-mpi-event-loop-control-my.py
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

class SingletPars:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        #print('%s SingletPars: %s' % (s_rsc, self.info_pars()))

    def get(self, pname, vdef):
        return self.kwargs.get(pname, vdef)

    def info_pars(self):
        return str(self.kwargs)

cbpars = SingletPars(**{'stepnum':2, 'stepevts':3})

def filter_callback(run):
    t0_sec_run = time()
    for nstep, step in enumerate(run.steps()):
        t0_sec_step = time()
        print('%s filter_callback nstep %d' % (s_rsc, nstep))
        for nevt, evt in enumerate(step.events()):
            if nevt%100==0:
                print('filter_callback yield', nstep, nevt, s_rsc)
            if nevt<600:
                yield evt
        print('%s filter_callback end of step %d end of %d event loop dt, sec = %.6f' % (s_rsc, nstep, nevt, time()-t0_sec_step))
    print('%s end of filter_callback t, sec = %.6f' % (s_rsc, time()-t0_sec_run))


t0_sec = time()
print('%s start' % s_rsc)

os.environ['PS_EB_NODES']='1'
os.environ['PS_SRV_NODES']='1'
ds = DataSource(exp='mfx100848724', run=49, max_events=3000, smd_callback=filter_callback, batch_size=1)
myrun = next(ds.runs())
print('%s DataSource startup time, sec = %.6f' % (s_rsc, time()-t0_sec))

t0_sec = time()
nstep = None
for nstep,step in enumerate(myrun.steps()):
    for nevt,evt in enumerate(step.events()):
        if nevt<10 or nevt%100 == 0: print('* %s step:%d evt:%d' % (s_rsc, nstep, nevt))
print('%s end of script t, sec = %.6f' % (s_rsc, time()-t0_sec))

# EOF
