
"""copy of ~cpo/junk11.py on 2026-02-11
   mpirun -n 5 python ~cpo/junk11.py
   mpirun -n 5 python ./lcls2/psana/psana/detector/testman/ex-mpi-event-loop-control-cpo.py
"""

from psana import DataSource
import numpy as np
import os

def filter_callback(run):
    for i_step, step in enumerate(run.steps()):
        for i_evt, evt in enumerate(step.events()):
            if i_evt<3:
                print('= filter_callback yield for step/evt:', i_step, i_evt)
                yield evt

os.environ['PS_EB_NODES']='1'
os.environ['PS_SRV_NODES']='1'
ds = DataSource(exp='mfx100848724', run=49, smd_callback=filter_callback, batch_size=1)
myrun = next(ds.runs())
for nstep, step in enumerate(myrun.steps()):
    for nevt, evt in enumerate(step.events()):
        print('* step/evt:', nstep, nevt)
