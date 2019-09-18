#!/usr/bin/env python

print 'Use command: mpirun -n 8 python hexanode/examples/ex-09-MPIDataSource.py'

from time import time
import psana
from pyimgalgos.GlobalUtils import print_ndarr

#ds = psana.DataSource('exp=xpptut15:run=390') # :smd ?
#det = psana.Detector('AmoETOF.0:Acqiris.0')



dsource = psana.MPIDataSource('exp=xpptut15:run=54:smd')
cspaddet = psana.Detector('cspad')
smldata = dsource.small_data('run54.h5',gather_interval=100)
 
partial_run_sum = None
for nevt,evt in enumerate(dsource.events()):
   calib = cspaddet.calib(evt)
   if calib is None: continue
   cspad_sum = calib.sum()      # number
   cspad_roi = calib[0][0][3:5] # array
   if partial_run_sum is None:
      partial_run_sum = cspad_roi
   else:
      partial_run_sum += cspad_roi
 
   # save per-event data
   smldata.event(cspad_sum=cspad_sum,cspad_roi=cspad_roi)
 
   if nevt>3: break
 
# get "summary" data
run_sum = smldata.sum(partial_run_sum)
# save HDF5 file, including summary data
smldata.save(run_sum=run_sum)
