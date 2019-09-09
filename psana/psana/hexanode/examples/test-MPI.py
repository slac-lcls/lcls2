#!/usr/bin/env python

print 'Use command: mpirun -n 8 python hexanode/examples/ex-09-MPI.py'

from time import time
import psana
from pyimgalgos.GlobalUtils import print_ndarr

ds = psana.DataSource('exp=xpptut15:run=390') # :smd ?
det = psana.Detector('AmoETOF.0:Acqiris.0')

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

t_sec = time()

### intensity = []

for nev,evt in enumerate(ds.events()):

        if nev%size!=rank: continue # different ranks look at different events

        if nev > 1000: break
	#print 'Rank: %d, ev: %d' % (rank, nev)
        res = det.raw(evt)
        if res is None : continue
        wf,wt = res
        #print_ndarr(wf,'wf')

        ### intensity.append(wf)

### allIntensities = comm.gather(intensity) # get intensities from all ranks

print "Rank %2d, ev: %4d consumed time (sec) = %.6f" % (rank, nev, time()-t_sec)

#smldata.close()

### if rank==0:
###     allIntensities = np.concatenate((allIntensities[:])) # put in one long list
###     print allIntensities 
### MPI.Finalize()
