#!/usr/bin/env python
#--------------------

import sys
import numpy as np
from psana import DataSource, Detector 
from pypsalg import find_edges

#dsname = 'exp=xpptut15:run=280'
dsname = 'exp=amox27716:run=100'
detname = 'AmoEndstation.0:Acqiris.1'
#detname = 'AmoEndstation.0:Acqiris.2' # for xpptut15 ch:0 - MCP

ds = DataSource(dsname)
det = Detector(detname)

ch = int(sys.argv[1]) if len(sys.argv) > 1 else 2 # Acqiris chanel

for nevent,evt in enumerate(ds.events()):
    r = det.raw(evt)
    nda = np.array(r)
    print 'Event:%03d'%nevent, '\n' if r is not None else '', r
    if r is None : continue
    waveforms,times = r
    # find edges for channel 0
    # parameters: baseline, threshold, fraction, deadtime, leading_edges
    edges = find_edges(waveforms[ch],0.0,-0.05,1.0,5.0,True)
    # pairs of (amplitude,sampleNumber)
    print 'edges for channel %02d:'%ch
    for n,(v,bin) in enumerate(edges) :
        print '  edge:%02d  bin:%05d  value:%6.3f' %(n,bin,v)
    break

import matplotlib.pyplot as plt
plt.figure(figsize=(16,4))
#plt.plot(times[ch,:40000], waveforms[ch,:40000])
plt.plot(waveforms[ch,:40000])
plt.title('%s %s chanel %02d'%(dsname, detname, ch))
plt.xlabel('bin', fontsize=14)
plt.ylabel('intensity', fontsize=14)
plt.show()

print 'command to run: ./ex-quad-01-cfd.py <Acqiris chanel>'
