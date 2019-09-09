#!/usr/bin/env python
#--------------------

import numpy as np
from psana import DataSource, Detector 

ds = DataSource('exp=amox27716:run=100')
det = Detector('AmoEndstation.0:Acqiris.1')

for n,evt in enumerate(ds.events()):
    v = det.raw(evt)
    print 'Event:%03d'%n, '\n' if v is not None else '', v
    if v is not None : break

nda = np.array(v)
print 'raw.shape:', nda.shape

#nbins = nda.shape[-1]; img.shape = (nda.size/nbins, nbins)

import matplotlib.pyplot as plt
style = ('b-', 'r-', 'g-', 'k-', 'm-', 'y-', 'c-', )
plt.figure(figsize=(16,4))
for ch in range(2,7) :
    w = nda[0,ch,5000:30000]
    print 'ch:%02d wave.shape: %s'%(ch,str(w.shape))
    plt.plot(w, style[ch])
    #t = nda[1,ch,5000:30000]
    #plt.plot(t, w, style[ch])
    plt.xlabel('bin', fontsize=14)
    plt.ylabel('intensity', fontsize=14)
plt.show()
