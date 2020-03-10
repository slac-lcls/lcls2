#!/usr/bin/env python
#------------------------------

from time import time
import numpy as np
from psana.pyalgos.generic.NDArrUtils import print_ndarr
from psana.peakFinder.pypsalg import peaks_adaptive, peaks_droplet

#------------------------------

def test_pfv3r3(tname):

    print('test_pfv3r3: %s' % {'1':'2-d np.array', '2':'3-d np.array', '3':'list of 2-d np.array'}[tname])
    
    data = None
    mask = None

    sh = (1000,1000) if tname == '1' else (32,185,388)

    mu, sigma = 200, 25
    data = np.array(mu + sigma*np.random.standard_normal(sh), dtype=np.double)
    mask = np.ones(sh, dtype=np.uint16)

    if tname == '3' :
        data = [data[i,:,:] for i in range(sh[0])] 
        mask = [mask[i,:,:] for i in range(sh[0])] 

    #print 'data object', str(data)
    #print 'mask object', str(mask)
    #print 'data[0].shape', data[0].shape
    print_ndarr(data, 'input data')
    print_ndarr(mask, 'input mask')

    t0_sec = time()
    peaks = peaks_adaptive(data, mask, rank=5, r0=7.0, dr=2.0, nsigm=3,\
                           npix_min=1, npix_max=None, amax_thr=0, atot_thr=0, son_min=8)

    print('peaks_adaptive: img.shape=%s consumed time = %.6f(sec)' % (str(sh), time()-t0_sec))

    for p in peaks : 
        #print dir(p)
        print('  seg:%4d, row:%4d, col:%4d, npix:%4d, son:%4.1f' % (p.seg, p.row, p.col, p.npix, p.son))

#------------------------------

def test_pfv4r3(tname):

    print('test_pfv4r3: %s' % {'4':'2-d np.array', '5':'3-d np.array', '6':'list of 2-d np.array'}[tname])
    
    data = None
    mask = None

    sh = (1000,1000) if tname == '4' else (32,185,388) # (4,512,512)

    mu, sigma = 0, 25
    data = np.array(mu + sigma*np.random.standard_normal(sh), dtype=np.double)
    mask = np.ones(sh, dtype=np.uint16)

    if tname == '6' :
        data = [data[i,:,:] for i in range(sh[0])] 
        mask = [mask[i,:,:] for i in range(sh[0])] 

    #print 'data object', str(data)
    #print 'mask object', str(mask)
    #print 'data[0].shape', data[0].shape
    print_ndarr(data, 'input data')
    print_ndarr(mask, 'input mask')

    t0_sec = time()
    peaks = peaks_droplet(data, mask, thr_low=50, thr_high=80, rank=5, r0=7.0, dr=2.0,\
                          npix_min=1, npix_max=None, amax_thr=0, atot_thr=0, son_min=5.5)

    print('peaks_droplet: img.shape=%s consumed time = %.6f(sec)' % (str(sh), time()-t0_sec))

    for p in peaks : 
        #print dir(p)
        print('  seg:%4d, row:%4d, col:%4d, npix:%4d, son:%4.1f' % (p.seg, p.row, p.col, p.npix, p.son))

#------------------------------
#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print(50*'_', '\nTest %s:' % tname)
    if   tname in {'1','2','3'} : test_pfv3r3(tname) # v3r3: peaks_adaptive 2-d, 3-d, list of 2-d np.array
    elif tname in {'4','5','6'} : test_pfv4r3(tname) # v4r3: peaks_droplet  2-d, 3-d, list of 2-d np.array
    else : sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of test %s' % tname)

#------------------------------
