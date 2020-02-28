#!/usr/bin/env python
#----------
""" test for local_minima_1d and local_maxima_1d for 1-d data array folded in 2-d image
"""
#from psalg_ext import local_minima_1d, local_maxima_1d,\
#                      local_minimums, local_maximums,\
#                      local_maximums_rank1_cross, threshold_maximums
import numpy as np

from time import time
import psalg_ext as algos
from psana.pyalgos.generic.NDArrUtils import print_ndarr
import psana.pyalgos.generic.Graphics as gr


#----------

def test01(tname='1', NUMBER_OF_EVENTS=10, DO_PRINT=False) :

    print('local extrema : %s' % ('minimums' if tname in ('1','2')\
                             else 'maximums'))

    #sh, fs = (200,200), (11,10)
    sh, fs = (50,50), (7,6)
    #sh, fs = (185,388), (11,5)
    fig1, axim1, axcb1 = gr.fig_img_cbar_axes(gr.figure(figsize=fs))
    fig2, axim2, axcb2 = gr.fig_img_cbar_axes(gr.figure(figsize=fs))
    imsh1 = None
    imsh2 = None

    print('Image shape: %s' % str(sh))

    mu, sigma = 200, 25

    for evnum in range(NUMBER_OF_EVENTS) :

        data = 10*np.ones(sh, dtype=np.float64) if tname in ('2','4') else\
               np.array(mu + sigma*np.random.standard_normal(sh), dtype=np.float64)
        mask = np.ones(sh, dtype=np.uint16).flatten()
        #mask = np.random.binomial(2, 0.80, data.size).astype(dtype=np.uint16)
        extrema = np.zeros(sh, dtype=np.uint16).flatten()

        rank=5
        
        nmax = 0

        if DO_PRINT : print_ndarr(data, 'input data')
        t0_sec = time()

        #----------
        if   tname in ('1','2') : nmax = algos.local_minima_1d(data.flatten(), mask, rank, extrema)
        elif tname in ('3','4') : nmax = algos.local_maxima_1d(data.flatten(), mask, rank, extrema)
        #----------
        print('Event: %4d,  consumed time = %10.6f(sec),  nmax = %d' % (evnum, time()-t0_sec, nmax))

        extrema.shape = sh
        
        if DO_PRINT : print_ndarr(extrema, 'output extrema')
        
        img1 = data
        img2 = extrema

        axim1.clear()
        axcb1.clear()
        if imsh1 is not None : del imsh1
        imsh1 = None

        axim2.clear()
        axcb2.clear()
        if imsh2 is not None : del imsh2
        imsh2 = None
        
        ave, rms = img1.mean(), img1.std()
        amin, amax = ave-1*rms, ave+5*rms
        #imsh1,cbar1=\
        gr.imshow_cbar(fig1, axim1, axcb1, img1, amin=amin, amax=amax, extent=None,\
                       interpolation='nearest', aspect='auto', origin='upper',\
                       orientation='vertical', cmap='inferno')
        fig1.canvas.set_window_title('Event: %d Random data'%evnum)
        gr.move_fig(fig1, x0=560, y0=30)
        
        #imsh2,cbar2=\
        gr.imshow_cbar(fig2, axim2, axcb2, img2, amin=0, amax=5, extent=None,\
                       interpolation='nearest', aspect='auto', origin='upper',\
                       orientation='vertical', cmap='inferno')
        fig2.canvas.set_window_title('Event: %d Local extrema (1d-folded in image)'%evnum)
        gr.move_fig(fig2, x0=0, y0=30)
        
        gr.show(mode='DO_NOT_HOLD')
    gr.show()

#----------
def usage() :
    msg = 'Usage: python psalgos/examples/ex-07-localextrema-1d.py <test-number>'\
          '\n  where <test-number> ='\
          '\n  1 - local_minima_1d for random image'\
          '\n  2 - local_minima_1d for constant image'\
          '\n  3 - local_maxima_1d for random image'\
          '\n  4 - local_maxima_1d for constant image'
    print(msg)

#----------
#----------
#----------
#----------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s:' % tname)
    if tname in ('1','2','3','4') : test01(tname)
    else : usage(); sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of test %s' % tname)

#----------
