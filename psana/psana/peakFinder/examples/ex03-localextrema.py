#!/usr/bin/env python
""" test of psalg_ext.local_minimums, local_maximums, threshold_maximums, local_maximums_rank1_cross
"""
#----------
import sys
import psalg_ext as algos
import numpy as np
#----------

def test01(tname='1', NUMBER_OF_EVENTS=5, DO_PRINT=True) :

    print('local extrema : %s' % ('minimums' if tname in ('1','2')\
                             else 'maximums' if tname in ('3','4')\
                             else 'maximums runk=1 cross' if tname in ('5','6')\
                             else 'two-threshold maximums' if tname == '7'\
                             else 'unknown test'))

    from time import time #, sleep
    from psana.pyalgos.generic.NDArrUtils import print_ndarr
    import psana.pyalgos.generic.Graphics as gr

    sh, fs = (50,50), (7,6)
    fig1, axim1, axcb1 = gr.fig_img_cbar_axes(gr.figure(figsize=fs))
    fig2, axim2, axcb2 = gr.fig_img_cbar_axes(gr.figure(figsize=fs))
    imsh1 = None
    imsh2 = None

    print('Image shape: %s' % str(sh))

    mu, sigma = 200, 25

    for evnum in range(NUMBER_OF_EVENTS) :

        data = 10.*np.ones(sh, dtype=np.float64) if tname in ('2','4','6') else\
               np.array(mu + sigma*np.random.standard_normal(sh), dtype=np.float64)
        mask = np.ones(sh, dtype=np.uint16)
        extrema = np.zeros(sh, dtype=np.uint16)
        rank=5
        
        thr_low = mu+3*sigma
        thr_high = mu+4*sigma

        nmax = 0

        if DO_PRINT : print_ndarr(data, '        input data')
        t0_sec = time()
        #----------
        if   tname in ('1','2') : nmax = algos.local_minimums(data, mask, rank, extrema)
        elif tname in ('3','4') : nmax = algos.local_maximums(data, mask, rank, extrema)
        elif tname in ('5','6') : nmax = algos.local_maximums_rank1_cross(data, mask, extrema)
        elif tname == '7'       : nmax = algos.threshold_maximums(data, mask, rank, thr_low, thr_high, extrema)
        elif tname == '8'       : nmax = algos.local_maximums_rank1_cross(data, mask, extrema)
        else : contunue
        #----------
        print('Event: %2d,  consumed time = %10.6f(sec),  nmax = %d' % (evnum, time()-t0_sec, nmax))

        if DO_PRINT : print_ndarr(extrema, '        output extrema')
        
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
        
        #ave, rms = img1.mean(), img1.std()
        #amin, amax = ave-1*rms, ave+5*rms
        amin, amax = img1.min(), img1.max()
        #imsh1,cbar1=\
        gr.imshow_cbar(fig1, axim1, axcb1, img1, amin=amin, amax=amax, extent=None,\
                       interpolation='nearest', aspect='auto', origin='upper',\
                       orientation='vertical', cmap='inferno')
        fig1.canvas.set_window_title('Event: %d Random data'%evnum)
        gr.move_fig(fig1, x0=560, y0=30)
        
        #imsh2,cbar2=\
        gr.imshow_cbar(fig2, axim2, axcb2, img2, amin=0, amax=img2.max(), extent=None,\
                       interpolation='nearest', aspect='auto', origin='upper',\
                       orientation='vertical', cmap='inferno')
        fig2.canvas.set_window_title('Event: %d Local extrema'%evnum)
        gr.move_fig(fig2, x0=0, y0=30)
        
        gr.show(mode='DO_NOT_HOLD')

    gr.show()

#----------

def test02(rank=6) :
    algos.print_matrix_of_diag_indexes(rank)
    algos.print_vector_of_diag_indexes(rank)

#----------

def usage() :
    msg = 'Usage: python examples/ex02-localextrema.py <test-number>'\
          '\n  where <test-number> ='\
          '\n  1 - local_minimums for random image'\
          '\n  2 - local_minimums for const image'\
          '\n  3 - local_maximums for random image'\
          '\n  4 - local_maximums for const image'\
          '\n  5 - local_maxima_rank1_cross for random image'\
          '\n  6 - local_maxima_rank1_cross for const image'\
          '\n  7 - threshold_maximums for random image'\
          '\n  8 - local_maximums_rank1_cross for random image'\
          '\n  9 - print_matrix_of_diag_indexes, print_vector_of_diag_indexes'
    print(msg)

#----------
#----------
#----------
#----------

if __name__ == "__main__" :
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s:' % tname)
    if   tname in ('1','2','3','4','5','6','7','8') : test01(tname)
    elif tname == '9' : test02()
    else : usage(); sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of test %s' % tname)

#----------
