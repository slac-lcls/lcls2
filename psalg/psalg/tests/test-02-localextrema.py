#!/usr/bin/env python
#------------------------------
import peakFinder as algos
import numpy as np
#------------------------------
#import psana.pyalgos.generic.NDArrGenerators as ag
#data = ag.random_standard(shape=sh, mu=200, sigma=25, dtype=np.float64)
#------------------------------

def test01(tname='1', NUMBER_OF_EVENTS=3, DO_PRINT=False) :

    print('local extrema : %s' % ('minimums' if tname in ('1','2')\
                             else 'maximums' if tname in ('3','4')\
                             else 'maximums runk=1 cross' if tname in ('5','6')\
                             else 'two-threshold maximums' if tname == '7'\
                             else 'unknown test'))

    from time import time
    from psana.pyalgos.generic.NDArrUtils import print_ndarr
    import psana.pyalgos.generic.Graphics as gg

    #sh, fs = (185,388), (11,5)
    sh, fs = (50,50), (7,7)
    fig1, axim1, axcb1 = gg.fig_img_cbar_axes(gg.figure(figsize=fs), (0.05, 0.05, 0.87, 0.90), (0.923, 0.05, 0.02, 0.90))
    fig2, axim2, axcb2 = gg.fig_img_cbar_axes(gg.figure(figsize=fs), (0.05, 0.05, 0.87, 0.90), (0.923, 0.05, 0.02, 0.90))

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

        if DO_PRINT : print_ndarr(data, 'input data')
        t0_sec = time()
        #----------
        if   tname in ('1','2') : nmax = algos.local_minimums(data, mask, rank, extrema)
        elif tname in ('3','4') : nmax = algos.local_maximums(data, mask, rank, extrema)
        elif tname in ('5','6') : nmax = algos.local_maximums_rank1_cross(data, mask, extrema)
        elif tname == '7'       : nmax = algos.threshold_maximums(data, mask, rank, thr_low, thr_high, extrema)
        else : contunue
        #----------
        print('Event: %4d,  consumed time = %10.6f(sec),  nmax = %d' % (evnum, time()-t0_sec, nmax))
        
        if DO_PRINT : print_ndarr(extrema, 'output extrema')
        
        img1 = data
        img2 = extrema

        axim1.clear()
        axim2.clear()
        
        ave, rms = img1.mean(), img1.std()
        amin, amax = ave-1*rms, ave+5*rms
        gg.imshow_cbar(fig1, axim1, axcb1, img1, amin=amin, amax=amax, cmap='inferno')
        axim1.set_title('Event: %d, Data'%evnum, color='k', fontsize=20)
        gg.move_fig(fig1, x0=550, y0=30)
        
        gg.imshow_cbar(fig2, axim2, axcb2, img2, amin=0, amax=5, cmap='inferno')
        axim2.set_title('Event: %d, Local extrema'%evnum, color='k', fontsize=20)
        gg.move_fig(fig2, x0=0, y0=30)
        
        gg.show(mode='DO_NOT_HOLD')
    gg.show()

#------------------------------

def test02() :
    rank=6
    algos.print_matrix_of_diag_indexes(rank)
    algos.print_vector_of_diag_indexes(rank)

#------------------------------

def usage() :
    msg = 'Usage: python psalgos/examples/ex-02-localextrema.py <test-number>'\
          '\n  where <test-number> ='\
          '\n  1 - local_minima_2d for random image'\
          '\n  2 - local_minima_2d for const image'\
          '\n  3 - local_maxima_2d for random image'\
          '\n  4 - local_maxima_2d for const image'\
          '\n  5 - local_maxima_rank1_cross_2d for random image'\
          '\n  6 - local_maxima_rank1_cross_2d for const image'\
          '\n  7 - threshold_maxima_2d for random image'\
          '\n  8 - print_matrix_of_diag_indexes, print_vector_of_diag_indexes'
    print(msg)

#------------------------------
#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s:' % tname)
    if   tname in ('1','2','3','4','5','6','7') : test01(tname)
    elif tname == '8' : test02()
    else : usage(); sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of test %s' % tname)

#------------------------------
