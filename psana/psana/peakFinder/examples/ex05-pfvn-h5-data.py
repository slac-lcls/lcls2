#!/usr/bin/env python

"""Test of peak-finders from psana/peakFinder/psalg_ext.pyx on hdf5 images
"""

#----------
import sys
import numpy as np

from psalg_ext import peak_finder_algos
from psana.pyalgos.generic.NDArrUtils import print_ndarr, reshape_to_2d
import psana.pyalgos.generic.Graphics as gr
from utils import plot_peaks_on_img, data_hdf5_v0

#----------

FNAME  = '/reg/g/psdm/tutorials/ami2/tmo/amox27716_run100.h5'
EVSKIP = 0
EVTMAX = 10 + EVSKIP

V3 = 3 # RANKER  v3r3
V4 = 4 # DROPLET v4r3

#----------

def test01(tname) :

    PF = V3 if tname == '3' else V4
    SHOW_PEAKS = tname != '0'

    ds = data_hdf5_v0(FNAME)
    #ds.print_images()

    alg = peak_finder_algos(pbits=0)
    if PF == V3 : alg.set_peak_selection_parameters(npix_min=0, npix_max=1e6, amax_thr=0, atot_thr=0, son_min=6)
    if PF == V4 : alg.set_peak_selection_parameters(npix_min=0, npix_max=1e6, amax_thr=0, atot_thr=0, son_min=6)

    img = ds.next_image()
    shape = img.shape
    mask = np.ones(shape, dtype=np.uint16)
    INDS = np.indices((shape[0],shape[1]), dtype=np.int64)
    imRow, imCol = INDS[0,:], INDS[1,:]  

    fig1, axim1, axcb1 = gr.fig_img_cbar_axes(gr.figure(figsize=(8,7)))

    for nev in range(min(EVTMAX, ds.nevmax)) :

        img = ds.next_image()

        #ave, rms = img.mean(), img.std()
        #amin, amax = ave-1*rms, ave+8*rms
        #amin, amax = img.min(), img.max()
        amin, amax = 0, img.max()
        axim1.clear()
        axcb1.clear()

        #imsh1,cbar1=\
        gr.imshow_cbar(fig1, axim1, axcb1, img, amin=amin, amax=amax, extent=None,\
                       interpolation='nearest', aspect='auto', origin='upper',\
                       orientation='vertical', cmap='inferno')
        fig1.canvas.set_window_title('Event: %04d random data'%nev)
        gr.move_fig(fig1, x0=400, y0=30)


        if SHOW_PEAKS :
            peaks = alg.peak_finder_v3r3_d2(img, mask, rank=5, r0=7, dr=2, nsigm=9)               if PF == V3 else\
                alg.peak_finder_v4r3_d2(img, mask, thr_low=100, thr_high=200, rank=5, r0=7, dr=2) if PF == V4 else\
                None
            plot_peaks_on_img(peaks, axim1, imRow, imCol, color='w', lw=1)

        gr.show(mode='do not hold') 

    gr.show()

#----------

USAGE =\
    'Usage: [python] %s <test-number>'\
    '\n  where <test-number> ='\
    '\n  0 - images'\
    '\n  3 - peak-finder V3 - ranker'\
    '\n  4 - peak-finder V4 - two-threshold'%(sys.argv[0])
 
if __name__ == "__main__" :
    tname = sys.argv[1] if len(sys.argv) > 1 else '4'
    print(50*'_', '\nTest %s:' % tname)
    if tname in ('0','3','4') : test01(tname)
    else : print(USAGE); sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of test %s' % tname)
 
#----------
