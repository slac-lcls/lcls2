#!/usr/bin/env python
"""
Test of the psana/peakFinder/psalg_ext.pyx
"""
#----------
import sys
from time import time
import numpy as np

from psalg_ext import peak_finder_algos

from psana.pyalgos.generic.NDArrGenerators import random_standard, add_random_peaks, add_ring
from psana.pyalgos.generic.NDArrUtils import print_ndarr, reshape_to_2d
import psana.pyalgos.generic.Graphics as gr

from utils_peak_graphics import plot_peaks_on_img

#----------

def plot_image(img, img_range=None, amp_range=None, figsize=(12,10)) : 
    axim = gr.plotImageLarge(img, img_range, amp_range, figsize)
    gr.show() 

#----------

def image_with_random_peaks(shape=(500, 500)) : 
    img = random_standard(shape, mu=0, sigma=10)
    peaks = add_random_peaks(img, npeaks=10, amean=100, arms=50, wmean=1.5, wrms=0.3)
    return img, peaks

#----------

hdr = 'Evnum  Reg  Seg  Row  Col  Npix      Amax      Atot   rcent   ccent '+\
      'rsigma  csigma rmin rmax cmin cmax    bkgd     rms     son' # +\
      #'  imrow   imcol     x[um]     y[um]     r[um]  phi[deg]'

fmt = '%5d  %3s  %3d %4d %4d  %4d  %8.1f  %8.1f  %6.1f  %6.1f %6.2f  %6.2f'+\
      ' %4d %4d %4d %4d  %6.2f  %6.2f  %6.2f' # +\
      #' %6d  %6d  %8.0f  %8.0f  %8.0f  %8.2f'

#----------
V1 = 1 # TBD
V2 = 2 # TBD
V3 = 3 # RANKER  v3r3
V4 = 4 # DROPLET v4r3
#----------

def test_pf(tname) : 

    ##-----------------------------

    PF = V4 # default
    if tname == '1' : PF = V1
    if tname == '2' : PF = V2
    if tname == '3' : PF = V3
    if tname == '4' : PF = V4

    SKIP   = 0
    EVTMAX = 10 + SKIP

    DO_PLOT_IMAGE           = True
    DO_PLOT_PIXEL_STATUS    = False  #True if PF in (V2,V4) else False
    DO_PLOT_CONNECED_PIXELS = False  #True if PF in (V2,V3,V4) else False
    DO_PLOT_LOCAL_MAXIMUMS  = False  #True if PF == V3 else False
    DO_PLOT_LOCAL_MINIMUMS  = False  #True if PF == V3 else False

    shape=(1000, 1000)

    mask = np.ones(shape, dtype=np.uint16)

    # Pixel image indexes
    #arr3d = np.array((1,shape[0],shape[1]))

    INDS = np.indices((1,shape[0],shape[1]), dtype=np.int64)
    imRow, imCol = INDS[1,:], INDS[2,:]  
    #iX  = np.array(det.indexes_x(evt), dtype=np.int64) #- xoffset
    #iY  = np.array(det.indexes_y(evt), dtype=np.int64) #- yoffset

    ##-----------------------------
    fs = (8,7) # (11,10)
    fig1, axim1, axcb1 = gr.fig_img_cbar_axes(gr.figure(figsize=fs)) if DO_PLOT_IMAGE           else (None, None, None)
    fig2, axim2, axcb2 = gr.fig_img_cbar_axes(gr.figure(figsize=fs)) if DO_PLOT_PIXEL_STATUS    else (None, None, None)
    fig3, axim3, axcb3 = gr.fig_img_cbar_axes(gr.figure(figsize=fs)) if DO_PLOT_CONNECED_PIXELS else (None, None, None)
    fig4, axim4, axcb4 = gr.fig_img_cbar_axes(gr.figure(figsize=fs)) if DO_PLOT_LOCAL_MAXIMUMS  else (None, None, None)
    fig5, axim5, axcb5 = gr.fig_img_cbar_axes(gr.figure(figsize=fs)) if DO_PLOT_LOCAL_MINIMUMS  else (None, None, None)
    imsh1 = None
    imsh2 = None
    imsh3 = None
    imsh4 = None
    imsh5 = None
    ##-----------------------------

    alg = peak_finder_algos(pbits=0)

    if   PF == V1 :
      alg.set_peak_selection_parameters(npix_min=0, npix_max=1e6, amax_thr=0, atot_thr=0, son_min=6)

    elif PF == V2 :
      alg.set_peak_selection_parameters(npix_min=0, npix_max=1e6, amax_thr=0, atot_thr=0, son_min=6)

    elif PF == V3 :
      alg.set_peak_selection_parameters(npix_min=0, npix_max=1e6, amax_thr=0, atot_thr=0, son_min=8)

    elif PF == V4 :
      alg.set_peak_selection_parameters(npix_min=0, npix_max=1e6, amax_thr=0, atot_thr=0, son_min=6)

    #alg.print_attributes()

    for ev in range(EVTMAX) :
        ev1 = ev + 1

        if ev<SKIP : continue
        #if ev>=EVTMAX : break

        print(50*'_', '\nEvent %04d' % ev1)

        img, peaks_sim = image_with_random_peaks(shape)

        # --- for debugging
        #np.save('xxx-image', img) 
        #np.save('xxx-peaks', np.array(peaks_sim)) 

        #img = np.load('xxx-image-crash.npy') 
        #peaks_sim = np.load('xxx-peaks-crash.npy') 
        # ---
        
        peaks_gen = [(0, r, c, a, a*s, 9*s*s) for r,c,a,s in peaks_sim]

        t0_sec = time()

        peaks = alg.peak_finder_v3r3_d2(img, mask, rank=5, r0=7, dr=2, nsigm=3)                 if PF == V3 else\
                alg.peak_finder_v4r3_d2(img, mask, thr_low=20, thr_high=40, rank=6, r0=7, dr=2) if PF == V4 else\
                None
                #alg.peak_finder_v3r3_d2(img, rank=5, r0=7, dr=2, nsigm=3)        if PF == V3 else\
                #alg.peak_finder_v4r3_d2(img, thr_low=20, thr_high=40, rank=6, r0=7, dr=2) if PF == V3 else\

        print('  Time consumed by the peak_finder = %10.6f(sec)' % (time()-t0_sec))

        map2 = reshape_to_2d(alg.maps_of_pixel_status())     if DO_PLOT_PIXEL_STATUS    else None # np.zeros((10,10))
        map3 = reshape_to_2d(alg.maps_of_connected_pixels()) if DO_PLOT_CONNECED_PIXELS else None # np.zeros((10,10))
        map4 = reshape_to_2d(alg.maps_of_local_maximums())   if DO_PLOT_LOCAL_MAXIMUMS  else None # np.zeros((10,10))
        map5 = reshape_to_2d(alg.maps_of_local_minimums())   if DO_PLOT_LOCAL_MINIMUMS  else None # np.zeros((10,10))

        print('arrays are extracted')

        #print_arr(map2, 'map_of_pixel_status')
        #print_arr(map3, 'map_of_connected_pixels')
        #maps.shape = shape 


        print('Simulated peaks:')
        for i, (r0, c0, a0, sigma) in enumerate(peaks_sim) :
            print('  %04d  row=%6.1f  col=%6.1f  amp=%6.1f  sigma=%6.3f' % (i, r0, c0, a0, sigma))
        #plot_image(img)

        print('Found peaks:')
        print(hdr)
        reg = 'IMG'
        #for pk in peaks :
        #    seg,row,col,npix,amax,atot,rcent,ccent,rsigma,csigma,\
        #    rmin,rmax,cmin,cmax,bkgd,rms,son = pk[0:17]
        #    rec = fmt % (ev, reg, seg, row, col, npix, amax, atot, rcent, ccent, rsigma, csigma,\
        #          rmin, rmax, cmin, cmax, bkgd, rms, son) #,\
        #          #imrow, imcol, xum, yum, rum, phi)
        #    print(rec)

        for p in peaks :
            #print('  algos:', p.parameters())
            print('  row:%4d, col:%4d, npix:%4d, son:%4.1f amp_tot:%4.1f' % (p.row, p.col, p.npix, p.son, p.amp_tot))

        if DO_PLOT_PIXEL_STATUS :
            gr.plot_imgcb(fig2, axim2, axcb2, imsh2, map2, amin=0, amax=30, title='Pixel status, ev: %04d' % ev1)
            gr.move_fig(fig2, x0=0, y0=30)


        if DO_PLOT_CONNECED_PIXELS :
            cmin, cmax = (map3.min(), map3.max()) if map3 is not None else (None,None)
            print('Connected pixel groups min/max:', cmin, cmax)
            gr.plot_imgcb(fig3, axim3, axcb3, imsh3, map3, amin=cmin, amax=cmax, title='Connected pixel groups, ev: %04d' % ev1)
            gr.move_fig(fig3, x0=100, y0=30)


        if DO_PLOT_LOCAL_MAXIMUMS :
            gr.plot_imgcb(fig4, axim4, axcb4, imsh4, map4, amin=0, amax=10, title='Local maximums, ev: %04d' % ev1)
            gr.move_fig(fig4, x0=200, y0=30)


        if DO_PLOT_LOCAL_MINIMUMS :
            gr.plot_imgcb(fig5, axim5, axcb5, imsh5, map5, amin=0, amax=10, title='Local minimums, ev: %04d' % ev1)
            gr.move_fig(fig5, x0=300, y0=30)


        if DO_PLOT_IMAGE:
            #nda = maps_of_conpix_arc        
            #nda = maps_of_conpix_equ        
            #img = det.image(evt, nda)[xoffset:xoffset+xsize,yoffset:yoffset+ysize]
            #img = det.image(evt, mask_img*nda)[xoffset:xoffset+xsize,yoffset:yoffset+ysize]
            #img = det.image(evt, maps_of_conpix_equ)[xoffset:xoffset+xsize,yoffset:yoffset+ysize]

            ave, rms = img.mean(), img.std()
            amin, amax = ave-1*rms, ave+8*rms
            axim1.clear()
            if imsh1 is not None : del imsh1
            imsh1 = None
            gr.imshow_cbar(fig1, axim1, axcb1, img, amin=amin, amax=amax, extent=None,\
                           interpolation='nearest', aspect='auto', origin='upper',\
                           orientation='vertical', cmap='inferno')
            #gr.plot_imgcb(fig1, axim1, axcb1, imsh1, img, amin=amin, amax=amax, title='Image, ev: %04d' % ev1)
            fig1.canvas.set_window_title('Event: %04d random data'%ev1)
            gr.move_fig(fig1, x0=400, y0=30)

            #plot_peaks_on_img(peaks_gen, axim1, imRow, imCol, color='g', lw=5)
            plot_peaks_on_img(peaks, axim1, imRow, imCol, color='w', lw=1)

            fig1.canvas.draw() # re-draw figure content

            #gr.plotHistogram(nda, amp_range=(-100,100), bins=200, title='Event %d' % i)
            gr.show(mode='do not hold') 

    gr.show()
 
#----------
#----------
#----------
#----------

def ex_image_with_random_peaks() :     
    img, peaks = image_with_random_peaks()
    print('peaks:')
    for i, (r0, c0, a0, sigma) in enumerate(peaks) :
        print('  %04d  row=%6.1f  col=%6.1f  amp=%6.1f  sigma=%6.3f' % (i, r0, c0, a0, sigma))
    plot_image(img)

#----------

def usage() :
    msg = 'Usage: [python] %s <test-number>'%(sys.argv[0])\
        + '\n  where <test-number> ='\
          '\n  0 - images'\
          '\n  3 - peak-finder V3'\
          '\n  4 - peak-finder V4'
    print(msg)
 
#----------

if __name__ == "__main__" :
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s:' % tname)
    if   tname == '0' : ex_image_with_random_peaks()
    elif tname in ('1','2','3','4') : test_pf(tname)
    else : usage(); sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of test %s' % tname)
 
#----------
