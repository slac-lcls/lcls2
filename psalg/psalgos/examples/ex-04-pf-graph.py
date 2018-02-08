#------------------------------
### #!/usr/bin/env python

from time import time
import numpy as np
from pyimgalgos.NDArrGenerators import random_standard, add_random_peaks, reshape_to_2d, add_ring
from ImgAlgos.PyAlgos import print_arr, print_arr_attr #, PyAlgos

import psalgos

import pyimgalgos.GlobalGraphics as gg

#------------------------------

def plot_image(img, img_range=None, amp_range=None, figsize=(12,10)) : 
    #import pyimgalgos.GlobalGraphics as gg
    axim = gg.plotImageLarge(img, img_range, amp_range, figsize)
    gg.show() 

#------------------------------

def image_with_random_peaks(shape=(1000, 1000), add_water_ring=True) : 
    img = random_standard(shape, mu=0, sigma=10)
    if add_water_ring : 
        rad = 0.3*shape[0]
        sigm = rad/4
        add_ring(img, amp=20, row=shape[0]/2, col=shape[1]/2, rad=rad, sigma=sigm)
    peaks = add_random_peaks(img, npeaks=50, amean=100, arms=25, wmean=1.5, wrms=0.3)
    return img, peaks

#------------------------------

hdr = 'Evnum  Reg  Seg  Row  Col  Npix      Amax      Atot   rcent   ccent '+\
      'rsigma  csigma rmin rmax cmin cmax    bkgd     rms     son' # +\
      #'  imrow   imcol     x[um]     y[um]     r[um]  phi[deg]'

fmt = '%5d  %3s  %3d %4d %4d  %4d  %8.1f  %8.1f  %6.1f  %6.1f %6.2f  %6.2f'+\
      ' %4d %4d %4d %4d  %6.2f  %6.2f  %6.2f' # +\
      #' %6d  %6d  %8.0f  %8.0f  %8.0f  %8.2f'

#------------------------------
V1 = 1 # DROPLET v1
V2 = 2 # FLOODF  v2r1
V3 = 3 # RANKER  v3r2
V4 = 4 # DROPLET v4r2
#------------------------------

def test_pf(tname) : 

    ##-----------------------------

    PF = V3 # default
    if tname == '1' : PF = V1
    if tname == '2' : PF = V2
    if tname == '3' : PF = V3
    if tname == '4' : PF = V4

    SKIP   = 0
    EVTMAX = 5 + SKIP

    DO_PLOT_IMAGE           = True
    DO_PLOT_CONNECED_PIXELS = True if PF in (V2,V3,V4) else False
    DO_PLOT_LOCAL_MAXIMUMS  = True if PF in (V3,V4) else False
    DO_PLOT_LOCAL_MINIMUMS  = True if PF == V3 else False

    #shape=(500, 500)
    shape=(1024, 1024)

    mask = np.ones(shape, dtype=np.uint16)

    # Pixel image indexes
    #arr3d = np.array((1,shape[0],shape[1]))

    INDS = np.indices((1,shape[0],shape[1]), dtype=np.int64)
    imRow, imCol = INDS[1,:], INDS[2,:]  
    #iX  = np.array(det.indexes_x(evt), dtype=np.int64) #- xoffset
    #iY  = np.array(det.indexes_y(evt), dtype=np.int64) #- yoffset

    fs = (8,7) # (11,10)
    ##-----------------------------
    fig5, axim5, axcb5, imsh5 = gg.fig_axim_axcb_imsh(figsize=fs) if DO_PLOT_LOCAL_MINIMUMS  else (None, None, None, None)
    fig4, axim4, axcb4, imsh4 = gg.fig_axim_axcb_imsh(figsize=fs) if DO_PLOT_LOCAL_MAXIMUMS  else (None, None, None, None)
    fig3, axim3, axcb3, imsh3 = gg.fig_axim_axcb_imsh(figsize=fs) if DO_PLOT_CONNECED_PIXELS else (None, None, None, None)
    fig1, axim1, axcb1, imsh1 = gg.fig_axim_axcb_imsh(figsize=fs) if DO_PLOT_IMAGE           else (None, None, None, None)
    ##-----------------------------

    #alg = PyAlgos(windows=None, mask=None, pbits=10) # 0177777)
    alg = psalgos.peak_finder_algos(seg=0, pbits=0)


    if   PF == V1 :
      alg.set_peak_selection_parameters(npix_min=0, npix_max=1e6, amax_thr=0, atot_thr=0, son_min=6)

    elif PF == V2 :
      alg.set_peak_selection_parameters(npix_min=0, npix_max=1e6, amax_thr=0, atot_thr=0, son_min=6)

    elif PF == V3 :
      alg.set_peak_selection_parameters(npix_min=1, npix_max=1e6, amax_thr=0, atot_thr=0, son_min=8)

    elif PF == V4 :
      alg.set_peak_selection_parameters(npix_min=1, npix_max=1e6, amax_thr=0, atot_thr=0, son_min=4)

    alg.print_attributes()

    for ev in range(EVTMAX) :
        ev1 = ev + 1

        if ev<SKIP : continue
        #if ev>=EVTMAX : break

        print 50*'_', '\nEvent %04d' % ev1

        add_water_ring = False if PF == V4 else True
        img, peaks_sim = image_with_random_peaks(shape, add_water_ring)

        # --- for debugging
        #np.save('xxx-image', img) 
        #np.save('xxx-peaks', np.array(peaks_sim)) 

        #img = np.load('xxx-image-crash.npy') 
        #peaks_sim = np.load('xxx-peaks-crash.npy') 
        # ---
        
        peaks_gen = [(0, r, c, a, a*s, 9*s*s) for r,c,a,s in peaks_sim]

        t0_sec = time()

        peaks = alg.peak_finder_v3r3_d2(img, mask, rank=4, r0=6, dr=3, nsigm=3) if PF == V3 else\
                alg.peak_finder_v4r3_d2(img, mask, thr_low=20, thr_high=40, rank=4, r0=5, dr=3)

        #peaks = alg.list_of_peaks_selected()
        #peaks_tot = alg.list_of_peaks()

#        peaks = alg.peak_finder_v1(img, thr_low=20, thr_high=40, radius=6, dr=2) if PF == V1 else\
#                alg.peak_finder_v2r1(img, thr=30, r0=7, dr=2)                    if PF == V2 else\
#                alg.peak_finder_v3r2(img, rank=5, r0=7, dr=2, nsigm=3)           if PF == V3 else\
#                alg.peak_finder_v4r2(img, thr_low=20, thr_high=40, rank=6, r0=7, dr=2)
#                #alg.peak_finder_v4r2(img, thr_low=20, thr_high=40, rank=6, r0=3.3, dr=0)

        print 'Time consumed by the peak_finder = %10.6f(sec) number of simulated/found peaks: %d/%d'%\
              (time()-t0_sec, len(peaks_sim), len(peaks))

        #map3 = reshape_to_2d(alg.maps_of_connected_pixels()) if DO_PLOT_CONNECED_PIXELS else None # np.zeros((10,10))
        #map4 = reshape_to_2d(alg.maps_of_local_maximums())   if DO_PLOT_LOCAL_MAXIMUMS  else None # np.zeros((10,10))
        #map5 = reshape_to_2d(alg.maps_of_local_minimums())   if DO_PLOT_LOCAL_MINIMUMS  else None # np.zeros((10,10))

        map3 = reshape_to_2d(alg.connected_pixels()) if DO_PLOT_CONNECED_PIXELS else None # np.zeros((10,10))
        map4 = reshape_to_2d(alg.local_maxima())     if DO_PLOT_LOCAL_MAXIMUMS  else None # np.zeros((10,10))
        map5 = reshape_to_2d(alg.local_minima())     if DO_PLOT_LOCAL_MINIMUMS  else None # np.zeros((10,10))

        #print_arr(map3, 'map_of_connected_pixels')
        #maps.shape = shape 

        #for i, (r0, c0, a0, sigma) in enumerate(peaks_sim) :
        #    print '  %04d  row=%6.1f  col=%6.1f  amp=%6.1f  sigma=%6.3f' % (i, r0, c0, a0, sigma)
        #plot_image(img)

        #print 'Found peaks:'
        #print hdr
        reg = 'IMG'

        peaks_rec = []

        if False :
          for pk in peaks :
            seg,row,col,npix,amax,atot,rcent,ccent,rsigma,csigma,\
            rmin,rmax,cmin,cmax,bkgd,rms,son = pk.parameters()
            #rmin,rmax,cmin,cmax,bkgd,rms,son = pk[0:17]
            rec = fmt % (ev, reg, seg, row, col, npix, amax, atot, rcent, ccent, rsigma, csigma,\
                  rmin, rmax, cmin, cmax, bkgd, rms, son) #,\
                  #imrow, imcol, xum, yum, rum, phi)
            peaks_rec.append((seg, row, col, amax, atot, npix))
            print rec

        peaks_rec = [(p.seg, p.row, p.col, p.amp_max, p.amp_tot, p.npix) for p in peaks]
        #s, r, c, amax, atot, npix = rec[0:6]


        if DO_PLOT_CONNECED_PIXELS :
            cmin, cmax = (map3.min(), map3.max()) if map3 is not None else (None,None)
            #print 'Connected pixel groups min/max:', cmin, cmax
            gg.plot_imgcb(fig3, axim3, axcb3, imsh3, map3, amin=cmin, amax=cmax, title='Connected pixel groups, ev: %04d' % ev1)
            gg.move_fig(fig3, x0=100, y0=30)


        if DO_PLOT_LOCAL_MAXIMUMS :
            gg.plot_imgcb(fig4, axim4, axcb4, imsh4, map4, amin=0, amax=10, title='Local maximums, ev: %04d' % ev1)
            gg.move_fig(fig4, x0=200, y0=30)


        if DO_PLOT_LOCAL_MINIMUMS :
            gg.plot_imgcb(fig5, axim5, axcb5, imsh5, map5, amin=0, amax=10, title='Local minimums, ev: %04d' % ev1)
            gg.move_fig(fig5, x0=300, y0=30)


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
            gg.plot_imgcb(fig1, axim1, axcb1, imsh1, img, amin=amin, amax=amax, title='Image, ev: %04d' % ev1)
            gg.move_fig(fig1, x0=400, y0=30)

            gg.plot_peaks_on_img(peaks_gen, axim1, imRow, imCol, color='g', lw=2) #, pbits=3)
            gg.plot_peaks_on_img(peaks_rec, axim1, imRow, imCol, color='w') #, pbits=3)

            fig1.canvas.draw() # re-draw figure content

            #gg.plotHistogram(nda, amp_range=(-100,100), bins=200, title='Event %d' % i)
            gg.show(mode='do not hold') 

    gg.show()
 
#------------------------------
#------------------------------
#------------------------------
#------------------------------

def ex_image_with_random_peaks() :     
    img, peaks = image_with_random_peaks()
    print 'peaks:'
    for i, (r0, c0, a0, sigma) in enumerate(peaks) :
        print '  %04d  row=%6.1f  col=%6.1f  amp=%6.1f  sigma=%6.3f' % (i, r0, c0, a0, sigma)
    plot_image(img)

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '3'
    print 50*'_', '\nTest %s:' % tname
    if   tname == '0' : ex_image_with_random_peaks()
    elif tname in ('3','3','3','4') : test_pf(tname)
    else : print 'Not-recognized test name: %s' % tname
    sys.exit('End of test %s' % tname)
 
#------------------------------
