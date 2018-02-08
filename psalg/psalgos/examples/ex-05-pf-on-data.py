#!/usr/bin/env python

import sys
import math
import numpy as np
from time import time
import psana

from Detector.AreaDetector import AreaDetector
from pyimgalgos.GlobalUtils import print_ndarr

#from pyimgalgos.GlobalUtils import subtract_bkgd
from psalgos.pypsalgos import peaks_adaptive # import * 

import pyimgalgos.GlobalGraphics as gg

#------------------------------

ntest = int(sys.argv[1]) if len(sys.argv)>1 else 1
print 'Test # %d' % ntest

#------------------------------
SKIP        = 3
EVTMAX      = 1 + SKIP
EVTPLOT     = 1 
DO_PLOT     = True
#------------------------------

# Chuck 2017-08-04 exp=mfxn8416:run=95 event number 3  # event_keys -d exp=mfxn8416:run=95
# EventKey(type=psana.Camera.FrameV1, src='DetInfo(MfxEndstation.0:Rayonix.0)', alias='Rayonix')
# EventKey(type=psana.Epix.ElementV3, src='DetInfo(MfxEndstation.0:Epix100a.0)', alias='Epix100a')
  
#dsname = 'exp=cxif5315:run=169'
#src    = psana.Source('DetInfo(CxiDs2.0:Cspad.0)')

runnum = 95
dsname = 'exp=mfxn8416:run=%d' % runnum
#src    = psana.Source('MfxEndstation.0:Epix100a.0')
src    = psana.Source('MfxEndstation.0:Rayonix.0')
print '%s\nExample for\n  dataset: %s\n  source : %s' % (85*'_',dsname, src)

# Non-standard calib directory
#psana.setOption('psana.calib-dir', './empty/calib')
#psana.setOption('psana.calib-dir', '/reg/d/psdm/CXI/cxif5315/calib')

ds  = psana.DataSource(dsname)
env = ds.env()
#evt = ds.events().next()
#runnum = evt.run()

#run = ds.runs().next()
#runnum = run.run()

#for key in evt.keys() : print key

##-----------------------------

det = AreaDetector(src, env, pbits=0)
print 85*'_', '\nInstrument: %s  run number: %d' % (det.instrument(), runnum)

nda_peds  = det.pedestals(runnum)
print_ndarr(nda_peds, 'nda_peds')

#nda_bkgd  = det.bkgd(runnum)
#smask = det.mask(runnum, calib=False, status=True, edges=True, central=True, unbond=True, unbondnbrs=True)
mask = det.mask(runnum, calib=False, status=True, edges=True).astype(np.uint16)

mask = np.ones(nda_peds.shape, dtype=np.uint16)

##-----------------------------
#mask_img = np.loadtxt('../rel-mengning/work/roi_mask_nda_equ_arc.txt')
#mask_arc.shape = mask_equ.shape = mask_img.shape = nda_peds.shape

#------------------------------

#xoffset, yoffset = 300, 300
#xsize,   ysize   = 1150, 1150

#xoffset, yoffset = 0, 0
#xsize,   ysize   = (1920, 1920) #(768,704) # 800, 800 # (704, 768)

xoffset, yoffset = 600, 600
xsize,   ysize   = 700, 700

# Pixel image indexes
iX  = np.array(det.indexes_x(runnum), dtype=np.int64) #- xoffset
iY  = np.array(det.indexes_y(runnum), dtype=np.int64) #- yoffset

# Protect indexes (should be POSITIVE after offset subtraction)
imRow = np.select([iX<xoffset], [0], default=iX-xoffset)
imCol = np.select([iY<yoffset], [0], default=iY-yoffset)

# Pixel coordinates [um] (transformed as needed)
Xum =  det.coords_y(runnum)
Yum = -det.coords_x(runnum)

# Derived pixel raduius in [um] and angle phi[degree]
Rum = np.sqrt(Xum*Xum + Yum*Yum)
Phi = np.arctan2(Yum,Xum) * 180 / np.pi

imRow.shape  = imCol.shape  = \
Xum.shape    = Yum.shape    = \
Rum.shape    = Phi.shape    = det.shape()

#------------------------------
fig1, axim1, axcb1, imsh1 = gg.fig_axim_axcb_imsh(figsize=(12,11))
#------------------------------

t0_sec_evloop = time()
nda = None
peaks = None

# loop over events in data set
for evnum, evt in enumerate(ds.events()) :

    if evnum%100==0 : print 'Event %d' % (evnum)

    if evnum<SKIP   : continue
    if evnum>=EVTMAX : break

    # get calibrated data ndarray and proccess it if it is available
    t1_sec = time()
    #nda = det.calib(evt)

    # Apply custom calibration: raw, -peds, -bkgd, *smask, -cmod
    nda_raw = det.raw(evt)

    #print_ndarr(nda_raw, 'nda_raw')

    if nda_raw is not None :

        nda =  np.array(nda_raw, dtype=np.float32, copy=True)
        nda -= nda_peds
        #nda =  subtract_bkgd(nda, nda_bkgd, mask=nda_smask, winds=winds_bkgd, pbits=0)
        #nda *= nda_smask
        #det.common_mode_apply(evt, nda)

        #print '  ----> calibration dt = %f sec' % (time()-t1_sec)

        #print_ndarr(nda, 'data: raw-peds')

        #print_arr_attr(nda, 'calibrated data')
        t0_sec = time()

        peaks = peaks_adaptive(nda, mask, rank=3, r0=3.0, dr=4.0, nsigm=3,\
                               npix_min=2, npix_max=None, amax_thr=0, atot_thr=0, son_min=3)

        ###===================
        print 'Event %d --- dt/evt = %f sec  img.shape=%s  number of peaks: %d' % (evnum, time()-t0_sec, str(nda.shape), len(peaks))
        ###===================

        #for p in peaks : 
        #    #print dir(p)
        #    print '  seg:%4d, row:%4d, col:%4d, npix:%4d, son:%4.1f' % (p.seg, p.row, p.col, p.npix, p.son)

        #peaks_equ = alg_equ.peak_finder_v2r1(nda, thr=30, r0=7, dr=2)
        #peaks_equ = alg_equ.peak_finder_v3r1(nda, rank=5, r0=7, dr=2, nsigm=0) # 1.64 (5%)
        #peaks_equ = alg_equ.peak_finder_v4r1(nda, thr_low=10, thr_high=150, rank=5, r0=7, dr=2)
        #peaks_equ = alg_equ.peak_finder_v4r2(nda, thr_low=10, thr_high=150, rank=5, r0=7, dr=2)
 
        peaks_rec = [(int(p.seg), int(p.row), int(p.col), p.amp_max, p.amp_tot, int(p.npix)) for p in peaks]

        if DO_PLOT and evnum%EVTPLOT==0 :

            #nda = maps_of_conpix_arc        
            #nda = maps_of_conpix_equ        
            #nda = nda_bkgd
            #nda = nda_bkgd + regs_check      
            #img = det.image(evt, nda)
            img = det.image(evt, nda)[xoffset:xoffset+xsize,yoffset:yoffset+ysize]
            #img = det.image(evt, mask_img*nda)[xoffset:xoffset+xsize,yoffset:yoffset+ysize]
            #img = det.image(evt, maps_of_conpix_equ)[xoffset:xoffset+xsize,yoffset:yoffset+ysize]
           # ave, rms = img.mean(), img.std()
           # amin, amax = max(0,ave-1*rms), ave+3*rms
            amin, amax = 0, 1000

            axim1.clear()
            if imsh1 is not None : del imsh1
            imsh1 = None
            gg.plot_imgcb(fig1, axim1, axcb1, imsh1, img, amin=amin, amax=amax, title='Image, ev: %04d' % evnum, cmap='inferno') 
            # cmap='inferno', 'gray, gray_r, jet, jet_r, magma, magma_r, ocean, ocean_r, pink,
            gg.move_fig(fig1, x0=400, y0=30)

            gg.plot_peaks_on_img(peaks_rec, axim1, imRow, imCol, color='w', lw=2) #, pbits=3)

            #fig.canvas.set_window_title('Event: %d' % i)    
            fig1.canvas.draw() # re-draw figure content

            gg.show(mode='do not hold') 

gg.show()

print ' ----> Total script execution time = %f sec' % (time()-t0_sec_evloop)

#pstore.close_file()

##-----------------------------

gg.show() # hold image untill it is closed
 
##-----------------------------

sys.exit('Test is completed')

##-----------------------------
