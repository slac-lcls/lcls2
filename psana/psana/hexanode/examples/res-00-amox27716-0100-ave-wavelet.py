#!/usr/bin/env python
#--------------------
"""Averages the shape of a single pulse per channel.
"""
#--------------------

import sys
from time import time
import psana
import numpy as np
from Detector.WFDetector import WFDetector

import psana.pyalgos.generic.Graphics as gr
#from psana.pyalgos.generic.NDArrUtils import print_ndarr

from pypsalg import find_edges

from psalgos.pypsalgos import local_maxima_1d, local_minima_1d

from scipy.ndimage.filters import gaussian_filter1d

#from scipy.signal import savgol_filter
#        wfsmo = savgol_filter(wfsel, wfsel.size-1, 3)

#------------------------------

BASE = 0.
THR = -0.04 # -0.16 # -0.04
THRU = 2*THR
CFR = 0.9
DEADTIME = 5.0
LEADINGEDGE = True # False # True

BBAV=1000
BEAV=2000

#TIME_RANGE=(0.000003,0.000005)
TIME_RANGE=(0.0000034,0.0000038)
#TIME_RANGE=(0.0000000,0.0000111)

BBEG=6000
BEND=22000 # 44000-2
BNUM=BEND-BBEG
#BBEG=0
#BEND=43000 # 44000-2

EVSKIP = 5
EVENTS = 4 + EVSKIP
EVENTS = 1000 + EVSKIP

# event_keys -d exp=xpptut15:run=280 -m3
# event_keys -d exp=amox27716:run=100 -m3

#dsname = 'exp=xpptut15:run=280'
dsname = 'exp=amox27716:run=91'
#dsname = 'exp=amox27716:run=92'
#dsname = 'exp=amox27716:run=93'
#dsname = 'exp=amox27716:run=100'

ofname ='wavelets-%s.npy' % dsname.replace('exp=','').replace(':run=','-r')

src1 = 'AmoEndstation.0:Acqiris.1' # 'ACQ1'
src2 = 'AmoEndstation.0:Acqiris.2' # 'ACQ2'

print('Example for\n  dataset: %s\n  source1 : %s\n  source2 : %s' % (dsname, src1, src2))

#opts = {'psana.calib-dir':'./calib',}
#psana.setOptions(opts)
#psana.setOption('psana.calib-dir', './calib')
#psana.setOption('psana.calib-dir', './empty/calib')

#------------------------------

NBINSWL  = 2000
NBINSOFF = 200

class Store :
    def __init__(self):
        self.w = np.zeros((5,NBINSWL), dtype=np.float64)

sp = Store()

#------------------------------

def wf_extremas(ax, wf, wt, rank=10) :
    sh = wf.shape
    #mask = np.ones(sh, dtype=np.uint16).flatten()
    mask_min = np.array(wf<THR, dtype=np.uint16)
    #mask_max = np.array(wf>-THR, dtype=np.uint16)
    extrema = np.zeros(sh, dtype=np.uint16)
    t0_sec = time()
    #nmax = local_maxima_1d(wf, mask_max, rank, extrema)
    nmin = local_minima_1d(wf, mask_min, rank, extrema)
    #print('  consumed time = %10.6f(sec)  nmin = %d' % (time()-t0_sec, nmin))
    #print('  consumed time = %10.6f(sec)  nmin = %d  nmax = %d' % (time()-t0_sec, nmin, nmax))

    inds = np.where(extrema>1)
    amps = wf[inds]
    inds = inds[0]

    #print('inds:', inds)
    #print('amps:', amps)

    return np.array(zip(amps,inds))
    #return np.hstack((amps,inds))

#------------------------------

def draw_times(ax, wf, wt) :

    #extrema = 0.1* wf_extremas(ax, wf, wt, rank=10)        
    #gr.drawLine(ax, wt, extrema, s=10, linewidth=1, color='k')
    #return

    #wf -= wf[0:1000].mean()
    #edges = find_edges(wf, BASE, THR, CFR, DEADTIME, LEADINGEDGE)

    edges = wf_extremas(ax, wf, wt, rank=10)
    # pairs of (amplitude,sampleNumber)

    nhits = edges.shape[0]

    #print(' nhits:', nhits,)
    #print(' edges:', edges)

    for (amp,ind) in edges :
        x0 = wt[int(ind)]
        xarr = (x0,x0)
        yarr = (amp,-amp)
        gr.drawLine(ax, xarr, yarr, s=10, linewidth=1, color='k')

    return edges

#------------------------------

def draw_times_old(ax, wf, wt) :

    #wf -= wf[0:1000].mean()
    edges = find_edges(wf, BASE, THR, CFR, DEADTIME, LEADINGEDGE)
    # pairs of (amplitude,sampleNumber)
    #print(' nhits:', edges.shape[0],)
    #print(' edges:', edges)

    for (amp,ind) in edges :
        x0 = wt[int(ind)]
        xarr = (x0,x0)
        yarr = (amp,-amp)
        gr.drawLine(ax, xarr, yarr, s=10, linewidth=1, color='k')

    return edges

#------------------------------

ds  = psana.DataSource(dsname)
env = ds.env()
#nrun = evt.run()
#evt = ds.events().next()
#for key in evt.keys() : print(key)

det2 = WFDetector(src2, env) #, pbits=1022)
det1 = WFDetector(src1, env) #, pbits=1022)
#det1.print_attributes()

#------------------------------

fig = gr.figure(figsize=(15,15), title='Image')
fig.clear()
#gr.move_fig(fig, 200, 100)
#fig.canvas.manager.window.geometry('+200+100')

NAXES = 5
dy = 1./NAXES

lw = 1
w = 0.87
h = dy - 0.04
x0, y0 = 0.07, 0.03

ch = (2,3,4,5,6)
gfmt = ('b-', 'r-', 'g-', 'k-', 'm-', 'y-', 'c-', )
ylab = ('X1', 'X2', 'Y1', 'Y2', 'MCP', 'XX', 'YY', )
ax = [gr.add_axes(fig, axwin=(x0, y0 + i*dy, w, h)) for i in range(NAXES)]

#------------------------------
wf,wt = None, None

for n,evt in enumerate(ds.events()) :
    if n<EVSKIP : continue
    if n>EVENTS : break
    #print(50*'_', '\n Event # %d' % n)
    print('  Event # %d' % n)
    gr.set_win_title(fig, titwin='Event: %d' % n)

    #print('Acqiris.1:')
    result = det1.raw(evt)
    if result is None : continue
    wf,wt = result
    #print('wf.shape: ', wf.shape, 'wt.shape:', wt.shape # (8, 44000))

    #gu.print_ndarr(wf, 'acqiris waveform')
    #gu.print_ndarr(wt, 'acqiris wavetime')

    #print('Acqiris.2:')
    wf2,wt2 = det2.raw(evt)
    #print('wf2.shape: ', wf2.shape, 'wt2.shape:', wt2.shape) # (2, 44000)

    for i in range(NAXES) :
        ax[i].clear()
        ax[i].set_xlim(TIME_RANGE)
        ax[i].set_ylabel(ylab[i], fontsize=14)
        if i==6 : break

        wftot = wf[ch[i],:]
        wttot = wt[ch[i],:]

        wfsel = wf[ch[i],BBEG:BEND]
        wtsel = wt[ch[i],BBEG:BEND]

        wfsel -= wftot[BBAV:BEAV].mean()

        #print('  == ch:', ch[i],)

        wfsmo = gaussian_filter1d(wfsel, 4)
        #wfsmo = gaussian_filter1d(wfsmo, 5)
        #wfsmo = gaussian_filter1d(wfsmo, 4)

        #grad1 = 100*np.gradient(wfsmo, edge_order=2)
        #grad2 = 10*np.gradient(grad1, edge_order=2)

        #edges = draw_times(ax[i], wfsel, wtsel)
        edges = draw_times(ax[i], wfsmo, wtsel)

        nhits = edges.shape[0]

        if nhits != 1 : continue

        amp,ind = edges[0]

        #if amp<THRU : continue
        if (ind<NBINSOFF) or (BNUM-ind<NBINSWL) : continue

        sp.w[i,:] += wfsmo[ind-NBINSOFF:ind+NBINSWL-NBINSOFF]

        #ax[i].plot(wtsel, wfsel, gfmt[i], linewidth=lw)
        ax[i].plot(wtsel, wfsmo, gfmt[0], linewidth=lw)

        #ax[i].plot(wtsel, grad2, gfmt[0], linewidth=lw)

        gr.drawLine(ax[i], ax[i].get_xlim(), (THR,THR), s=10, linewidth=1, color='k')

    #wf2sel = wf2[ch[i],BBEG:BEND]
    #wt2sel = wt2[ch[i],BBEG:BEND]

    #ax[i].plot(wt2sel, wf2sel, gfmt[i], linewidth=lw)
    #edges = draw_times(ax[i], wf2sel, wt2sel)

    gr.draw_fig(fig)
    gr.show(mode='non-hold')

for i in range(NAXES) :
    ax[i].clear()
    ax[i].plot(wttot[:NBINSWL], sp.w[i,:], gfmt[0], linewidth=lw)
    gr.drawLine(ax[i], ax[i].get_xlim(), (0,0), s=10, linewidth=1, color='k')

gr.show()

np.save(ofname, sp.w)
print('Wavelets saved in file %s' % ofname)

#ch=0
#fig, ax = gr.plotGraph(wt[ch,:-1], wf[ch,:-1], figsize=(15,5))
#gr.show()

#------------------------------

sys.exit(0)

#------------------------------
