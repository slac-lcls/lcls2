#!/usr/bin/env python
#--------------------

import sys
import psana
from time import time
import numpy as np
from psana import DataSource
from ndarray import wfpkfinder_cfd

from psana.pyalgos.generic.NDArrUtils import print_ndarr
import psana.pyalgos.generic.Graphics as gr

#--------------------

BASE = 0.
THR = -0.05
CFR = 0.85
DEADTIME = 10.0
LEADINGEDGE = True # False # True

BBAV=1000
BEAV=2000

#TIME_RANGE=(0.0000000,0.0000111) # entire wf duration
#TIME_RANGE=(0.000003,0.000005)
TIME_RANGE=(0.0000014,0.0000056)

BBEG=6000
BEND=22000 # 44000-2
#BBEG=0
#BEND=43000 # 44000-2

EVSKIP = 0
EVENTS = 5 + EVSKIP

#dsname = 'exp=amox27716:run=100'
#src1 = 'AmoEndstation.0:Acqiris.1' # 'ACQ1'
#src2 = 'AmoEndstation.0:Acqiris.2' # 'ACQ2'

#--------------------

def draw_times(ax, wf, wt) :
    #wf -= wf[0:1000].mean()
    t0_sec = time()
    #wf  = np.array(WF, dtype=np.float64)
    pkvals = np.zeros((100,), dtype=np.float64)
    pkinds = np.zeros((100,), dtype=np.uint32)
    npks = wfpkfinder_cfd(wf, BASE, THR, CFR, DEADTIME, LEADINGEDGE, pkvals, pkinds)

    print('    wf proc  npks:%3d  time(sec) = %8.6f' % (npks, time()-t0_sec))

    #edges = np.array(((100,5000),(200,10000)))
    edges = zip(pkvals,pkinds)

    for (amp,ind) in edges :
        x0 = wt[int(ind)]
        xarr = (x0,x0)
        yarr = (amp,-amp)
        gr.drawLine(ax, xarr, yarr, s=10, linewidth=1, color='k')

#--------------------

fig = gr.figure(figsize=(15,15), title='Image')
fig.clear()

#gr.move_fig(fig, 200, 100)
##fig.canvas.manager.window.geometry('+200+100')

naxes = 5
ch = (0,1,2,3,4)
gfmt = ('b-', 'r-', 'g-', 'k-', 'm-', 'y-', 'c-', )
ylab = ('X1', 'X2', 'Y1', 'Y2', 'MCP', 'XX', 'YY', )

dy = 1./naxes

lw = 1
w = 0.87
h = dy - 0.04
x0, y0 = 0.07, 0.03

ax = [gr.add_axes(fig, axwin=(x0, y0 + i*dy, w, h)) for i in range(naxes)]

#--------------------

def draw_waveforms(wf, wt) :
    for i in range(naxes) :
        ax[i].clear()
        ax[i].set_xlim(TIME_RANGE)
        ax[i].set_ylabel(ylab[i], fontsize=14)

        ich = ch[i]
        print('  == ch:%2d %3s'%(ich,ylab[i]), end = '')

        wftot = wf[ich,:]
        wttot = wt[ich,:]

        wfsel = np.copy(wftot[BBEG:BEND])
        wtsel = np.copy(wttot[BBEG:BEND])

        wfsel -= wftot[BBAV:BEAV].mean()
        ax[i].plot(wtsel, wfsel, gfmt[i], linewidth=lw)

        gr.drawLine(ax[i], ax[i].get_xlim(), (THR,THR), s=10, linewidth=1, color='k')
        draw_times(ax[i], wfsel, wtsel)

    gr.draw_fig(fig)
    gr.show(mode='non-hold')

#--------------------
#---- Event loop ----
#--------------------

ds = DataSource(files='/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2')
orun = next(ds.runs())
det = orun.Detector('tmo_hexanode')
det_raw = det.raw

for n,evt in enumerate(orun.events()):

    if n<EVSKIP : continue
    if n>EVENTS : break

    print(50*'_', '\n Event # %d' % n)
    gr.set_win_title(fig, titwin='Event: %d' % n)

    wf = det_raw.waveforms(evt); print_ndarr(wf, '  wforms: ', last=4)
    wt = det_raw.times(evt);     print_ndarr(wt, '  wtimes: ', last=4)

    draw_waveforms(wf, wt)

gr.show()

#--------------------

sys.exit(0)

#--------------------
