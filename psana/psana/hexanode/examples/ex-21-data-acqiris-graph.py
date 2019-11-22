#!/usr/bin/env python
#--------------------

import sys
import psana
from time import time
import numpy as np
from psana import DataSource

from psana.pyalgos.generic.NDArrUtils import print_ndarr
import psana.pyalgos.generic.Graphics as gr

#from ndarray import wfpkfinder_cfd
from psana.hexanode.WFPeaks import WFPeaks

#--------------------

# Parameters of the CFD descriminator for hit time finding algotithm
cfdpars= {'cfd_base'       :  0.,
          'cfd_thr'        : -0.05,
          'cfd_cfr'        :  0.85,
          'cfd_deadtime'   :  10.0,
          'cfd_leadingedge':  True,
          'cfd_ioffsetbeg' :  1000,
          'cfd_ioffsetend' :  2000,
          'cfd_wfbinbeg'   :  6000,
          'cfd_wfbinend'   : 22000,
         }

peaks = WFPeaks(**cfdpars) # algorithm


#TIME_RANGE=(0.0000000,0.0000111) # entire wf duration
#TIME_RANGE=(0.000003,0.000005)
TIME_RANGE=(0.0000014,0.0000056)

EVSKIP = 0
EVENTS = 5 + EVSKIP

#--------------------

def draw_times(ax, wt, pkvals, pkinds) :

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
ch = (0,1,2,3,4,5,6)
gfmt = ('b-', 'r-', 'g-', 'k-', 'm-', 'y-', 'c-', )
ylab = ('X1', 'X2', 'Y1', 'Y2', 'MCP', 'XX', 'YY', )

dy = 1./naxes

lw = 1
w = 0.87
h = dy - 0.04
x0, y0 = 0.07, 0.03

ax = [gr.add_axes(fig, axwin=(x0, y0 + i*dy, w, h)) for i in range(naxes)]

#--------------------

def draw_waveforms(wfs, wts) :

    t0_sec = time()
    nhits, pkinds, pkvals, pktns = peaks(wfs,wts)
    print('    wf proc time(sec) = %8.6f' % (time()-t0_sec))

    p = peaks

    print_ndarr(nhits, '  nhits: ', last=10)

    for i in range(naxes) :
        ax[i].clear()
        ax[i].set_xlim(TIME_RANGE)
        ax[i].set_ylabel(ylab[i], fontsize=14)

        ich = ch[i]
        print('  == ch:%2d %3s'%(ich,ylab[i]), end = '')

        wftot = wfs[ich,:]
        wttot = wts[ich,:]

        wfsel = np.copy(wftot[p.WFBINBEG:p.WFBINEND])
        wtsel = np.copy(wttot[p.WFBINBEG:p.WFBINEND])

        wfsel -= wftot[peaks.IOFFSETBEG:peaks.IOFFSETEND].mean()
        ax[i].plot(wtsel, wfsel, gfmt[i], linewidth=lw)

        gr.drawLine(ax[i], ax[i].get_xlim(), (p.THR,p.THR), s=10, linewidth=1, color='k')
        draw_times(ax[i], wtsel, pkvals[i], pkinds[i])

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

    wfs = det_raw.waveforms(evt); print_ndarr(wfs, '  wforms: ', last=4)
    wts = det_raw.times(evt);     print_ndarr(wts, '  wtimes: ', last=4)

    draw_waveforms(wfs, wts)

gr.show()

#--------------------

sys.exit(0)

#--------------------
