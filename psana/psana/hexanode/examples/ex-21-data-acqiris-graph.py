#!/usr/bin/env python
#----------
"""
   See example at the end of this file in __main__:

   - loop over events of psana dataset (xtc2 file), 
   - draw acqiris waveforms in selected ROI for all quad-DLD channels,
   - for each waveform run peakfinder peaks(wfs,wts) and
     draw peak times as vertical lines.
"""

from time import time
import numpy as np

from psana import DataSource
from psana.hexanode.WFPeaks import WFPeaks
from psana.pyalgos.generic.NDArrUtils import print_ndarr
import psana.pyalgos.generic.Graphics as gr

#----------

# parameters for CFD descriminator - waveform processing algorithm
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

# algorithm initialization in global scope
peaks = WFPeaks(**cfdpars)

#----------
# global parameters for graphics

time_range_sec=(0.0000014,0.0000056)
#time_range_sec=(0.0000000,0.0000111) # entire wf duration in this experiment

naxes = 5 # 5 for quad- or 7 for hex-anode
# assumes that lcls2 detector data returns channels 
# in desired order for u1, u2, v1, v2, [w1, w2,] mcp

gfmt = ('b-', 'r-', 'g-', 'k-', 'm-', 'y-', 'c-', )
ylab = ('X1', 'X2', 'Y1', 'Y2', 'MCP', 'XX', 'YY', )

dy = 1./naxes
lw = 1
w = 0.87
h = dy - 0.04
x0, y0 = 0.07, 0.03

fig = gr.figure(figsize=(15,15), title='Image')
fig.clear()
ax = [gr.add_axes(fig, axwin=(x0, y0 + i*dy, w, h)) for i in range(naxes)]


#----------

def array_of_selected_channels(a, ch = (2,3,4,5,6)) :
    """converts shape:(8, 44000) -> shape:(5, 44000)"""
    return a[ch,:]

#----------

def draw_waveforms(wfs, wts, nev) :
    """Draws all waveforms on figure axes, one waveform per axis.
       Parameters:
       - wfs [np.array] shape=(NUM_CHANNELS, NUM_SAMPLES) - waveform intensities
       - wts [np.array] shape=(NUM_CHANNELS, NUM_SAMPLES) - waveform times
    """
    t0_sec = time()

    #======== peak-finding algorithm ============
    #wfs, wts = array_of_selected_channels(wfs), array_of_selected_channels(wts)
    nhits, pkinds, pkvals, pktsec = peaks(wfs,wts)
    dt_sec = time()-t0_sec
    wfssel,wtssel = peaks.waveforms_preprocessed(wfs, wts) # selected time range and subtracted offset
    thr = peaks.THR

    #============================================

    print_ndarr(wtssel,'  wtssel: ', last=4)
    print('  wf processing time(sec) = %8.6f' % dt_sec)
    print_ndarr(nhits, '  nhits : ', last=10)
    print_ndarr(pkinds,'  pkinds: ', last=4)
    print_ndarr(pktsec,'  pktsec: ', last=4)
    print_ndarr(pkvals,'  pkvals: ', last=4)

    for ch in range(naxes) :
        ax[ch].clear()
        ax[ch].set_xlim(time_range_sec)
        ax[ch].set_ylabel(ylab[ch], fontsize=14)

        # draw waveform
        ax[ch].plot(wtssel[ch], wfssel[ch], gfmt[ch], linewidth=lw)

        # draw line for threshold level
        gr.drawLine(ax[ch], ax[ch].get_xlim(), (thr,thr), s=10, linewidth=1, color='k')

        # draw lines for peak times
        draw_times(ax[ch], pkvals[ch], pkinds[ch], wtssel[ch])

    gr.set_win_title(fig, 'Event: %d' % nev)
    gr.draw_fig(fig)
    gr.show(mode='non-hold')

#----------

def draw_times(axis, pkvals, pkinds, wt) :
    """Adds to figure axis a set of vertical lines for found peaks.
       Parameters:
       - axis - figure axis to draw a single waveform
       - pkvals [np.array] - 1-d peak values 
       - pkinds [np.array] - 1-d peak indexes in wt
       - wt [np.array] - 1-d waveform sample times - is used to get time [sec] from pkinds
    """
    for v,i in zip(pkvals,pkinds) :
        t = wt[i]
        gr.drawLine(axis, (t,t), (-v,v), s=10, linewidth=1, color='k')

#----------

if __name__ == "__main__" :

    EVSKIP = 0
    EVENTS = 10 + EVSKIP

    #ds   = DataSource(files='/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2')
    ds   = DataSource(files='/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e001000.xtc2')
    orun = next(ds.runs())
    det  = orun.Detector('tmo_quadanode') # 'tmo_hexanode'

    for n,evt in enumerate(orun.events()):
        if n<EVSKIP : continue
        if n>EVENTS : break
        print('%s\nEvent # %d' % (50*'_',n))

        wfs = det.raw.waveforms(evt); print_ndarr(wfs, '  wforms: ', last=4)
        wts = det.raw.times(evt);     print_ndarr(wts, '  wtimes: ', last=4)

        draw_waveforms(wfs, wts, n)

    gr.show()

#----------
