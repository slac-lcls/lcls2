#!/usr/bin/env python
#--------------------
"""
"""
import sys
import numpy as np
from time import time
import psana.pyalgos.generic.Graphics as gr
from psana.hexanode.WFUtils import bpf
from psana.pyalgos.generic.NDArrUtils import print_ndarr
from scipy.signal import find_peaks

ifname = '/reg/g/psdm/detector/data2_test/npy/waveforms-amox27716-r0100-e000010.npy'
wfs = np.load(ifname)
print('Waveforms loaded from file %s' % ifname)
print('Array shape: %s' % str(wfs.shape))

tname = sys.argv[1] if len(sys.argv) > 1 else '1'
binmin, binmax =\
  (7500,26500)  if tname == '1' else\
  (8500,15500)  if tname == '2' else\
  (10600,11600) if tname == '3' else\
  (0,44000)

iwf = 3
wf = wfs[iwf,binmin:binmax]
nbins = wf.size

kwargs = {}
fig = gr.figure(figsize=(12,10), dpi=80, facecolor='w', edgecolor='w', frameon=True, **kwargs)
fig.canvas.set_window_title('Waveform filters', **kwargs)
fig.clear()

def axes_h(fig, naxes=5, x0=0.07, y0=0.03, width=0.87, ygap=0.04) :
    dy = 1./naxes
    return [gr.add_axes(fig, axwin=(x0, y0 + i*dy, width, dy-ygap)) for i in range(naxes)]

#ax0, ax1, ax2, ax3 = axes = axes_h(fig)
ax0, ax1, ax2, ax3, ax4 = axes = axes_h(fig)

for i,ax in enumerate(axes) :
    ax.set_xlim(left=binmin, right=binmax)
    ax.set_ylabel('ax%d'%i)
    ax.grid(True)

#----------

ax0.set_ylabel('waveform')
ax0.set_xlabel('sample/time')
t = np.arange(binmin, binmax, 1)
#ax0.plot(t, wf, 'b-', linewidth=1, **kwargs)
ax0.hist(t, bins=nbins, weights=wf, color='b', histtype='step', **kwargs)

#----------

kwa = {'numchs': 5,
       'numhits'   : 16,
       'version'   :  3,
       'pf3_sigmabins'  :     3,
       'pf3_basebins'   :   100,
       'pf3_nstdthr'    :     5,
       'pf3_gapbins'    :   200,
       'pf3_deadbins'   :    10,
       'pf3_ioffsetbeg' :  1000,
       'pf3_ioffsetend' :  2000,
       'pf3_wfbinbeg'   : binmin,
       'pf3_wfbinend'   : binmax,
       }

t0_sec = time()
wfgi, wff, wfg, thrg, edges =\
    bpf(wf, sigmabins=kwa['pf3_sigmabins'], basebins=kwa['pf3_basebins'], nstdthr=kwa['pf3_nstdthr'], gapbins=kwa['pf3_gapbins'])
print('==== consumed time %.6f sec'%(time()-t0_sec))

for i,(b,e) in enumerate(edges) :
    print('gr:%02d range:%04d-%04d size:%04d'%(i, b, e, e-b))

smask = np.zeros_like(wfg, dtype=np.int16)
for (b,e) in edges : smask[b:e] = 1

#----------

ax0.set_ylabel('gaussian_filter1d')
ax0.hist(t, bins=nbins, weights=wff, color='y', histtype='step', **kwargs)

#----------

ax1.set_ylabel('gradient')
ax1.hist(t, bins=nbins, weights=wfg, color='b', histtype='step', **kwargs)
gr.drawLine(ax1, ax1.get_xlim(), (thrg,thrg), s=10, linewidth=1, color='k')
gr.drawLine(ax1, ax1.get_xlim(), (-thrg,-thrg), s=10, linewidth=1, color='k')

#----------

ax2.set_ylabel('signal selected')
ax2.hist(t, bins=nbins, weights=wfgi, color='r', histtype='step', **kwargs)
ax2.hist(t, bins=nbins, weights=0.1+smask*0.1, color='k', histtype='step', **kwargs)

#----------

ax3.set_ylabel('comparison')
ax3.hist(t, bins=nbins, weights=wf, color='b', histtype='step', **kwargs)
ax3.hist(t, bins=nbins, weights=wfgi, color='r', histtype='step', linewidth=2, **kwargs)
ax3.hist(t, bins=nbins, weights=0.2+smask*0.1, color='k', histtype='step', **kwargs)

#----------

ax4.set_ylabel('peakfinder')
ax4.hist(t, bins=nbins, weights=wfgi, color='r', histtype='step', linewidth=2, **kwargs)
pinds, _ = find_peaks(-wfgi, distance=kwa['pf3_deadbins'])
for p in pinds :
    pt = p + binmin
    gr.drawLine(ax4, (pt,pt), (0,wfgi[p]), s=10, linewidth=1, color='k')

#----------

if False :

    from psana.hexanode.WFPeaks import WFPeaks, peak_finder_v3

    ax4.set_ylabel('peakfinder')

    peaks = WFPeaks(**kwa)

    x = np.arange(0, wfs.shape[1], 1, dtype=None)
    wts = np.stack((x,x,x,x,x))
    t0_sec = time()
    nhits, pkinds, pkvals, pktsec = peaks(wfs,wts)
    print('  wf processing time(sec) = %8.6f' % (time()-t0_sec))

    print_ndarr(pkinds,'pkinds')

    #arr = peaks.wfgi
    arr = peaks.wfgi #peaks.wfgi, peaks.wff, peaks.wfg, peaks.thrg, peaks.edges

    ax4.hist(t, bins=nbins, weights=arr, color='b', histtype='step')

    ch_nhits = nhits[iwf]
    ch_pinds = pkinds[iwf]

    for p in ch_pinds[:ch_nhits] :
        pt = p + binmin
        gr.drawLine(ax4, (pt,pt), (0,wf[p]), s=10, linewidth=1, color='r')

#----------

gr.show()
gr.save_fig(fig, prefix='fig-', suffix='-b%04d-%04d-wf-bpf.png' % (binmin, binmax), **kwargs)

#----------
