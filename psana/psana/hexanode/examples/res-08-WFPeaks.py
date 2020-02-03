#!/usr/bin/env python
#--------------------
""" https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.gaussian_filter1d.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
"""
import sys
import numpy as np
from time import time
from scipy.ndimage.filters import gaussian_filter1d
from psana.pyalgos.generic.NDArrUtils import print_ndarr
import psana.pyalgos.generic.Graphics as gr
from psana.hexanode.WFPeaks import WFPeaks, peak_finder_v2

#----------

ifname = '/reg/g/psdm/detector/data2_test/npy/waveforms-amox27716-r0100-e000010.npy'
wfs = np.load(ifname)
print('Waveforms loaded from file %s' % ifname)
print_ndarr(wfs,'wfs')

t = np.arange(0, wfs.shape[1], 1, dtype=None)
wts = np.stack((t,t,t,t,t))
print_ndarr(wts,'wts')

#----------

if True :

    print(50*'_')
    kwa = {'numchs'   : 5,
           'numhits'  : 16,
           'version'  : 2,
           'pf2_sigmabins'  :     3,
           'pf2_nstdthr'    :    -3,
           'pf2_deadbins'   :  10.0,
           'pf2_ioffsetbeg' :  1000,
           'pf2_ioffsetend' :  2000,
           'pf2_wfbinbeg'   :  6000,
           'pf2_wfbinend'   : 22000,
          }

basemin = kwa['pf2_ioffsetbeg']
basemax = kwa['pf2_ioffsetend']
binmin  = kwa['pf2_wfbinbeg']
binmax  = kwa['pf2_wfbinend']

iwf  = 3
std  = wfs[iwf,basemin:basemax].std()
mean = wfs[iwf,basemin:basemax].mean()
wf   = wfs[iwf,binmin:binmax] - mean

ones_wf = np.ones_like(wf)
#zeros_wf = np.zeros_like(wf)

SIGMABINS = kwa['pf2_sigmabins']
DEADBINS  = kwa['pf2_deadbins']
NSTDTHR   = kwa['pf2_nstdthr']
THR       = NSTDTHR*std

#shret = (kwa['numchs'],kwa['numhits'])
shret = (kwa['numhits'],)
pkvals = np.zeros(shret, dtype=np.double)
pkinds = np.zeros(shret, dtype=np.uint32)
pktsec = np.zeros(shret, dtype=np.double)

#----------

print('===== test WFUtils.peak_finder_v2 =====')

t0_sec = time()
npeaks = peak_finder_v2(wf, SIGMABINS, THR, DEADBINS, pkvals, pkinds)
print('==== peak_finder_v2 consumed time %.6f sec'%(time()-t0_sec))
    
wff = gaussian_filter1d(wf, SIGMABINS)

print_ndarr(wff,'wff')
print_ndarr(pkinds,'pkinds', last=10)
print_ndarr(pkvals,'pkvals', last=10)

kwargs = {}
fig = gr.figure(figsize=(12,8), dpi=80, facecolor='w', edgecolor='w', frameon=True, **kwargs)

fig.canvas.set_window_title('Test WFUtils.peak_finder_v2 and WFPeaks', **kwargs)
fig.clear()

def axes_h(fig, naxes=2, x0=0.07, y0=0.05, width=0.87, ygap=0.04) :
    dy = (1-y0)/naxes
    return [gr.add_axes(fig, axwin=(x0, y0 + i*dy, width, dy-ygap)) for i in range(naxes)]

ax0, ax1  = axes = axes_h(fig)

for i,ax in enumerate(axes) :
    ax.set_xlim(left=binmin, right=binmax)
    ax.set_ylabel('ax%d'%i)
    ax.grid(True)

#----------

ax0.set_xlabel('sample/time')
ax0.set_ylabel('waveform')
t = np.arange(binmin, binmax, 1)
#ax0.plot(t, sig, 'b-', linewidth=1, **kwargs)
ax0.hist(t, bins=wf.size, weights=wf, color='b', histtype='step', **kwargs)
ax0.hist(t, bins=wff.size, weights=wff, color='y', histtype='step', **kwargs)

gr.drawLine(ax0, ax0.get_xlim(), (THR,THR), s=10, linewidth=1, color='k')

cond_wff = wff<THR
sig_wff = np.select([cond_wff,], [ones_wf,], default=0)
ax0.hist(t, bins=wff.size, weights=cond_wff*0.1, color='g', histtype='step', **kwargs)

print('XXXX pkinds', pkinds)

for p in pkinds :
    pt = p+binmin
    gr.drawLine(ax0, (pt,pt), (0,wf[p]), s=10, linewidth=2, color='r')

#----------

if True :

    print('===== test WFPeaks =====')

    peaks = WFPeaks(**kwa)
    #peaks.proc_waveforms(wfs, wts)

    t0_sec = time()

    #======== peak-finding algorithm ============
    #wfs, wts = array_of_selected_channels(wfs), array_of_selected_channels(wts)
    nhits, pkinds, pkvals, pktsec = peaks(wfs,wts)
    dt_sec = time()-t0_sec
    wfssel,wtssel = peaks.waveforms_preprocessed(wfs, wts) # selected time range and subtracted offset

    print_ndarr(wtssel,'  wtssel: ', last=4)
    print('  wf processing time(sec) = %8.6f' % dt_sec)
    print_ndarr(nhits, '  nhits : ', last=10)
    print_ndarr(pktsec,'  pktsec: ', last=4)
    print_ndarr(pkvals,'  pkvals: ', last=4)
    print_ndarr(pkinds,'  pkinds: ', last=4)
    print_ndarr(pkinds[iwf],'  pkinds[%d]: '%iwf, last=4)

    #============================================

    ax1.set_ylabel('direct test WFPeaks')
    t = np.arange(binmin, binmax, 1)
    #ax0.plot(t, sig, 'b-', linewidth=1, **kwargs)
    ax1.hist(t, bins=wf.size, weights=wf, color='b', histtype='step', **kwargs)

    #iwf = 3
    ch_nhits = nhits[iwf]
    ch_pinds = pkinds[iwf]

    for p in ch_pinds[:ch_nhits] :
        pt = p + binmin
        gr.drawLine(ax1, (pt,pt), (0,wf[p]), s=10, linewidth=1, color='r')

#----------

gr.show()

gr.save_fig(fig, prefix='fig-', suffix='-b%04d-%04d-WFPeaks.png' % (binmin, binmax), **kwargs)

#----------
