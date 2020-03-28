#!/usr/bin/env python
#--------------------
""" https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.gaussian_filter1d.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
"""

import numpy as np
from time import time
from math import sqrt, pi, pow, exp #floor
#from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks, butter, sosfilt
from scipy import signal
#from scipy.signal import gaussian

import psana.pyalgos.generic.Graphics as gr
from psana.pyalgos.generic.NDAMath import exp_desc_step
from psana.hexanode.WFUtils import wavelet, split_consecutive
from psana.pyalgos.generic.NDArrUtils import print_ndarr

ifname = '/reg/g/psdm/detector/data2_test/npy/waveforms-amox27716-r0100-e000010.npy'
wfs = np.load(ifname)
print('Waveforms loaded from file %s' % ifname)
print('Array shape: %s' % str(wfs.shape))

basemin = 4000; basemax = basemin + 2000
#binmin = 8000; binmax = binmin + 7000
binmin = 10600; binmax = binmin + 1000
binmin = 8500; binmax = binmin + 18000
#binmin = 10750; binmax = binmin + 1024*8
#binmin = 10750; binmax = binmin + 1024*4
#binmin = 10750; binmax = binmin + 1024
#binmin = 10750; binmax = binmin + 256
#binmin = 10750; binmax = binmin + 512

iwf = 3

std   = wfs[iwf,basemin:basemax].std()
mean  = wfs[iwf,basemin:basemax].mean()
wf    = wfs[iwf,binmin:binmax] - mean
nbins = wf.size
THR   = -5*std
print('XXXX wf base level mean:%.3f std:%.3f THR:%.3f' %(mean,std,THR))

ones_wf = np.ones_like(wf)
zeros_wf = np.zeros_like(wf)

kwargs = {}
fig = gr.figure(figsize=(12,10), dpi=80, facecolor='w', edgecolor='w', frameon=True, **kwargs)
fig.canvas.set_window_title('Waveform filters', **kwargs)
fig.clear()


def axes_h(fig, naxes=4, x0=0.07, y0=0.03, width=0.87, ygap=0.04) :
    dy = 1./naxes
    return [gr.add_axes(fig, axwin=(x0, y0 + i*dy, width, dy-ygap)) for i in range(naxes)]

ax0, ax1, ax2, ax3 = axes = axes_h(fig)

for i,ax in enumerate(axes) :
    ax.set_xlim(left=binmin, right=binmax)
    ax.set_ylabel('ax%d'%i)
    ax.grid(True)

#colors = ('x', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'r', 'g', 'm', 'k')

#----------

ax0.set_ylabel('waveform')
ax0.set_xlabel('sample/time')
t = np.arange(binmin, binmax, 1)
#ax0.plot(t, wf, 'b-', linewidth=1, **kwargs)
ax0.hist(t, bins=wf.size, weights=wf, color='b', histtype='step', **kwargs)

#----------

ax1.set_ylabel('gaussian_filter1d')

t0_sec = time()
wff = gaussian_filter1d(wf, 3, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
print('==== gaussian_filter1d for signal consumed time %.6f sec'%(time()-t0_sec))

#ax0.plot(t, wff, 'g-', linewidth=1, **kwargs)
ax1.hist(t, bins=nbins, weights=wff, color='b', histtype='step', **kwargs)
gr.drawLine(ax1, ax1.get_xlim(), (THR,THR), s=10, linewidth=1, color='k')

cond_wff_sig = wff<THR
sig_wff = np.select([cond_wff_sig,], [ones_wf,], default=0)
ax1.hist(t, bins=nbins, weights=sig_wff*0.1, color='g', histtype='step', **kwargs)

#sideband region
if False :
  cond_wff_sb = np.logical_not(cond_wff_sig)
  sb_region = np.select([cond_wff_sb,], [ones_wf,], default=0)
  ax1.hist(t, bins=nbins, weights=sb_region*0.1, color='b', histtype='step', **kwargs)

wffs = np.select([cond_wff_sig,], [wff,], default=0)
print('wff.size', wff.size)
print('wffs.size', wffs.size)

# cipy.signal.find_peaks(x, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)[source]

peaks, _ = find_peaks(-wffs, height=None, distance=20)
print('peaks',peaks)

#peaks += binmin
#print('peaks',peaks)

for p in peaks :
    pt = p+binmin
    gr.drawLine(ax1, (pt,pt), (0,wff[p]), s=10, linewidth=3, color='r')

#----------

ax2.set_ylabel('gradient')
#np.gradient(f, *varargs, **kwargs)
wfg = np.gradient(wff)
ax2.hist(t, bins=nbins, weights=wfg, color='b', histtype='step', **kwargs)

std_wfg = wfg[1:100].std()
THRD = std_wfg*6 # wfg.min()/5
print('YYYY wf DERIVATIVE std:%.4f THR:%.4f'%(std_wfg,THRD))
gr.drawLine(ax2, ax2.get_xlim(), (-THRD,-THRD), s=10, linewidth=1, color='k')

cond_wfg = wfg<-THRD
sig_wfg = np.select([cond_wfg,], [ones_wf,], default=0)
ax2.hist(t, bins=nbins, weights=sig_wfg*0.005, color='g', histtype='step', **kwargs)

cond_wfg_sb = np.absolute(wfg)<THRD
sb_wfg = np.select([cond_wfg_sb,], [ones_wf,], default=0)
#ax2.hist(t, bins=nbins, weights=sb_wfg*0.01, color='r', histtype='step', **kwargs)

inds = np.nonzero(sig_wfg)[0] # index selects 1st dimension in tuple
print_ndarr(inds,"ZZZZ inds")

grinds = split_consecutive(inds)
print_ndarr(grinds,"ZZZZ argwhere")

for i,group in enumerate(grinds) :
    print('gr:%02d range:%04d-%04d size:%04d'%(i, group[0], group[-1], len(group)))
    pt = group[-1]+binmin
    gr.drawLine(ax2, (pt,pt), (0,-THRD), s=10, linewidth=3, color='r')
    #gr.drawLine(ax2, (pt,pt), (0,wff[p]), s=10, linewidth=1, color='r')

#----------

ax3.set_ylabel('cumsum')
#sig_wfg_only = np.select([cond_wfg_sb,], [zeros_wf,], default=1)
#wfgs = wfg * sig_wfg_only
#wfgi = np.cumsum(wfgs) * sig_wfg_only
wfgi = np.cumsum(wf)
ax3.hist(t, bins=nbins, weights=wfgi, color='b', histtype='step', **kwargs)

#----------
#----------
#----------
if False :

    ax4.set_ylabel('other filter')
    ##sos = butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
    #sos = butter(10, 10, 'hp', fs=5000, output='sos')
    #wff2 = sosfilt(sos, wff)

    #wff2 = wfnal.hilbert(wff, N=None)

    wff2 = signal.savgol_filter(wff, 5, 2)

    ax4.plot(t, wff, 'g-', linewidth=3, **kwargs)
    ax4.plot(t, wff2, 'r-', linewidth=1, **kwargs)

#----------

if False :
    xmin, xmax, dx = -100, 300, 1
    ax4.set_xlim((xmin, xmax)) # xmin, xmax = ax1.get_xlim()
    x = np.arange(xmin, xmax, dx)
    f = wavelet(x, xoff=0)
    ax4.plot(x, f, 'g-', linewidth=1, **kwargs)
    print('x.shape', x.shape)
    print('f.shape', f.shape)
    print('f.min', f.min())


gr.show()

gr.save_fig(fig, prefix='fig-', suffix='-b%04d-%04d-wf-filters.png' % (binmin, binmax), **kwargs)

#----------

