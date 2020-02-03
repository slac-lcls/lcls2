#!/usr/bin/env python
#--------------------
""" https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
"""

import numpy as np
from time import time
from math import sqrt, pow, floor
from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d

import matplotlib.pyplot as plt
import pywt
import psana.pyalgos.generic.Graphics as gr

#t = np.linspace(-1, 1, 200, endpoint=False)
#sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)

ifname = '/reg/g/psdm/detector/data2_test/npy/waveforms-amox27716-r0100-e000010.npy'
wf = np.load(ifname)
print('Waveforms loaded from file %s' % ifname)
print('Array shape: %s' % str(wf.shape))


nlmin = 1
nlmax = 6
nlmin_sum = 2
wavelet = 'haar' # 'db2' # 'haar'

#binmin = 10000; binmax = binmin + 1024*16
binmin = 10000; binmax = binmin + 1024*2
#binmin = 10500; binmax = binmin + 1024
#binmin = 10750; binmax = binmin + 256

sig = wf[3,binmin:binmax]
t0_sec = time()
sigf = gaussian_filter1d(sig, 3, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
print('==== gaussian_filter1d for signal consumed time %.6f sec'%(time()-t0_sec))

dwt_max_level = pywt.dwt_max_level(sig.size, wavelet) # 12
print('==== pywt.dwt_max_level', dwt_max_level)

kwargs = {}
fig = plt.figure(figsize=(12,12), dpi=80, facecolor='w', edgecolor='w', frameon=True, **kwargs)
fig.canvas.set_window_title('Waveform dwt decomposition', **kwargs)
fig.clear()

def axes_h(fig, naxes=5, x0=0.07, y0=0.03, width=0.87, ygap=0.04) :
    dy = 1./naxes
    return [gr.add_axes(fig, axwin=(x0, y0 + i*dy, width, dy-ygap)) for i in range(naxes)]

ax0, ax1, ax2, ax3, ax4 = axes = axes_h(fig)

for i,ax in enumerate(axes) :
    ax.set_xlim(left=binmin, right=binmax)
    ax.set_ylabel('ax%d'%i)
    ax.grid(True)

ax0.set_xlabel('sample/time')
ax0.set_ylabel('Intensity')
ax1.set_ylabel('ca')
ax2.set_ylabel('cd')
ax3.set_ylabel('Layer')
ax4.set_ylabel('Sum')

t = np.arange(binmin, binmax, 1)
#ax0.plot(t, sig, 'b-', linewidth=1, **kwargs)
ax0.hist(t, bins=sig.size, weights=sig, color='b', histtype='step', **kwargs)
#ax0.hist(t, bins=sig.size, weights=sigf, color='y', histtype='step', **kwargs)

#linefmt = ('b-', 'r-', 'g-', 'm-', 'k-', 'k-', 'b-', 'r-', 'g-', 'm-', 'k-', 'k-')
colors = ('x', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'r', 'g', 'm', 'k',\
               'b', 'g', 'r', 'c', 'm', 'y', 'k', 'r', 'g', 'm', 'k',)
f = 1.0/sqrt(2)

img = np.zeros((nlmax, sig.size), dtype=np.float)
binds = np.arange(0, sig.size, dtype=np.uint)
print('img.shape',img.shape)

ca = sig

for nlev in range(nlmin,nlmax) :
    t0_sec = time()
    #ca = pywt.downcoef('a', sig, wavelet, mode='symmetric', level=nlev)
    #cd = pywt.downcoef('d', sig, wavelet, mode='symmetric', level=nlev)
    (ca, cd) = pywt.dwt(ca,wavelet)
    print('Level:%2d  nbins:%4d signal.cwt consumed time %.6f sec'%(nlev, ca.size, time()-t0_sec))

    if nlev<nlmin_sum : continue

    fl = pow(f,nlev)
    binwid = 1<<nlev
    tl = np.arange(binmin, binmax, binwid)
    csize = cd.size

    cap = np.array(ca*fl)
    cdp = np.array(cd*fl)

    ax1.hist(tl, bins=csize-1, weights=cap, color=colors[nlev], histtype='step', **kwargs)
    ax2.hist(tl, bins=csize-1, weights=cdp, color=colors[nlev], histtype='step', **kwargs)

    lbinds = np.floor(binds/binwid).astype(np.uint64)
    img[nlev,binds] = cdp[lbinds]

#ax0.clear()

imsh = ax3.imshow(img[nlmin:nlmax,:],
    aspect='auto', 
    origin='upper',
    cmap='gray',
    interpolation='nearest',
    extent=(binmin, binmax, nlmax-0.5, nlmin-0.5)) #cmap='inferno', 'jet', 'spring'
imsh.set_clim(img.min(), img.max())

prj = np.sum(img[nlmin_sum:nlmax,:], axis=0)
#ax4.plot(t, prj, 'b-', linewidth=1, **kwargs)
ax4.hist(t, bins=len(prj)-1, weights=prj, color='b', histtype='step', **kwargs)

prjc = gaussian_filter1d(prj, 5, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
ax4.plot(t, prjc, 'g-', linewidth=1, **kwargs)
ax4.hist(t, bins=len(prj), weights=prjc, color='g', histtype='step', **kwargs)


gr.show()

suf='-wf-dwt-decomp-l%02d:%02d-b%04d:%04d.png' % (nlmin, nlmax-1,binmin, binmax)
gr.save_fig(fig, prefix='fig-', suffix=suf, **kwargs)

#----------
