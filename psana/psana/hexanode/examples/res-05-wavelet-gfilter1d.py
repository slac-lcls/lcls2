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
#from scipy.signal import gaussian

import psana.pyalgos.generic.Graphics as gr
from psana.pyalgos.generic.NDAMath import exp_desc_step

ifname = '/reg/g/psdm/detector/data2_test/npy/waveforms-amox27716-r0100-e000010.npy'
wf = np.load(ifname)
print('Waveforms loaded from file %s' % ifname)
print('Array shape: %s' % str(wf.shape))

basemin = 4000; basemax = basemin + 2000

binmin = 8000; binmax = 20000
#binmin = 10000; binmax = binmin + 1024*16
#binmin = 10750; binmax = binmin + 1024*8
#binmin = 10750; binmax = binmin + 1024*4
#binmin = 10750; binmax = binmin + 1024
#binmin = 10750; binmax = binmin + 256
#binmin = 10750; binmax = binmin + 512

iwf = 3
sig = wf[iwf,binmin:binmax] - wf[iwf,basemin:basemax].mean()
nbins = sig.size

THR = -0.1

kwargs = {}
fig = gr.figure(figsize=(12,10), dpi=80, facecolor='w', edgecolor='w', frameon=True, **kwargs)
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
ax0.set_ylabel('Signal')
ax1.set_ylabel('gaussian_filter1d')
ax3.set_ylabel('decomposition')
ax4.set_ylabel('wavelet')
ax2.set_ylabel('ax2')

#colors = ('x', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'r', 'g', 'm', 'k')

#----------

t0_sec = time()
sigf = gaussian_filter1d(sig, 3, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
#sos = butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
#sos = butter(10, 10, 'hp', fs=10000, output='sos')
#sigf = sosfilt(sos, sigf)

print('==== gaussian_filter1d for signal consumed time %.6f sec'%(time()-t0_sec))

t = np.arange(binmin, binmax, 1)
#ax0.plot(t, sig, 'b-', linewidth=1, **kwargs)
ax0.hist(t, bins=sig.size, weights=sig, color='b', histtype='step', **kwargs)

#----------

#sigf = gaussian_filter1d(sig, 5, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
#ax0.plot(t, sigf, 'g-', linewidth=1, **kwargs)
ax1.hist(t, bins=nbins, weights=sigf, color='g', histtype='step', **kwargs)
gr.drawLine(ax1, ax1.get_xlim(), (THR,THR), s=10, linewidth=1, color='k')

thrmask = sigf<THR
ax1.hist(t, bins=nbins, weights=thrmask*0.1, color='r', histtype='step', **kwargs)


#grinds = np.transpose(np.array(np.argwhere(thrmask)))
grinds = np.array(np.nonzero(thrmask))
print('XXXXXXXXX grinds.shape:',grinds.shape)
for i,group in enumerate(grinds) :
    print('gr:%02d range:%04d-%04d n in group:%04d'%(i, group[0], group[-1], len(group)))


sigm = np.select([thrmask,], [sigf,], default=0)
print('sigf.size', sigf.size)
print('sigm.size', sigm.size)


peaks, _ = find_peaks(-sigm, height=None, distance=20)
# cipy.signal.find_peaks(x, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)[source]

print('peaks',peaks)

#peaks += binmin
#print('peaks',peaks)


for p in peaks :
    pt = p+binmin
    gr.drawLine(ax1, (pt,pt), (0,sigf[p]), s=10, linewidth=1, color='b')

#----------
#----------
#----------

def wavelet(x, scx=0.5, xoff=-2.4, trise=11, tdecr=40, n=5) :
    a = 1./trise
    g = 1./tdecr
    b = a + g
    B = -(a+b)*n/(2*a*b*scx)
    C = n*(n-1)/(a*b*scx*scx)
    xpk = -B-sqrt(B*B-C)
    x0 = xpk*scx
    norm = pow(x0,n-1)*(n-a*x0)*exp(-b*x0)
    xc = (x+xpk+xoff)*scx
    return -np.power(xc,n-1)*(n-a*xc)*exp_desc_step(b*xc) / norm

if True :
    xmin, xmax, dx = -100, 300, 1
    ax4.set_xlim((xmin, xmax)) # xmin, xmax = ax1.get_xlim()
    x = np.arange(xmin, xmax, dx)
    f = wavelet(x, xoff=0)
    ax4.plot(x, f, 'g-', linewidth=1, **kwargs)
    print('x.shape', x.shape)
    print('f.shape', f.shape)
    print('f.min', f.min())

#----------

ax3.plot(t, sigf, 'b-', linewidth=3, **kwargs)

sigw = np.array(sigf)

p0 = peaks[0]
a0 = sigf[p0]

for i,p in enumerate(peaks) :

    #if i>3 : break

    xmin, xmax, dx = -p, sigf.size-p, 1
    x = np.arange(xmin, xmax, 1)
    ai = sigw[p]
    print('peak:%d ai:%.3f'%(i, ai))
    if ai>0 : continue

    f = wavelet(x, scx=0.5*a0/ai) * sigw[p]

    sigs = np.zeros_like(sigf)
    ax3.plot(t, f, 'g-', linewidth=1, **kwargs)

    sigw = sigw+f
    ax3.plot(t, sigw, 'm-', linewidth=1, **kwargs)

#----------

gr.show()

gr.save_fig(fig, prefix='fig-', suffix='-b%04d-%04d-wf-gfilter.png' % (binmin, binmax), **kwargs)

#----------
