#!/usr/bin/env python
#--------------------
"""
https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.signal.cwt.html
"""

import numpy as np
from time import time
from scipy import signal
import psana.pyalgos.generic.Graphics as gr

#t = np.linspace(-1, 1, 200, endpoint=False)
#sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)

ifname = '/reg/g/psdm/detector/data2_test/npy/waveforms-amox27716-r0100-e000010.npy'
wf = np.load(ifname)
print('Waveforms loaded from file %s' % ifname)
print('Array shape: %s' % str(wf.shape))

binmin, binmax = 8000, 16000
sig = wf[3,binmin:binmax]

kwargs = {}
fig = gr.figure(figsize=(12,10), dpi=80, facecolor='w', edgecolor='w', frameon=True, **kwargs)
fig.canvas.set_window_title('Waveforms', **kwargs)
fig.clear()

w = 0.87
x0, y0 = 0.07, 0.05
ax0 = fig.add_axes((x0, y0 + 0,   w, 0.2), **kwargs)
ax1 = fig.add_axes((x0, y0 + 0.3, w, 0.3), **kwargs)
ax2 = fig.add_axes((x0, y0 + 0.7, w, 0.2), **kwargs)

#ax0.clear()


ax2.set_xlim(left=binmin, right=binmax)
ax0.set_xlim(left=binmin, right=binmax)
ax0.set_xlabel('sample/time')
ax0.set_ylabel('intensity')
ax1.set_ylabel('frequency layer')
ax2.set_ylabel('f-projection')

ax0.grid(True)

nlevels = 30
level_beg = 5

widths = np.arange(1, nlevels)
t0_sec = time()
cwtmatr = signal.cwt(sig, signal.ricker, widths)
print('signal.cwt consumed time %.6f sec'%(time()-t0_sec))
print('cwtmatr.shape', cwtmatr.shape)

cwtplot = cwtmatr[level_beg:nlevels-1,]

ax1.imshow(cwtplot, extent=[binmin, binmax, nlevels-1, level_beg], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

t = np.arange(binmin, binmax, 1)
ax0.plot(t, sig, 'b-', linewidth=1, **kwargs)

prj = np.sum(cwtplot, axis=0)
ax2.plot(t, prj, 'r-', linewidth=1, **kwargs)

gr.show()

gr.save_fig(fig, prefix='fig-', suffix='-wf-cwt-decomp.png', **kwargs)
