#!/usr/bin/env python
#--------------------
"""Plot waveforms from npy array of shape=(5, 2000) loaded from file.
"""
import sys
import numpy as np

import matplotlib
import matplotlib.pyplot  as plt
import matplotlib.lines   as lines
import matplotlib.patches as patches

#----------

def drawLine(axes, xarr, yarr, s=10, linewidth=1, color='w', **kwargs) : 
    line = lines.Line2D(xarr, yarr, linewidth=linewidth, color=color, **kwargs)   
    axes.add_artist(line)

def drawRectangle(axes, xy, width, height, linewidth=1, color='w', **kwargs) :
    rect = patches.Rectangle(xy, width, height, linewidth=linewidth, color=color, **kwargs)
    axes.add_artist(rect)

def drawCircle(axes, xy0, radius, linewidth=1, color='w', fill=False, **kwargs) : 
    circ = patches.Circle(xy0, radius=radius, linewidth=linewidth, color=color, fill=fill, **kwargs)
    axes.add_artist(circ)

#----------

#ifname = '/reg/g/psdm/detector/data2_test/npy/wavelets-amox27716-r0091.npy'
ifname = '/reg/g/psdm/detector/data2_test/npy/waveforms-amox27716-r0100-e000010.npy'
wf = np.load(ifname)
print('Waveforms loaded from file %s' % ifname)
print('Array shape: %s' % str(wf.shape))

kwargs = {}

fig = plt.figure(figsize=(12,10), dpi=80, facecolor='w', edgecolor='w', frameon=True, **kwargs)
fig.canvas.set_window_title('Waveforms', **kwargs)

fig.clear()

naxes = 5
dy = 1./naxes
w, h = 0.87, (dy-0.04)
x0, y0 = 0.07, 0.03
gfmt = ('b-', 'r-', 'g-', 'k-', 'm-', 'y-', 'c-', )
ylab = ('X1', 'X2', 'Y1', 'Y2', 'MCP', 'XX', 'YY', )
ax = [fig.add_axes((x0, y0 + i*dy, w, h), **kwargs) for i in range(naxes)]


zerocross = (382,344,388,336,500)

#----------

if False :
  for i in range(naxes) :
    ax[i].clear()
    #wl = wf[i,:]
    wl = wf[i,8000:16000]
    ax[i].plot(wl, gfmt[0], linewidth=1)
    drawLine(ax[i], ax[i].get_xlim(), (0,0), s=10, linewidth=1, color='k')
    z = zerocross[i]
    print('ch:%d sum-signal: %.1f sum-tail: %.1f' % (i, np.sum(wl[:z]), np.sum(wl[z:])))

#----------

if True :
    for i in range(naxes) :
        ax[i].clear()
 
    iax = 0
    iwf = 3
    w1 = wf[iwf,8000:16000]

    ax[iax].plot(w1, gfmt[iax], linewidth=1)
    drawLine(ax[iax], ax[iax].get_xlim(), (0,0), s=10, linewidth=1, color='k')

plt.show()

#======================
sys.exit('ENO OF TEST')
#======================
