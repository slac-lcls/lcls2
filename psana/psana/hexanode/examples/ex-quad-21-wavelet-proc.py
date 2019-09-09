#!/usr/bin/env python
#--------------------
"""Averages the shape of a single pulse per channel.
"""

import sys
from time import time
import psana
import numpy as np
from Detector.WFDetector import WFDetector

import pyimgalgos.Graphics       as gr
import pyimgalgos.GlobalGraphics as gg
#import pyimgalgos.GlobalUtils    as gu

#----------

IFNAME = '/reg/g/psdm/detector/data_test/npy/wavelets-amox27716-r0091.npy'

fig = gr.figure(figsize=(15,15), title='Wavelets')
fig.clear()
#gr.move_fig(fig, 200, 100)
#fig.canvas.manager.window.geometry('+200+100')

NAXES = 5
dy = 1./NAXES

w, h = 0.87, (dy-0.04)
x0, y0 = 0.07, 0.03

gfmt = ('b-', 'r-', 'g-', 'k-', 'm-', 'y-', 'c-', )
ylab = ('X1', 'X2', 'Y1', 'Y2', 'MCP', 'XX', 'YY', )
ax = [gr.add_axes(fig, axwin=(x0, y0 + i*dy, w, h)) for i in range(NAXES)]

#----------

w = np.load(IFNAME)
print 'Wavelets liaded from file %s' % IFNAME

#----------
zerocr = [382,344,388,336,500]

for i in range(NAXES) :
    ax[i].clear()
    wl = w[i,:]
    ax[i].plot(w[i,:], gfmt[0], linewidth=1)
    gg.drawLine(ax[i], ax[i].get_xlim(), (0,0), s=10, linewidth=1, color='k')
    z = zerocr[i]
    print 'ch:%d sum-signal: %.1f sum-tail: %.1f' % (i, np.sum(wl[:z]), np.sum(wl[z:]))

gr.show()

sys.exit(0)

#----------
