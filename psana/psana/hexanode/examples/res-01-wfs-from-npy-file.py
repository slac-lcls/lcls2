#!/usr/bin/env python
#--------------------
"""Plot waveforms from npy array of shape=(5, 2000) loaded from file.
"""

import sys
import numpy as np
import psana.pyalgos.generic.Graphics as gr
from psana.pyalgos.generic.NDArrUtils import print_ndarr

#----------

tname = sys.argv[1] if len(sys.argv) > 1 else '1'
print('%s\nTEST %s' % (50*'_', tname))

ifname = '/reg/g/psdm/detector/data2_test/npy/wavelets-amox27716-r0091.npy' if tname=='1' else\
         '/reg/g/psdm/detector/data2_test/npy/waveforms-amox27716-r0100-e000010.npy'

wf = np.load(ifname)
print('Waveforms loaded from file %s' % ifname)
print_ndarr(wf, 'Array of waveforms')

fig = gr.figure(figsize=(15,15), title='Waveforms')
fig.clear()

naxes = 5
dy = 1./naxes
w, h = 0.87, (dy-0.04)
x0, y0 = 0.07, 0.03
gfmt = ('b-', 'r-', 'g-', 'k-', 'm-', 'y-', 'c-', )
ylab = ('X1', 'X2', 'Y1', 'Y2', 'MCP', 'XX', 'YY', )
ax = [gr.add_axes(fig, axwin=(x0, y0 + i*dy, w, h)) for i in range(naxes)]
zerocross = (382,344,388,336,500)

#----------

for i in range(naxes) :
    ax[i].clear()
    wl = wf[i,:] if tname=='1' else\
         wf[i,8000:16000]
    ax[i].plot(wl, gfmt[0], linewidth=1)
    gr.drawLine(ax[i], ax[i].get_xlim(), (0,0), s=10, linewidth=1, color='k')
    z = zerocross[i]
    print('ch:%d sum-signal: %.1f sum-tail: %.1f' % (i, np.sum(wl[:z]), np.sum(wl[z:])))

gr.show()

#----------
