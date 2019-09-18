#!/usr/bin/env python
#------------------------------

import sys
import psana
import numpy as np
from Detector.WFDetector import WFDetector

import pyimgalgos.Graphics       as gr
import pyimgalgos.GlobalGraphics as gg
import pyimgalgos.GlobalUtils    as gu



from pypsalg import find_edges


#------------------------------
BASE = 0.
THR = -0.04
CFR = 0.9
DEADTIME = 5.0
LEADINGEDGE = True # False # True
#------------------------------

def draw_times(ax, wf, wt) :
    #wf -= wf[0:1000].mean()
    edges = find_edges(wf, BASE, THR, CFR, DEADTIME, LEADINGEDGE)
    # pairs of (amplitude,sampleNumber)
    print 'MCP edges'
    print edges

    for (amp,ind) in edges :
        x0 = wt[int(ind)]
        xarr = (x0,x0)
        yarr = (amp,-amp)
        gg.drawLine(ax, xarr, yarr, s=10, linewidth=1, color='k')

#------------------------------

#dsname = 'exp=xpptut15:run=280'
#src1 = 'AmoEndstation.0:Acqiris.1' # 'ACQ1'
#src2 = 'AmoEndstation.0:Acqiris.2' # 'ACQ2'

# event_keys -d exp=amod3814:run=85
# event_keys -d exp=xpptut15:run=390

#dsname = 'exp=amod3814:run=85'
dsname = 'exp=xpptut15:run=390'
src1 = 'AmoETOF.0:Acqiris.0'
src2 = 'AmoITOF.0:Acqiris.0'

print 'Example for\n  dataset: %s\n  source1 : %s\n  source2 : %s' % (dsname, src1, src2)

#opts = {'psana.calib-dir':'./calib',}
#psana.setOptions(opts)
#psana.setOption('psana.calib-dir', './calib')
#psana.setOption('psana.calib-dir', './empty/calib')

ds  = psana.DataSource(dsname)
env = ds.env()
#nrun = evt.run()
#evt = ds.events().next()
#for key in evt.keys() : print key

det2 = WFDetector(src2, env, pbits=1022)
det1 = WFDetector(src1, env, pbits=1022)
det1.print_attributes()

#------------------------------

fig = gr.figure(figsize=(15,15), title='Image')
#gr.move_fig(fig, 200, 100)
#fig.canvas.manager.window.geometry('+200+100')

naxes = 7
dy = 1./naxes

lw = 1
w = 0.87
h = dy - 0.04
x0, y0 = 0.07, 0.03

#ch = (0,1,2,3,4,5)
ch = (6,7,8,9,10,11,0)
gfmt = ('b-', 'r-', 'g-', 'k-', 'm-', 'y-', 'c-', )
ylab = ('Y1', 'Y2', 'Z1', 'Z2', 'X1', 'X2', 'MCP', )
ax = [gr.add_axes(fig, axwin=(x0, y0 + i*dy, w, h)) for i in range(naxes)]

#------------------------------
wf,wt = None, None

for i,evt in enumerate(ds.events()) :
    #if i< 140 : continue
    if i>4 : break
    print 50*'_', '\n Event # %d' % i
    gr.set_win_title(fig, titwin='Event: %d' % i)

    print 'Acqiris.1:'
    result = det1.raw(evt)
    if result is None : continue
    wf,wt = result

    #gu.print_ndarr(wf, 'acqiris waveform')
    #gu.print_ndarr(wt, 'acqiris wavetime')

    print 'Acqiris.2:'
    wf2,wt2 = det2.raw(evt)

    bbeg=0     # 0
    bend=20000 # -1

    for i in range(naxes) :
        ax[i].clear()
        #ax[i].set_xlim((0,0.00001))
        ax[i].set_xlim((0.0000025,0.0000045))
        ax[i].set_ylabel(ylab[i], fontsize=14)
        if i==6 : break

        wfsel = wf[ch[i],bbeg:bend]
        wtsel = wt[ch[i],bbeg:bend]

        wfsel -= wfsel[0:1000].mean()

        ax[i].plot(wtsel, wfsel, gfmt[i], linewidth=lw)
        draw_times(ax[i], wfsel, wtsel)
        gg.drawLine(ax[i], ax[i].get_xlim(), (THR,THR), s=10, linewidth=1, color='k')

    wf2sel = wf2[ch[i],bbeg:bend]
    wt2sel = wt2[ch[i],bbeg:bend]

    ax[i].plot(wt2sel, wf2sel, gfmt[i], linewidth=lw)
    draw_times(ax[i], wf2sel, wt2sel)

    gr.draw_fig(fig)
    gr.show(mode='non-hold')

gr.show()

#ch=0
#fig, ax = gg.plotGraph(wt[ch,:-1], wf[ch,:-1], figsize=(15,5))
#gg.show()

#------------------------------

sys.exit(0)

#------------------------------
