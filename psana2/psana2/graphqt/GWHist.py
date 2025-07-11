
"""
Class :py:class:`GWHist`
========================

Usage ::
    from psana2.graphqt.GWHist import *

    rs = QRectF(0, 0, 1, 1)
    s = QGraphicsScene(rs)
    v = QGraphicsView(s)
    v.setGeometry(20, 20, 600, 400)

    v.fitInView(rs, Qt.IgnoreAspectRatio)

    ruler1 = GWHist(v, 'L')
    hist1.remove()

See:
    - :class:`FWView`
    - :class:`FWViewImage`
    - :class:`QWSpectrum`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2020-11-02 by Mikhail Dubrovin
"""

import sip
from PyQt5.QtGui import QPen, QBrush, QColor, QPainterPath
from PyQt5.QtCore import Qt, QPointF
import numpy as np

class GWHist():

    def __init__(self, view, **kwargs):

        self.view   = view
        self.scene  = view.scene()
        #self.rect   = self.scene.sceneRect()

        color       = kwargs.get('color', QColor(Qt.black)) # color for default pen (histogram sline)
        self.pen_def = QPen(color, 0, Qt.SolidLine)
        self.hbins  = kwargs.get('hbins', None)             # histogram container HBins
        self.orient = kwargs.get('orient','H')              # histogram orientation H or V
        self.pen    = kwargs.get('pen',   self.pen_def)     # histogram line pen
        self.brush  = kwargs.get('brush', QBrush())         # histogram filling brush
        self.zvalue = kwargs.get('zvalue',10)               # z value for heirarchial visibility

        self.pen.setCosmetic(self.pen == self.pen_def)

        self.path = None
        self.path_item = None
        self.lst_of_items = []

        if self.hbins is None: self.add_test()
        else:                  self.add_hist()


    def remove(self):
        if not sip.isdeleted(self.scene):
          for item in self.lst_of_items:
            self.scene.removeItem(item)
        self.lst_of_items=[]
        #self.scene.removeItem(self.path_item)
        #self.scene.destroyItemGroup(self.item_group)


    def update(self, hbins):
        self.remove()
        self.hbins = hbins
        self.add_hist()


    def __del__(self):
        self.remove()


    def add_hist(self):

        if self.path_item is not None: self.scene.removeItem(self.path_item)

        edges = self.hbins.binedges() # n+1
        values = self.hbins.bin_data() #dtype=np.float64)

        self.path = QPainterPath() #QPointF(edges[0], 0))

        #self.path.closeSubpath()
        if self.orient == 'V':
          self.path.moveTo(QPointF(0, edges[0]))
          for el, er, v in zip(edges[:-1], edges[1:], values):
            self.path.lineTo(QPointF(v, el))
            self.path.lineTo(QPointF(v, er))
          self.path.lineTo(QPointF(0, edges[-1]))

        else:
          self.path.moveTo(QPointF(edges[0], 0))
          for el, er, v in zip(edges[:-1], edges[1:], values):
            self.path.lineTo(QPointF(el,v))
            self.path.lineTo(QPointF(er,v))
          self.path.lineTo(QPointF(edges[-1], 0.))

        # add path with hist lines to scene
        self.lst_of_items=[]
        self.path_item = self.scene.addPath(self.path, self.pen, self.brush)
        self.path_item.setZValue(self.zvalue)
        self.lst_of_items.append(self.path_item)

        #print('path_item is created')

#def bifurgauss(x, *p):
#    """Bifurcated Gaussian - assumes ydata = f(xdata, *params) + eps."""
#    a0, x0, s1, s2 = p
#    sig = np.select((x<x0,), (s1,), default=s2)
#    return a0 * np.exp(-0.5*((x-x0)/sig)**2)

def gauss(x, *p):
    a0, x0, s = p
    return a0 * np.exp(-0.5*((x-x0)/s)**2)


def test_histogram():
import psana2.pyalgos.generic.NDArrGenerators as ag
    from psana2.pyalgos.generic.HBins import HBins
    xmin, xmax = 100, 1100
    nbins = 1000
    a0, x0, s = 80, 400, 40
    #x = np.arange(-5,5,.01)
    x = np.linspace(xmin, xmax, num=nbins, endpoint=True)
    f = ag.random_standard((nbins,), mu=50, sigma=10, dtype=ag.np.float64)
    f += gauss(x, a0, x0, s) + gauss(x, a0/3, x0+50, s+50)
    f[10]=10000
    f[11]=5000
    hb = HBins((xmin, xmax), nbins=nbins)
    hb.set_bin_data(f, dtype=ag.np.float64)
    return hb

# EOF
