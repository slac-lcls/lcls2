
"""
Class :py:class:`FWHist` 
===========================

Usage ::

    rs = QRectF(0, 0, 1, 1)
    s = QGraphicsScene(rs)
    v = QGraphicsView(s)
    v.setGeometry(20, 20, 600, 400)

    v.fitInView(rs, Qt.IgnoreAspectRatio)

    ruler1 = FWHist(v, 'L')
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

from PyQt5.QtGui import QPen, QBrush, QColor, QPainterPath
from PyQt5.QtCore import Qt, QPointF

class FWHist():

    def __init__(self, view, **kwargs):

        self.view   = view
        self.scene  = view.scene()
        self.rect   = self.scene.sceneRect()

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


    if __name__ == "__main__":

      def add_test(self):

        r = self.rect
        w,h = r.width(), r.height()
        v = self.view
        sv = 1 if v._origin_u else -1
        sh = 1 if v._origin_l else -1

        self.pbl   = r.bottomLeft()
        self.pbr   = r.bottomRight()
        self.ptl   = r.topLeft()
        self.ptr   = r.topRight()
        self.ptc   = (self.ptl + self.ptr)/2
        self.pbc   = (self.pbl + self.pbr)/2
        self.plc   = (self.pbl + self.ptl)/2
        self.prc   = (self.pbr + self.ptr)/2
        self.pcc   = (self.pbc + self.ptc)/2

        if self.path_item is not None: self.scene.removeItem(self.path_item)

        self.path = QPainterPath(self.pbl)
        #self.path.closeSubpath()
        self.path.moveTo(self.pbl)
        self.path.lineTo(self.ptc)
        self.path.lineTo(self.pcc)
        self.path.lineTo(self.prc)
        self.path.lineTo(self.pbl)

        # add path with hist lines to scene
        self.lst_of_items=[]

        self.path_item = self.scene.addPath(self.path, self.pen, self.brush)
        self.path_item.setZValue(self.zvalue)
        self.lst_of_items.append(self.path_item)

        print('test path_item is created')



def test_histogram():
    import psana.pyalgos.generic.NDArrGenerators as ag
    from psana.pyalgos.generic.HBins import HBins
    nbins = 1000
    arr = ag.random_standard((nbins,), mu=50, sigma=10, dtype=ag.np.float64)
    hb = HBins((0,nbins), nbins=nbins)
    hb.set_bin_data(arr, dtype=ag.np.float64)
    return hb


if __name__ == "__main__":

  import os
  import sys

  def test_fwhist(tname):

    from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView
    from PyQt5.QtCore import QRectF

    app = QApplication(sys.argv)
    rs = QRectF(0, 0, 100, 100) # scene rect
    ro = QRectF(-2, -1, 4, 2)   # origin rect for mark only
    s = QGraphicsScene(rs)
    v = QGraphicsView(s)
    v.setGeometry(20, 20, 600, 600)
    v._origin_u = True
    v._origin_l = True

    print('screenGeometry():', app.desktop().screenGeometry())
    print('scene rect=', s.sceneRect())

    v.fitInView(rs, Qt.KeepAspectRatio) # Qt.IgnoreAspectRatio Qt.KeepAspectRatioByExpanding Qt.KeepAspectRatio

    # mark scene and origin by colored rects
    s.addRect(rs, pen=QPen(Qt.black, 0, Qt.SolidLine), brush=QBrush(Qt.yellow))
    s.addRect(ro, pen=QPen(Qt.black, 0, Qt.SolidLine), brush=QBrush(Qt.red))

    if   tname ==  '0': h = FWHist(v, hbins=test_histogram())
    elif tname ==  '1': h = FWHist(v, hbins=None)
    elif tname ==  '2': h = FWHist(v, hbins=test_histogram(), orient='V')
    elif tname ==  '3': h = FWHist(v, hbins=test_histogram(), orient='V', brush=QBrush(Qt.green), color=Qt.blue)
    elif tname ==  '4': h = FWHist(v, hbins=test_histogram(), orient='H', brush=QBrush(Qt.green), pen=QPen(Qt.magenta, 2, Qt.SolidLine))
    else:
        print('test %s is not implemented' % tname)
        return

    v.setWindowTitle("FWHist test")
    v.setContentsMargins(0,0,0,0)
    v.show()
    app.exec_()

    # prevent crash on exit...
    del h, v, s, app


if __name__ == "__main__":

    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_fwhist(tname)
    #sys.exit('End of Test %s' % tname)

# EOF
