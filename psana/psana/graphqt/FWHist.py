"""
Class :py:class:`FWHist` adds a ruller to one of the sides of the scene rectangle
==================================================================================

Usage ::

    rs = QRectF(0, 0, 1, 1)
    s = QGraphicsScene(rs)
    v = QGraphicsView(s)
    v.setGeometry(20, 20, 600, 400)

    v.fitInView(rs, Qt.IgnoreAspectRatio) # Qt.IgnoreAspectRatio Qt.KeepAspectRatioByExpanding Qt.KeepAspectRatio

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

from PyQt5.QtWidgets import QGraphicsItem
from PyQt5.QtGui import QFont, QPen, QBrush, QColor, QPainterPath
from PyQt5.QtCore import Qt, QPointF, QPoint, QRectF

#----

class FWHist():

    def __init__(self, view, side='U', **kwargs):

        self.view   = view
        self.scene  = self.view.scene()
        self.axside = side.upper()
        self.horiz  = self.axside in ('D','U')

        self.font        = kwargs.get('font',  QFont('Courier', 12, QFont.Normal))
        self.pen         = kwargs.get('pen',   QPen(Qt.black, 5, Qt.SolidLine))
        self.brush       = kwargs.get('brush', QBrush(Qt.blue))
        self.color       = kwargs.get('color', QColor(Qt.red))
        self.tick_fr     = kwargs.get('tick_fr',  0.15)
        self.txtoff_vfr  = kwargs.get('txtoff_vfr',  0)
        self.txtoff_hfr  = kwargs.get('txtoff_hfr',  0)
        self.size_inches = kwargs.get('size_inches', 3)
        self.zvalue      = kwargs.get('zvalue',     10)
        self.fmt         = kwargs.get('fmt',      '%g')

        #QPainterPath.__init__(self)
   
        self.pen.setCosmetic(True)
        self.pen.setColor(self.color)

        self.path = None
        self.path_item = None
        self.lst_of_items = []

        r = self.rect=self.scene.sceneRect()
        vmin = r.x() if self.horiz else r.y()
        vmax = r.x()+r.width() if self.horiz else r.y()+r.height()

        self.set_pars()
        self.add()


    def set_pars(self):
        r = self.rect
        w,h = r.width(), r.height()
        v = self.view
        sv = 1 if v._origin_u else -1
        sh = 1 if v._origin_l else -1
        self.dtxt0 = QPointF(0, 0)

        #print('Scales sv, sh=', sv, sh)

        if True:
            self.pbl   = r.bottomLeft()
            self.pbr   = r.bottomRight()
            self.ptl   = r.topLeft()
            self.ptr   = r.topRight()
            self.ptc   = (self.ptl + self.ptr)/2
            self.pbc   = (self.pbl + self.pbr)/2
            self.plc   = (self.pbl + self.ptl)/2
            self.prc   = (self.pbr + self.ptr)/2
            self.pcc   = (self.pbc + self.ptc)/2

    def add(self):
        # add ruller to the path of the scene
        if self.path_item is not None: self.scene.removeItem(self.path_item)

        self.path = QPainterPath(self.pbl)
        #self.path.closeSubpath()
        self.path.moveTo(self.pbl)
        self.path.lineTo(self.ptc)
        self.path.lineTo(self.pcc)
        self.path.lineTo(self.prc)
        self.path.lineTo(self.pbl)

        
        #self.path.moveTo(self.pbl)
        #self.path.lineTo(self.ptc)
        #self.path.lineTo(self.pbr)
        #self.path.lineTo(self.plc)
        #self.path.lineTo(self.ptr)
        #self.path.lineTo(self.pbc)
        #self.path.lineTo(self.ptl)
        #self.path.lineTo(self.prc)
        #self.path.lineTo(self.pbl)

        # add path with hist lines to scene
        self.lst_of_items=[]

        self.path_item = self.scene.addPath(self.path, self.pen, self.brush)
        self.path_item.setZValue(self.zvalue)
        self.lst_of_items.append(self.path_item)

        print('path_item is created')


    def remove(self):
        for item in self.lst_of_items:
            self.scene.removeItem(item)
        self.lst_of_items=[]

        #self.scene.removeItem(self.path_item)
        #self.scene.destroyItemGroup(self.item_group)


    def update(self):
        self.remove()
        self.set_pars()
        self.add()


    def __del__(self):
        self.remove()

#----

if __name__ == "__main__":

    import sys
    from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

    app = QApplication(sys.argv)
    rs = QRectF(0, 0, 100, 100)
    ro = QRectF(-1, -1, 3, 2)
    s = QGraphicsScene(rs)
    v = QGraphicsView(s)
    v.setGeometry(20, 20, 600, 600)
    v._origin_u = True
    v._origin_l = True

    print('screenGeometry():', app.desktop().screenGeometry())
    print('scene rect=', s.sceneRect())

    v.fitInView(rs, Qt.KeepAspectRatio) # Qt.IgnoreAspectRatio Qt.KeepAspectRatioByExpanding Qt.KeepAspectRatio

    s.addRect(rs, pen=QPen(Qt.black, 0, Qt.SolidLine), brush=QBrush(Qt.yellow))
    s.addRect(ro, pen=QPen(Qt.black, 0, Qt.SolidLine), brush=QBrush(Qt.blue))

    hist1 = FWHist(v, 'D')

    v.setWindowTitle("My window")
    v.setContentsMargins(0,0,0,0)
    v.show()
    app.exec_()

    hist1.remove()

    del app

#----
