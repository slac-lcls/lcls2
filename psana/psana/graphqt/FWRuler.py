"""
Class :py:class:`FWRuler` adds a ruller to one of the sides of the scene rectangle
==================================================================================

Usage ::

    rs = QRectF(0, 0, 1, 1)
    s = QGraphicsScene(rs)
    v = QGraphicsView(s)
    v.setGeometry(20, 20, 600, 400)

    v.fitInView(rs, Qt.IgnoreAspectRatio) # Qt.IgnoreAspectRatio Qt.KeepAspectRatioByExpanding Qt.KeepAspectRatio

    ruler1 = FWRuler(v, 'L')
    ruler2 = FWRuler(v, 'D')
    ruler3 = FWRuler(v, 'U')
    ruler4 = FWRuler(v, 'R')

    ruler1.remove()
    
    ruler4.remove()

See:
    - :class:`FWView`
    - :class:`FWViewImage`
    - :class:`QWSpectrum`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2017-12-12 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-20
"""
#------------------------------
from PyQt5.QtWidgets import QGraphicsItem
from PyQt5.QtGui import QFont, QPen, QBrush, QColor, QPainterPath
from PyQt5.QtCore import Qt, QPointF, QPoint, QRectF

from psana.graphqt.AxisLabeling import best_label_locs
#------------------------------

class FWRuler() :
    def __init__(self, view, side='U', **kwargs) :

        self.view   = view
        self.scene  = self.view.scene()
        self.axside = side.upper()
        self.horiz  = self.axside in ('D','U')

        self.font        = kwargs.get('font',  QFont('Courier', 12, QFont.Normal))
        self.pen         = kwargs.get('pen',   QPen(Qt.black, 1, Qt.SolidLine))
        self.brush       = kwargs.get('brush', QBrush(Qt.red))
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
        self.labels = best_label_locs(vmin, vmax, self.size_inches, density=1, steps=None)

        #print('labels', self.labels)

        self.set_pars()
        self.add()


    def set_pars(self) :
        r = self.rect
        w,h = r.width(), r.height()
        v = self.view



        sv = 1 if v._origin_u else -1
        sh = 1 if v._origin_l else -1
        self.dtxt0 = QPointF(0, 0)

        #print('Scales sv, sh=', sv, sh)

        if self.axside == 'D' :
            if sv > 0 :
              self.p1   = r.bottomLeft()
              self.p2   = r.bottomRight()
            else :
              self.p1   = r.topLeft()
              self.p2   = r.topRight()
            self.dt1  = QPointF(0, -sv*self.tick_fr * h)
            self.dtxt = QPointF(-0.5, -1)
            self.vort = self.p2.y()

        elif self.axside == 'U' :
            if sv > 0 :
              self.p1   = r.topLeft()
              self.p2   = r.topRight()
            else :
              self.p1   = r.bottomLeft()
              self.p2   = r.bottomRight()
            self.dt1  = QPointF(0, sv*self.tick_fr * h)
            self.dtxt = QPointF(-0.5, 0.1)
            self.vort = self.p1.y()
 
        elif self.axside == 'L' :
            if sh > 0 :
              self.p1   = r.topLeft()
              self.p2   = r.bottomLeft()
            else :
              self.p1   = r.topRight()
              self.p2   = r.bottomRight()
            self.dt1  = QPointF(sh*self.tick_fr * w, 0)
            self.dtxt = QPointF(0, -0.5)
            self.dtxt0 = QPointF(6, 0)
            self.vort = self.p1.x()

        elif self.axside == 'R' : 
            if sh > 0 :
              self.p1   = r.topRight()
              self.p2   = r.bottomRight()
            else :
              self.p1   = r.topLeft()
              self.p2   = r.bottomLeft()
            self.dt1  = QPointF(-sh*self.tick_fr * w, 0)
            self.dtxt = QPointF(-1, -0.5)            
            self.dtxt0 = QPointF(-3, 0)
            self.vort = self.p2.x()

            #print('p1,p2, dt1, dtxt, vort', self.p1, self.p2, self.dt1, self.dtxt, self.vort)

        else :
            print('ERROR: non-defined axis side "%s". Use L, R, U, or D.' % str(self.axside))


    def add(self) :
        # add ruller to the path of the scene
        if self.path_item is not None : self.scene.removeItem(self.path_item)

        self.path = QPainterPath(self.p1)
        #self.path.closeSubpath()
        self.path.moveTo(self.p1)
        self.path.lineTo(self.p2)

        #print('self.p1', self.p1)
        #print('self.p2', self.p2)

        for v in self.labels :
            pv = QPointF(v, self.vort) if self.horiz else QPointF(self.vort, v)
            self.path.moveTo(pv)
            self.path.lineTo(pv+self.dt1)

        # add path with ruler lines to scene

        self.lst_of_items=[]

        self.path_item = self.scene.addPath(self.path, self.pen, self.brush)
        self.path_item.setZValue(self.zvalue)
        self.lst_of_items.append(self.path_item)

        #print('path_item is created')

        r = self.rect
        w,h = r.width(), r.height()
        # add labels to scene 
        for v in self.labels :
            pv = QPointF(v, self.vort) if self.horiz else QPointF(self.vort, v)
            vstr = self.fmt%v
            txtitem = self.scene.addText(vstr, self.font)
            txtitem.setDefaultTextColor(self.color)

            pp = self.view.mapFromScene(pv)
            wtxt = txtitem.boundingRect().width()
            htxt = txtitem.boundingRect().height()

            #print('XXX: str(%s), bbox=' % (vstr), '  bbox=', txtitem.boundingRect(), '  pp=', pp, wtxt, htxt)

            #pt = pv + self.dtxt + QPointF(self.hoff*len(vstr),0)\
            #     + QPointF(self.txtoff_hfr*h, self.txtoff_vfr*h)

            of0 = self.dtxt0
            off = self.dtxt
            pt = self.view.mapToScene(pp + QPoint(of0.x() + off.x()*wtxt, of0.y() + off.y()*htxt))

            txtitem.moveBy(pt.x(), pt.y())

            txtitem.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            txtitem.setZValue(self.zvalue)


            self.lst_of_items.append(txtitem)       

        #self.item_group = self.scene.createItemGroup(self.lst_of_items)

    def remove(self) :
        for item in self.lst_of_items :
            self.scene.removeItem(item)
        self.lst_of_items=[]

        #self.scene.removeItem(self.path_item)
        #self.scene.destroyItemGroup(self.item_group)


    def update(self) :
        self.remove()
        self.set_pars()
        self.add()


    def __del__(self) :
        self.remove()

#-----------------------------
if __name__ == "__main__" :

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
    s.addRect(ro, pen=QPen(Qt.black, 0, Qt.SolidLine), brush=QBrush(Qt.red))

    ruler1 = FWRuler(v, 'L')
    ruler2 = FWRuler(v, 'D')
    ruler3 = FWRuler(v, 'U')
    ruler4 = FWRuler(v, 'R')

    v.setWindowTitle("My window")
    v.setContentsMargins(-9,-9,-9,-9)
    v.show()
    app.exec_()

    #s.clear()

    ruler1.remove()
    ruler2.remove()
    ruler3.remove()
    ruler4.remove()

    del app

#-----------------------------
