#!@PYTHON@
"""
Class :py:class:`GURuler` adds a ruller to one of the sides of the scene rectangle
==================================================================================

Usage ::

    rv = QtCore.QRectF(-0.2, -0.2, 1.4, 1.4)
    rs = QtCore.QRectF(0, 0, 1, 1)

    s = QtGui.QGraphicsScene(rs)
    v = QtGui.QGraphicsView(s)

    v.setGeometry(20, 20, 600, 400)

    v.fitInView(rv, 0) # Qt.IgnoreAspectRatio Qt.KeepAspectRatioByExpanding Qt.KeepAspectRatio

    ruler1 = GURuler(s, GURuler.VL)
    ruler2 = GURuler(s, GURuler.HD)
    ruler3 = GURuler(s, GURuler.HU)
    ruler4 = GURuler(s, GURuler.VR)

    ruler3.remove()
    ruler4.remove()

Created on June 12, 2016 by Mikhail Dubrovin
"""

#import os
#import math
#from graphqt.NDArrGenerators import random_array_xffffffff
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, QPointF

#from pyapps.graphqt.AxisLabeling import best_label_locs
from AxisLabeling import best_label_locs

class GURuler() :
    
    HD=0; HU=1; VL=2; VR=3
    orient_inds  = (HD, HU, VL, VR)
    orient_names = ('HD', 'HU', 'VL', 'VR')

    def __init__( self
                , scene
                , orient=HD
                , rect=None
                , txtoff_hfr=0
                , txtoff_vfr=0
                , tick_fr=0.015
                , color=QtGui.QColor(Qt.red)
                , pen=QtGui.QPen(Qt.black, 1, Qt.SolidLine)
                , font=QtGui.QFont('Courier', 12, QtGui.QFont.Normal)
                , fmt='%g'
                , size_inches=3
                , zvalue=10
                ) :

        #QtGui.QPainterPath.__init__(self)
        self.scene=scene
        self.orient=orient
        self.rect=rect if rect is not None else self.scene.sceneRect()
        self.txtoff_hfr=txtoff_hfr
        self.txtoff_vfr=txtoff_vfr
        self.tick_fr=tick_fr
        self.color=color
        self.pen=pen
        self.font=font
        self.fmt=fmt
        self.horiz = (orient==self.HD) or (orient==self.HU)
        self.size_inches=size_inches
        self.zvalue = zvalue
        self.brush = QtGui.QBrush(Qt.red)

        self.pen.setCosmetic(True)
        self.pen.setColor(color)

        self.path = None
        self.path_item = None

        r = self.rect

        vmin = r.x() if self.horiz else r.y()
        vmax = r.x()+r.width() if self.horiz else r.y()+r.height()
        self.labels = best_label_locs(vmin, vmax, self.size_inches, density=1, steps=None)

        #print 'labels', self.labels

        self.textitems=[]
        self.set_pars()
        self.add()


    def set_pars(self) :

        r = self.rect
        x,y,w,h = r.x(), r.y(), r.width(), r.height()

        self.hoff = -0.01*w # label offset for each character 

        if self.orient == self.HU :
            self.p1   = r.topLeft()
            self.p2   = r.topRight()
            self.dt1  = QPointF(0, -self.tick_fr * h)
            self.dtxt = QPointF(-0.01*w, -0.10*h)
            self.vort = y
 
        elif self.orient == self.VL : 
            self.p1   = r.topLeft()
            self.p2   = r.bottomLeft()
            self.dt1  = QPointF(-self.tick_fr * w, 0)
            self.dtxt = QPointF(-0.03*w,-0.03*h)
            self.vort = x
            self.hoff = -0.02*w

        elif self.orient == self.VR : 
            self.p1   = r.topRight()
            self.p2   = r.bottomRight()
            self.dt1  = QPointF(self.tick_fr * w, 0)
            self.dtxt = QPointF(0.02*w,-0.03*h)
            self.vort = x + w
            self.hoff = 0*w

        elif self.orient == self.HD :
            self.p1   = r.bottomLeft()
            self.p2   = r.bottomRight()
            self.dt1  = QPointF(0, self.tick_fr * h)
            self.dtxt = QPointF(-0.01*w, 0.015*h)
            self.vort = y + h
            #print 'p1,p2, dt1, dtxt, vort', self.p1, self.p2, self.dt1, self.dtxt, self.vort

        else :
            print 'ERROR: non-defined axis orientation'


    def add(self) :
        # add ruller to the path of the scene
        if self.path_item is not None : self.scene.removeItem(self.path_item)

        self.path = QtGui.QPainterPath(self.p1)
        #self.path.closeSubpath()
        self.path.moveTo(self.p1)
        self.path.lineTo(self.p2)

        #print 'self.p1', self.p1
        #print 'self.p2', self.p2

        for v in self.labels :
            pv = QPointF(v, self.vort) if self.horiz else QPointF(self.vort, v)
            self.path.moveTo(pv)
            self.path.lineTo(pv+self.dt1)

        # add path with ruler lines to scene

        self.lst_of_items=[]

        self.path_item = self.scene.addPath(self.path, self.pen, self.brush)
        self.path_item.setZValue(self.zvalue)
        self.lst_of_items.append(self.path_item)

        #print 'path_item is created'

        r = self.rect
        w,h = r.width(), r.height()
        # add labels to scene 
        for v in self.labels :
            pv = QPointF(v, self.vort) if self.horiz else QPointF(self.vort, v)
            vstr = self.fmt%v
            pt = pv + self.dtxt + QPointF(self.hoff*len(vstr),0)\
                 + QPointF(self.txtoff_hfr*h, self.txtoff_vfr*h)
            txtitem = self.scene.addText(vstr, self.font)
            txtitem.setDefaultTextColor(self.color)
            txtitem.moveBy(pt.x(), pt.y())
            txtitem.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, True)
            txtitem.setZValue(self.zvalue)

            self.lst_of_items.append(txtitem)       

        #self.item_group = self.scene.createItemGroup(self.lst_of_items)

    def remove(self) :
        #self.scene.removeItem(self.path_item)
        for item in self.lst_of_items :
            self.scene.removeItem(item)
        self.lst_of_items=[]

        #self.scene.destroyItemGroup(self.item_group)


    def update(self) :
        self.remove()
        self.set_pars()
        self.add()


    def __del__(self) :
        #print 'in __del__'
        self.remove()

#-----------------------------
if __name__ == "__main__" :

    import sys

    app = QtGui.QApplication(sys.argv)
    print 'screenGeometry():', app.desktop().screenGeometry()

    rv = QtCore.QRectF(-0.3, -0.3, 1.6, 1.6)
    rs = QtCore.QRectF(0, 0, 1, 1)
    ra = QtCore.QRectF(0.1, 0.1, 0.8, 0.8)

    s = QtGui.QGraphicsScene(rs)
    #s.setSceneRect(r)
    print 'scene rect=', s.sceneRect()

    v = QtGui.QGraphicsView(s)
    v.setGeometry(20, 20, 600, 400)

    v.fitInView(rv, 0) # Qt.IgnoreAspectRatio Qt.KeepAspectRatioByExpanding Qt.KeepAspectRatio
    #v.ensureVisible(rv, xMargin=50, yMargin=50)
    #v.scale(5, 5)

    ruler1 = GURuler(s, GURuler.VL, rect=ra)
    ruler2 = GURuler(s, GURuler.HD, rect=ra)
    ruler3 = GURuler(s, GURuler.HU)
    ruler4 = GURuler(s, GURuler.VR)

    #ruler2.remove()

    w=v
    w.setWindowTitle("My window")
    #w.setContentsMargins(-9,-9,-9,-9)
    w.show()
    app.exec_()
#-----------------------------
