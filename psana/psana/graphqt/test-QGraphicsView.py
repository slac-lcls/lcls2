#!@PYTHON@
"""
Class :py:class:`test-QGraphicsView` - test for QGraphicsView geometry
==================================================================================

Usage ::

    python graphqt/src/test-QGraphicsView.py

Created on December 12, 2017 by Mikhail Dubrovin
"""
from time import time, sleep

from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QGraphicsView, QApplication, QGraphicsScene

DT_sec = 3
#-----------------------------

class MyQGraphicsView(QGraphicsView) :

   def __init__(self, parent=None) : 
       QGraphicsView.__init__(self, parent)


#   def paintEvent(self, e):
#       print('XXX: paintEvent')#; sleep(DT_sec)
#       QGraphicsView.paintEvent(self, e)
#       print('XXX: After  paintEvent'); sleep(DT_sec)


   def mousePressEvent(self, e):
       self.p0 = pw = e.pos()
       #self.r0 = self.sceneRect() #.center()
       self.rs_center = self.sceneRect().center()
       #self.rs_topleft = self.sceneRect().topLeft()
       self.invscalex = 1./self.transform().m11()
       self.invscaley = 1./self.transform().m22()
       ps = self.mapToScene(pw)
       print('XXX: MyQGraphicsView.mousPressEvent in win: %4d %4d on scene: %.1f %.1f' %\
              (pw.x(), pw.y(), ps.x(), ps.y()))


   def mouseReleseEvent(self, e):
       self.p0 = None


   def mouseMoveEvent(self, e):

       if self.p0 is None : return        
       t0_sec = time()

       dp = e.pos() - self.p0
       #dx = dp.x()*self.invscalex
       #dy = dp.y()*self.invscaley
       dpsc = QtCore.QPointF(dp.x()*self.invscalex, dp.y()*self.invscaley)

       #dpsc = self.mapToScene(dp)
       #print('XXX: dp on scene 1: %8.1f %8.1f time = %.6f'% (dpsc.x(), dpsc.y(), time() - t0_sec))

       rs = self.sceneRect()
       rs.moveCenter(self.rs_center - dpsc)
       #rs.moveTo(self.rs_topleft - dpsc)
       #rs.translate(-dpsc); self.p0 = e.pos()
       self.setSceneRect(rs)

       #print('P3')#; sleep(DT_sec)
       print('time = %.6f' % (time() - t0_sec))


   def moveEvent(self, e):
       print('XXX: MyQGraphicsView.moveEvent topLeft:', self.geometry().topLeft())


   def resizeEvent(self, e):
       print('XXX: MyQGraphicsView.resizeEvent size:', self.geometry().size())
       s = self.scene()
       rs = self.sceneRect()

       mx,my = 100,50 # 0,0
       x, y, w, h = rs.getRect()
       rv = QtCore.QRectF(x-mx, y-my, w+2*mx, h+2*my)

       print('P0')#; sleep(DT_sec)
       self.fitInView(rv, Qt.IgnoreAspectRatio) # Qt.IgnoreAspectRatio KeepAspectRatio KeepAspectRatioByExpanding
       print('P1')#; sleep(DT_sec)
       #self.ensureVisible(rv, xMargin=0, yMargin=0)

#-----------------------------
if __name__ == "__main__" :

    import sys

    app = QApplication(sys.argv)
    rs = QtCore.QRectF(0, 0, 800, 800)
    ro = QtCore.QRectF(-2, -2, 10, 4)
    re = QtCore.QRectF(640-2, 480-2, 4, 4)

    s = QGraphicsScene(rs)
    w = MyQGraphicsView(s)
    w.setGeometry(20, 20, 500, 500)
    #w.setGeometry(20, 20, 800, 800)

    # Invert x,y scales
    sx, sy = -1, -1
    t2 = w.transform().scale(sx, sy)
    #t2 = w.transform().rotate(5)
    w.setTransform(t2)

    print('screenGeometry():', app.desktop().screenGeometry())
    print('geometry():', w.geometry())
    print('scene rect=', s.sceneRect())

    irs = s.addRect(rs, pen=QtGui.QPen(Qt.black, 0, Qt.SolidLine), brush=QtGui.QBrush(Qt.yellow))
    iro = s.addRect(ro, pen=QtGui.QPen(Qt.black, 0, Qt.SolidLine), brush=QtGui.QBrush(Qt.red))
    ire = s.addRect(re, pen=QtGui.QPen(Qt.black, 0, Qt.SolidLine), brush=QtGui.QBrush(Qt.green))

    w.setWindowTitle("My window")
    #w.setContentsMargins(-9,-9,-9,-9)
    w.show() # opens window
    print('A after w.show()')#; sleep(DT_sec)
    app.exec_() # draws graphics
    print('E')
    #s.clear()
    #s.clear()
    s.removeItem(irs)
    s.removeItem(iro)
    s.removeItem(ire)

    del w
    del app

#-----------------------------
