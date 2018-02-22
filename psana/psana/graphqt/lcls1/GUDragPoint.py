#!@PYTHON@
"""
Class :py:class:`GUDragPoint` - for draggable shape item
========================================================

Created on 2016-10-11 by Mikhail Dubrovin
"""
#-----------------------------

#import os
#import math
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QGraphicsPathItem
from graphqt.GUDragBase import *

#-----------------------------

class GUDragPoint(QGraphicsPathItem, GUDragBase) :
                # QPointF, QGraphicsItem, QGraphicsScene
    def __init__(self, point=QtCore.QPointF(0,0), parent=None, scene=None,\
                 brush=QtGui.QBrush(Qt.white, Qt.SolidPattern),\
                 pen=QtGui.QPen(Qt.black, 2, Qt.SolidLine),\
                 orient='v', rsize=7) :

        self._lst_circle = None

        path = self.pathForPointV(point, scene, rsize) if orient=='v' else\
               self.pathForPointH(point, scene, rsize) if orient=='h' else\
               self.pathForPointR(point, scene, rsize)

        QGraphicsPathItem.__init__(self, path, parent, scene)
        GUDragBase.__init__(self, parent, brush, pen)
        
        self.setAcceptHoverEvents(True)
        self.setAcceptTouchEvents(True)
        #self.setAcceptedMouseButtons(Qt.LeftButton)

        self.setPen(self._pen_pos)
        self.setBrush(self._brush)
        self.setFlags(self.ItemIsSelectable | self.ItemIsMovable)

        #self.setBoundingRegionGranularity(0.95)
        self._mode = ADD
        #self.grabMouse() # makes available mouseMoveEvent 


    def pathForPointH(self, p, scene, rsize=5) :
        t = scene.views()[0].transform()
        sx, sy = rsize/t.m11(), rsize/t.m22()
        dx = QtCore.QPointF(sx,0)
        dy = QtCore.QPointF(0,sy)
        path = QtGui.QPainterPath(p+dx+dy)
        path.lineTo(p-dx+dy)
        path.lineTo(p-dx-dy)
        path.lineTo(p+dx-dy)
        path.lineTo(p+dx+dy)
        return path


    def pathForPointV(self, p, scene, rsize=7) :
        t = scene.views()[0].transform()
        sx, sy = rsize/t.m11(), rsize/t.m22()
        dx = QtCore.QPointF(sx,0)
        dy = QtCore.QPointF(0,sy)
        path = QtGui.QPainterPath(p+dx)
        path.lineTo(p+dy)
        path.lineTo(p-dx)
        path.lineTo(p-dy)
        path.lineTo(p+dx)
        return path


    def pathForPointR(self, p, scene, rsize=5) :
        t = scene.views()[0].transform()
        rx, ry = rsize/t.m11(), rsize/t.m22()
        #lst = self.listOfCirclePoints(p, rx, ry, np=12)
        path = QtGui.QPainterPath()
        path.addEllipse(p, rx, ry) 
        return path


    def listOfCirclePoints(self, p, rx=5, ry=5, np=12) :
        if self._lst_circle is None :
            from math import cos, sin, pi, ceil
            dphi = pi/np
            lst_sc = [(sin(i*dphi),cos(i*dphi)) for i in range(np)]
            self._lst_circle = [QtCore.QPoint(ceil(p.x()+rx*c), ceil(p.y()+ry*s)) for s,c in lst_sc]
        return self._lst_circle


    def pathForPointV0(self, p, scene, rsize=3) :
        t = scene.views()[0].transform()
        pc = view.mapFromScene(point)
        dr = rsize
        dx = QtCore.QPointF(dr,0)
        dy = QtCore.QPointF(0,dr)
        path = QtGui.QPainterPath(p+dx)
        path.lineTo(p+dy)
        path.lineTo(p-dx)
        path.lineTo(p-dy)
        path.lineTo(p+dx)
        return path


    def pathForPointV1(self, point, scene, rsize=3) :
        view = scene.views()[0]
        pc = view.mapFromScene(point)
        dp = QtCore.QPoint(rsize, rsize)
        recv = QtCore.QRect(pc-dp, pc+dp)
        poly = view.mapToScene(recv)
        path = QtGui.QPainterPath()
        path.addPolygon(poly)
        path.closeSubpath() 
        return path


#    def mousePressEvent(self, e) :
#        print '%s.mousePressEvent pos():' % self.__class__.__name__, self.pos()#, self.isSelected()
#        QGraphicsPathItem.mousePressEvent(self, e)
#        self.p0 = e.pos()
#        # this line should be commented; othervise selection is transferred to shape/rect.
#        #if self.parentItem() is not None : self.parentItem().setEnabled(self.isSelected())
#        #self.setSelected(True)
#        #print 'GUDragPoint.mousePressEvent, at point: ', e.pos(), ' scenePos: ', e.scenePos()
#        # COMMENTED!!! in ordert ot receive further events
#        #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(self.grub_cursor))


#    def mouseMoveEvent(self, e) :
#        print 'GUDragPoint:mouseMoveEvent'
#        #print 'GUDragPoint.mouseMoveEvent, at point: ', e.pos(), ' scenePos: ', e.scenePos()
#        dp = e.scenePos() - e.lastScenePos() + self.p0
#        self.moveBy(dp.x(), dp.y())
#        #QGraphicsPathItem.mouseMoveEvent(self, e)


    def mouseReleaseEvent(self, e) :
        QGraphicsPathItem.mouseReleaseEvent(self, e)
        if self._mode == ADD :
            self.set_mode()


#        print '%s.mouseReleaseEvent isSelected():' % self.__class__.__name__, self.isSelected()
#            self.ungrabMouse()
#        if self.parentItem() is not None : self.parentItem().setEnabled(True)
#        self.setSelected(False)
#        print 'mouseReleaseEvent'
#        QGraphicsPathItem.mouseReleaseEvent(self, e)
#        QtGui.QApplication.setOverrideCursor(QtGui.QCursor(self.hover_cursor))
#        QtGui.QApplication.restoreOverrideCursor()


#    def hoverEnterEvent(self, e) :
#        print '%s.hoverEnterEvent' % self.__class__.__name__
#        QGraphicsPathItem.hoverEnterEvent(self, e)
#        #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(self.hover_cursor))
     
     
#    def hoverLeaveEvent(self, e) :
#        print '%s.hoverLeaveEvent' % self.__class__.__name__
#        QGraphicsPathItem.hoverLeaveEvent(self, e)
#        #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(self.hover_cursor))
#        #QtGui.QApplication.restoreOverrideCursor()
        
     
#    def hoverMoveEvent(self, e) :
#        #print '%s.hoverMoveEvent' % self.__class__.__name__
#        QGraphicsPathItem.hoverMoveEvent(self, e)


#    def mouseDoubleClickEvent(self, e) :
#        QGraphicsPathItem.hoverLeaveEvent(self, e)
#        print '%s.mouseDoubleClickEvent, at point: ' % self.__class__.__name__, e.pos() #e.globalX(), e.globalY() 


#    def wheelEvent(self, e) :
#        QGraphicsPathItem.wheelEvent(self, e)
#        #print '%s.wheelEvent, at point: ' % self.__class__.__name__, e.pos() #e.globalX(), e.globalY() 


#    def focusInEvent(self, e) :
#        """for keyboard events"""
#        print 'GUDragPoint.focusInEvent, at point: ', e.pos()


#    def focusOutEvent(self, e) :
#        """for keyboard events"""
#        print 'GUDragPoint.focusOutEvent, at point: ', e.pos()


#    def emit_signal(self, msg='click') :
#        self.emit(QtCore.SIGNAL('event_on_rect(QString)'), msg)
#        #print msg

#-----------------------------
if __name__ == "__main__" :
    print 'Self test is not implemented...'
    print 'use > python GUViewImageWithShapes.py'
#-----------------------------
