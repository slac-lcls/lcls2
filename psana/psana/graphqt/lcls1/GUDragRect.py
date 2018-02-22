#!@PYTHON@
"""
Class :py:class:`GUDragRect` - for draggable shape item
=======================================================

Created on 2016-10-10 by Mikhail Dubrovin
"""
#-----------------------------

#import os
#import math
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QGraphicsRectItem

from graphqt.GUDragBase import *
from graphqt.GUDragPoint import GUDragPoint

#-----------------------------

class GUDragRect(QGraphicsRectItem, GUDragBase) :
                # QRectF, QGraphicsItem, QGraphicsScene
    def __init__(self, obj, parent=None, scene=None,\
                 brush=QtGui.QBrush(), pen=QtGui.QPen(Qt.blue, 0, Qt.SolidLine)) :
        """Adds QGraphics(Rect)Item to the scene. 

        Parameters

        obj : QPointF or shape type e.g. QRectF
              obj is QPointF - shape parameters are defined at first mouse click
              obj is QRectF - it will be drawn as is
        """
        GUDragBase.__init__(self, parent, brush, pen)

        rect = None
        if isinstance(obj, QtCore.QPointF) :
            rect = QtCore.QRectF(obj, obj + QtCore.QPointF(5,5))
            self._mode = ADD

        elif isinstance(obj, QtCore.QRectF) :
            rect = obj

        else : print 'GUDragRect - wrong init object type:', str(obj)
        parent_for_base = None
        QGraphicsRectItem.__init__(self, rect, parent_for_base, scene)
        if self._mode == ADD :
            self.grabMouse() # makes available mouseMoveEvent 
        
        self.setAcceptHoverEvents(True)
        #self.setAcceptTouchEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)

        self.setPen(self._pen_pos)
        self.setBrush(self._brush)

        # Flags: ItemIsPanel, ItemClipsChildrenToShape, ItemIsSelectable,
        # ItemIsMovable, itemClipsToShape, ItemSendsScenePositionChanges
        self.setFlags(self.ItemIsSelectable)
        #self.setEnabled(False) # is visible, but do not receive events
        #self.setVisible(False) # is not visible, do not receive events
        #self.setSelected(True)

        self.setHandlesChildEvents(True) # will responsive to child events

        #self.rotate(10)


    def set_control_points(self) :
        parent = self # None # 
        r = self.rect()
        self.ptr = GUDragPoint(r.topRight(),    parent, scene=self.scene(), rsize=5, orient='h')
        self.ptl = GUDragPoint(r.topLeft(),     parent, scene=self.scene(), rsize=5, orient='h')
        self.pbr = GUDragPoint(r.bottomRight(), parent, scene=self.scene(), rsize=5, orient='h')
        self.pbl = GUDragPoint(r.bottomLeft(),  parent, scene=self.scene(), rsize=5, orient='h')

        self.pct = GUDragPoint(0.5*(r.topRight()+r.topLeft()),       parent, scene=self.scene())
        self.pcl = GUDragPoint(0.5*(r.topLeft()+r.bottomLeft()),     parent, scene=self.scene())
        self.pcb = GUDragPoint(0.5*(r.bottomRight()+r.bottomLeft()), parent, scene=self.scene())
        self.pcr = GUDragPoint(0.5*(r.topRight()+r.bottomRight()),   parent, scene=self.scene())

        self.ped = GUDragPoint(0.7*r.topRight()+0.3*r.topLeft(), parent, scene=self.scene(),\
                               pen=QtGui.QPen(Qt.black, 2, Qt.SolidLine),\
                               brush=QtGui.QBrush(Qt.yellow, Qt.SolidPattern), orient='r', rsize=6)

        self.lst_ctl_points = [self.ptr, self.ptl, self.pbr, self.pbl,\
                               self.pct, self.pcl, self.pcb, self.pcr, self.ped]

        for cpt in self.lst_ctl_points : self.setZValue(100)


    def move_control_points(self, e) :

        r = self.rect().normalized()
        r0 = self.rect0

        dptr = r.topRight()   -r0.topRight()    + self.p0_ptr
        dptl = r.topLeft()    -r0.topLeft()     + self.p0_ptl
        dpbr = r.bottomRight()-r0.bottomRight() + self.p0_pbr
        dpbl = r.bottomLeft() -r0.bottomLeft()  + self.p0_pbl

        self.ptr.setPos(dptr)
        self.ptl.setPos(dptl)
        self.pbr.setPos(dpbr)
        self.pbl.setPos(dpbl)

        self.pct.setPos(0.5*(dptr+dptl))
        self.pcl.setPos(0.5*(dptl+dpbl))
        self.pcb.setPos(0.5*(dpbl+dpbr))
        self.pcr.setPos(0.5*(dptr+dpbr))

        self.ped.setPos(0.7*dptr+0.3*dptl)


    def itemChange(self, change, value) :
        #print '%s.itemChange' % (self.__class__.__name__), ' change: %d, value:' % change, value 
        valnew = QGraphicsRectItem.itemChange(self, change, value)
        if change == self.ItemSelectedHasChanged :
            self.set_control_points_visible(visible=self.isSelected())            
        return valnew


    def mousePressEvent(self, e) :
        #print '%s.mousePressEvent, at point: ' % self.__class__.__name__, e.pos(), e.scenePos()
        QGraphicsRectItem.mousePressEvent(self, e) # points would not show up w/o this line

        ps = e.scenePos()
        #print '%s.mousePressEvent itemAt:' % self.__class__.__name__, self.scene().itemAt(ps)

        item_sel = self.scene().itemAt(ps)
        if item_sel in self.lst_ctl_points :
            #print 'set mode EDIT'
            self.set_mode(EDIT)
            self.set_child_item_sel(item_sel)
            self.rect0 = self.rect().normalized()
            self.p0 = self.pos()

            self.p0_ptr = self.ptr.pos()
            self.p0_ptl = self.ptl.pos()
            self.p0_pbr = self.pbr.pos()
            self.p0_pbl = self.pbl.pos()

            if item_sel == self.ped : self.control_point_menu()

            #print '%s.mousePressEvent rect0' % self.__class__.__name__, self.rect0            
            #print '%s.mousePressEvent: pcb.pos()' % self.__class__.__name__, self.pcb.pos()


    def mouseMoveEvent(self, e) :
        #QGraphicsPathItem.mouseMoveEvent(self, e)
        #print '%s.mouseMoveEvent' % self.__class__.__name__
        #print '%s.mouseMoveEvent, at point: ' % self.__class__.__name__, e.pos(), ' scenePos: ', e.scenePos()

        dp = e.scenePos() - e.lastScenePos() 

        if self._mode == MOVE and self.isSelected() :
            self.moveBy(dp.x(), dp.y())

        elif self._mode == ADD :
            rect = self.rect()
            rect.setBottomRight(rect.bottomRight() + dp)
            self.setRect(rect)

        elif self._mode == EDIT :
            r = self.rect()
            i = self.child_item_sel()
            if   i == self.pbr : r.setBottomRight(r.bottomRight() + dp)
            elif i == self.ptr : r.setTopRight   (r.topRight()    + dp)
            elif i == self.ptl : r.setTopLeft    (r.topLeft()     + dp)
            elif i == self.pbl : r.setBottomLeft (r.bottomLeft()  + dp)

            elif i == self.pct : r.setTop   (r.top()    + dp.y())
            elif i == self.pcl : r.setLeft  (r.left()   + dp.x())
            elif i == self.pcb : r.setBottom(r.bottom() + dp.y())
            elif i == self.pcr : r.setRight (r.right()  + dp.x())

            elif i == self.ped : pass

            self.setRect(r)
            self.move_control_points(e)


    def mouseReleaseEvent(self, e):
        #QGraphicsPathItem.mouseReleaseEvent(self, e)

        if self._mode == ADD :
            self.ungrabMouse()
            self.setRect(self.rect().normalized())
            self.set_control_points()
            #self.setSelected(False)

        if self._mode == EDIT :
            self.set_child_item_sel(None)

        self.set_mode()


#    def hoverEnterEvent(self, e) :
#        #print '%s.hoverEnterEvent' % self.__class__.__name__
#        QGraphicsRectItem.hoverEnterEvent(self, e)
#        #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(self.hover_cursor))


#    def hoverLeaveEvent(self, e) :
#        #print '%s.hoverLeaveEvent' % self.__class__.__name__
#        QGraphicsRectItem.hoverLeaveEvent(self, e)
#        #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(self.hover_cursor))
#        #QtGui.QApplication.restoreOverrideCursor()
        

#    def hoverMoveEvent(self, e) :
#        #print '%s.hoverMoveEvent' % self.__class__.__name__
#        QGraphicsRectItem.hoverMoveEvent(self, e)


#    def mouseDoubleClickEvent(self, e) :
#        QGraphicsRectItem.hoverLeaveEvent(self, e)
#        print '%s.mouseDoubleClickEvent, at point: ' % self.__class__.__name__, e.pos() #e.globalX(), e.globalY() 


#    def wheelEvent(self, e) :
#        QGraphicsRectItem.wheelEvent(self, e)
#        #print '%s.wheelEvent, at point: ' % self.__class__.__name__, e.pos() #e.globalX(), e.globalY() 


#    def emit_signal(self, msg='click') :
#        self.emit(QtCore.SIGNAL('event_on_rect(QString)'), msg)
#        #print msg

#-----------------------------
if __name__ == "__main__" :
    print 'Self test is not implemented...'
    print 'use > python GUViewImageWithShapes.py'
#-----------------------------
