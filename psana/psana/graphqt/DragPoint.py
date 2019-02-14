"""
Class :py:class:`DragPoint` - for draggable shape item
========================================================

Created on 2016-10-11 by Mikhail Dubrovin
"""
#-----------------------------

from PyQt5.QtGui import QBrush, QPen, QPainterPath#, QCursor
from PyQt5.QtCore import Qt, QPointF #, QPoint, QRect, QRectF
from PyQt5.QtWidgets import QGraphicsPathItem # QApplication

from psana.graphqt.DragBase import DragBase, FROZEN, ADD, MOVE, EDIT, DELETE #, dic_mode_type_to_name

#-----------------------------

class DragPoint(QGraphicsPathItem, DragBase) :
                # QPointF, QGraphicsItem, QGraphicsScene
    def __init__(self, point=QPointF(0,0), parent=None, scene=None,\
                 brush=QBrush(Qt.white, Qt.SolidPattern),\
                 pen=QPen(Qt.black, 2, Qt.SolidLine),\
                 orient='v', rsize=7) :

        self._lst_circle = None

        path = self.pathForPointV(point, scene, rsize) if orient=='v' else\
               self.pathForPointH(point, scene, rsize) if orient=='h' else\
               self.pathForPointR(point, scene, rsize)

        QGraphicsPathItem.__init__(self, path, parent)
        if scene is not None: scene.addItem(self)

        DragBase.__init__(self, parent, brush, pen)
        
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
        dx = QPointF(sx,0)
        dy = QPointF(0,sy)
        path = QPainterPath(p+dx+dy)
        path.lineTo(p-dx+dy)
        path.lineTo(p-dx-dy)
        path.lineTo(p+dx-dy)
        path.lineTo(p+dx+dy)
        return path


    def pathForPointV(self, p, scene, rsize=7) :
        t = scene.views()[0].transform()
        sx, sy = rsize/t.m11(), rsize/t.m22()
        dx = QPointF(sx,0)
        dy = QPointF(0,sy)
        path = QPainterPath(p+dx)
        path.lineTo(p+dy)
        path.lineTo(p-dx)
        path.lineTo(p-dy)
        path.lineTo(p+dx)
        return path


    def pathForPointR(self, p, scene, rsize=5) :
        t = scene.views()[0].transform()
        rx, ry = rsize/t.m11(), rsize/t.m22()
        #lst = self.listOfCirclePoints(p, rx, ry, np=12)
        path = QPainterPath()
        path.addEllipse(p, rx, ry) 
        return path


    def listOfCirclePoints(self, p, rx=5, ry=5, np=12) :
        if self._lst_circle is None :
            from math import cos, sin, pi, ceil
            dphi = pi/np
            lst_sc = [(sin(i*dphi),cos(i*dphi)) for i in range(np)]
            self._lst_circle = [QPoint(ceil(p.x()+rx*c), ceil(p.y()+ry*s)) for s,c in lst_sc]
        return self._lst_circle


    def pathForPointV0(self, p, scene, rsize=3) :
        t = scene.views()[0].transform()
        pc = view.mapFromScene(point)
        dr = rsize
        dx = QPointF(dr,0)
        dy = QPointF(0,dr)
        path = QPainterPath(p+dx)
        path.lineTo(p+dy)
        path.lineTo(p-dx)
        path.lineTo(p-dy)
        path.lineTo(p+dx)
        return path


    def pathForPointV1(self, point, scene, rsize=3) :
        view = scene.views()[0]
        pc = view.mapFromScene(point)
        dp = QPoint(rsize, rsize)
        recv = QRect(pc-dp, pc+dp)
        poly = view.mapToScene(recv)
        path = QPainterPath()
        path.addPolygon(poly)
        path.closeSubpath() 
        return path


#    def mousePressEvent(self, e) :
#        print('%s.mousePressEvent pos():' % self.__class__.__name__, self.pos()#, self.isSelected())
#        QGraphicsPathItem.mousePressEvent(self, e)
#        self.p0 = e.pos()
#        # this line should be commented; othervise selection is transferred to shape/rect.
#        #if self.parentItem() is not None : self.parentItem().setEnabled(self.isSelected())
#        #self.setSelected(True)
#        #print('DragPoint.mousePressEvent, at point: ', e.pos(), ' scenePos: ', e.scenePos())
#        # COMMENTED!!! in ordert ot receive further events
#        #QApplication.setOverrideCursor(QCursor(self.grub_cursor))


#    def mouseMoveEvent(self, e) :
#        print('DragPoint:mouseMoveEvent', e.pos())
#        #print('DragPoint.mouseMoveEvent, at point: ', e.pos(), ' scenePos: ', e.scenePos())
#        dp = e.scenePos() - e.lastScenePos() + self.p0
#        self.moveBy(dp.x(), dp.y())
#        QGraphicsPathItem.mouseMoveEvent(self, e)


    def mouseReleaseEvent(self, e) :
        print('DragPoint:mouseReleaseEvent', e.pos())
        QGraphicsPathItem.mouseReleaseEvent(self, e)
        if self._mode == ADD :
            self.set_mode()


#        print('%s.mouseReleaseEvent isSelected():' % self.__class__.__name__, self.isSelected())
#            self.ungrabMouse()
#        if self.parentItem() is not None : self.parentItem().setEnabled(True)
#        self.setSelected(False)
#        print('mouseReleaseEvent')
#        QGraphicsPathItem.mouseReleaseEvent(self, e)
#        QApplication.setOverrideCursor(QCursor(self.hover_cursor))
#        QApplication.restoreOverrideCursor()


#    def hoverEnterEvent(self, e) :
#        print('%s.hoverEnterEvent' % self.__class__.__name__)
#        QGraphicsPathItem.hoverEnterEvent(self, e)
#        #QApplication.setOverrideCursor(QCursor(self.hover_cursor))
     
     
#    def hoverLeaveEvent(self, e) :
#        print('%s.hoverLeaveEvent' % self.__class__.__name__)
#        QGraphicsPathItem.hoverLeaveEvent(self, e)
#        #QApplication.setOverrideCursor(QCursor(self.hover_cursor))
#        #QApplication.restoreOverrideCursor()
        
     
#    def hoverMoveEvent(self, e) :
#        #print('%s.hoverMoveEvent' % self.__class__.__name__)
#        QGraphicsPathItem.hoverMoveEvent(self, e)


#    def mouseDoubleClickEvent(self, e) :
#        QGraphicsPathItem.hoverLeaveEvent(self, e)
#        print('%s.mouseDoubleClickEvent, at point: ' % self.__class__.__name__, e.pos() #e.globalX(), e.globalY())


#    def wheelEvent(self, e) :
#        QGraphicsPathItem.wheelEvent(self, e)
#        #print('%s.wheelEvent, at point: ' % self.__class__.__name__, e.pos() #e.globalX(), e.globalY())


#    def focusInEvent(self, e) :
#        """for keyboard events"""
#        print('DragPoint.focusInEvent, at point: ', e.pos())


#    def focusOutEvent(self, e) :
#        """for keyboard events"""
#        print('DragPoint.focusOutEvent, at point: ', e.pos())


#    def emit_signal(self, msg='click') :
#        self.emit(QtCore.SIGNAL('event_on_rect(QString)'), msg)
#        #print(msg)

#-----------------------------
if __name__ == "__main__" :
    print('Self test is not implemented...')
    print('use > python FWViewImageShapes.py')
#-----------------------------
