"""
Class :py:class:`DragEllipse` - for draggable shape item
=======================================================

Created on 2019-07-11 by Mikhail Dubrovin
"""
#-----------------------------

from math import atan2, degrees
import logging
logger = logging.getLogger(__name__)

from PyQt5.QtCore import QRectF #Qt, QPointF#, QRect, QRectF
from psana.graphqt.DragBase import FROZEN, ADD, MOVE, EDIT, DELETE, ELLIPSE
from psana.graphqt.DragPoint import * # DragPoint, DragBase, Qt, QPen, QBrush, QCursor
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsItem

#-----------------------------

class DragEllipse(QGraphicsEllipseItem, DragBase) :
                # QRectF, QGraphicsItem, QGraphicsScene
    def __init__(self, obj, parent=None, scene=None,\
                 brush=QBrush(), pen=QPen(Qt.blue, 0, Qt.SolidLine)) :
        """Adds QGraphics(Rect)Item to the scene. 

        Parameters

        obj : QPointF or shape type e.g. QRectF
              obj is QPointF - shape parameters are defined at first mouse click
              obj is QRectF - it will be drawn as is
        """
        logger.debug('In DragEllipse')

        rect = obj if isinstance(obj, QRectF) else\
               QRectF(obj, obj + QPointF(5,5)) if isinstance(obj, QPointF) else\
               None
        if rect is None :
            logger.warning('DragEllipse - wrong init object type:', str(obj))
            return

        self._dragtype = ELLIPSE
        parent_for_base = None
        QGraphicsEllipseItem.__init__(self, rect, parent_for_base)
        #DragBase.__init__(self, parent, brush, pen) # is called inside QGraphicsEllipseItem

        logger.debug('In DragEllipse - superclass initialization is done')
        
        if isinstance(obj, QPointF) :
            self._drag_mode = ADD
            logger.debug('set elf._drag_mode = ADD, ADD:%d  _drag_mode:%d' % (ADD, self._drag_mode))

        if scene is not None: scene.addItem(self)

        if self._drag_mode == ADD :
            self.grabMouse() # makes available mouseMoveEvent 
            logger.debug('In DragEllipse mode grabMouse()')

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

        #self.setHandlesChildEvents(True) # will responsive to child events
        #self.setFiltersChildEvents(True) # replacement?

        #self.rotate(10)

        #self.setSelected(True)
        #self.setEnabled(True)


    def set_control_points(self) :
        parent = self # None
        r = self.rect()
        scene=self.scene()
        self.ptr = DragPoint(r.topRight(),    parent, scene, rsize=5, pshape='h')
        self.ptl = DragPoint(r.topLeft(),     parent, scene, rsize=5, pshape='h')
        self.pbr = DragPoint(r.bottomRight(), parent, scene, rsize=5, pshape='h')
        self.pbl = DragPoint(r.bottomLeft(),  parent, scene, rsize=5, pshape='h')

        ct = 0.5*(r.topRight()+r.topLeft())
        cb = 0.5*(r.bottomRight()+r.bottomLeft())
        cl = 0.5*(r.topLeft()+r.bottomLeft())
        cr = 0.5*(r.topRight()+r.bottomRight())

        self.pct = DragPoint(ct, parent, scene)
        self.pcl = DragPoint(cl, parent, scene)
        self.pcb = DragPoint(cb, parent, scene)
        self.pcr = DragPoint(cr, parent, scene)

        self.ped = DragPoint(0.7*r.topRight()+0.3*r.topLeft(), parent, scene,\
                               pen=QPen(Qt.black, 2, Qt.SolidLine),\
                               brush=QBrush(Qt.yellow, Qt.SolidPattern), pshape='r', rsize=6)

        self.pro = DragPoint(ct+(ct-cb)/5, parent, scene, pshape='r', rsize=6)

        self.lst_ctl_points = [self.ptr, self.ptl, self.pbr, self.pbl,\
                               self.pct, self.pcl, self.pcb, self.pcr, self.ped, self.pro]

        for cpt in self.lst_ctl_points : self.setZValue(100)


    def move_control_points(self) :

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

        ct = 0.5*(dptr+dptl)
        cb = 0.5*(dpbl+dpbr)
        cl = 0.5*(dptl+dpbl)
        cr = 0.5*(dptr+dpbr)

        self.pct.setPos(ct)
        self.pcl.setPos(cl)
        self.pcb.setPos(cb)
        self.pcr.setPos(cr)

        self.ped.setPos(0.7*dptr+0.3*dptl)

        self.pro.setPos(ct+(ct-cb)/5)


    def itemChange(self, change, value) :
        #print('%s.itemChange' % (self.__class__.__name__), ' change: %d, value:' % change, value)
        valnew = QGraphicsEllipseItem.itemChange(self, change, value)
        if change == self.ItemSelectedHasChanged :
            #self.set_control_points_visible(visible=True)            
            self.set_control_points_visible(visible=self.isSelected())            
        return valnew


    def mousePressEvent(self, e) :
        logger.debug('DragEllipse.mousePressEvent, at point: %s on scene: %s '%\
                     (str(e.pos()), str(e.scenePos()))) # self.__class__.__name__
        QGraphicsEllipseItem.mousePressEvent(self, e) # points would not show up w/o this line
        #print("DragEllipse is selected: ", self.isSelected())

        ps = e.scenePos()
        #print('%s.mousePressEvent itemAt:' % self.__class__.__name__, self.scene().itemAt(ps))

        t = self.scene().views()[0].transform()
        item_sel = self.scene().itemAt(ps.x(), ps.y(), t)
        #item_sel = self.scene().itemAt(ps)

        if self.lst_ctl_points is None : 
            logger.warning('DragEllipse.lst_ctl_points is None')
            return

        if item_sel in self.lst_ctl_points :
            #print('set mode EDIT')
            self.set_drag_mode(EDIT)
            self.set_child_item_sel(item_sel)
            self.rect0 = self.rect().normalized()
            #self.p0 = self.pos()

            self.p0_ptr = self.ptr.pos()
            self.p0_ptl = self.ptl.pos()
            self.p0_pbr = self.pbr.pos()
            self.p0_pbl = self.pbl.pos()

            if item_sel == self.ped : self.control_point_menu()

            #print('%s.mousePressEvent rect0' % self.__class__.__name__, self.rect0)      
            #print('%s.mousePressEvent: pcb.pos()' % self.__class__.__name__, self.pcb.pos())


    def mouseMoveEvent(self, e) :
        QGraphicsEllipseItem.mouseMoveEvent(self, e)
        #logger.debug('%s.mouseMoveEvent' % self.__class__.__name__)
        #print('%s.mouseMoveEvent, at point: ' % self.__class__.__name__, e.pos(), ' scenePos: ', e.scenePos())

        dp = e.scenePos() - e.lastScenePos() 

        if self._drag_mode == MOVE and self.isSelected() :
            self.moveBy(dp.x(), dp.y())
            #self.move_control_points()

        elif self._drag_mode == ADD :
            #print('%s.mouseMoveEvent _drag_mode=ADD' % self.__class__.__name__)
            rect = self.rect()
            rect.setBottomRight(rect.bottomRight() + dp)
            self.setRect(rect)

        elif self._drag_mode == EDIT :
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

            elif i == self.pro :
                c = r.center()
                #print('=== rect center %6.1f %6.1f' % (c.x(), c.y()))
                v = e.scenePos() - self.mapToScene(c.x(), c.y()) # in scene coordinates
                angle = degrees(atan2(v.y(), v.x())) + 90
                self.setRotation(angle)

            r = r.normalized()
            self.setRect(r)

            if i != self.pro:
                self.setTransformOriginPoint(r.center())

            self.move_control_points()


    def mouseReleaseEvent(self, e):
        #logger.debug('DragEllipse.mouseReleaseEvent') # % self.__class__.__name__)
        QGraphicsEllipseItem.mouseReleaseEvent(self, e)

        if self._drag_mode == ADD :
            self.ungrabMouse()
            self.setRect(self.rect().normalized())

            self.set_control_points()
            #self.setSelected(False)

            self.setStartAngle(10*16)
            self.setSpanAngle(300*16)

        if self._drag_mode == EDIT :
            self.set_child_item_sel(None)

        self.set_drag_mode()

        c = self.rect().center()
        self.setTransformOriginPoint(c)
        #print('XXX set rect transform origin to rect center x:%6.1f y:%6.1f' % (c.x(), c.y()))
        #print('XXX item scenePos() x:%6.1f y:%6.1f' % (p.x(), p.y()))



#    def hoverEnterEvent(self, e) :
#        #print('%s.hoverEnterEvent' % self.__class__.__name__)
#        QGraphicsEllipseItem.hoverEnterEvent(self, e)
#        #QApplication.setOverrideCursor(QCursor(self.hover_cursor))


#    def hoverLeaveEvent(self, e) :
#        #print('%s.hoverLeaveEvent' % self.__class__.__name__)
#        QGraphicsEllipseItem.hoverLeaveEvent(self, e)
#        #QApplication.setOverrideCursor(QCursor(self.hover_cursor))
#        #QApplication.restoreOverrideCursor()
        

#    def hoverMoveEvent(self, e) :
#        #print('%s.hoverMoveEvent' % self.__class__.__name__)
#        QGraphicsEllipseItem.hoverMoveEvent(self, e)


#    def mouseDoubleClickEvent(self, e) :
#        QGraphicsEllipseItem.hoverLeaveEvent(self, e)
#        print('%s.mouseDoubleClickEvent, at point: ' % self.__class__.__name__, e.pos() #e.globalX(), e.globalY())


#    def wheelEvent(self, e) :
#        QGraphicsEllipseItem.wheelEvent(self, e)
#        #print('%s.wheelEvent, at point: ' % self.__class__.__name__, e.pos() #e.globalX(), e.globalY())


#    def emit_signal(self, msg='click') :
#        self.emit(QtCore.SIGNAL('event_on_rect(QString)'), msg)
#        #print(msg)

#-----------------------------
if __name__ == "__main__" :
    print('Self test is not implemented...')
    print('use > python FWViewImageShapes.py')
#-----------------------------
