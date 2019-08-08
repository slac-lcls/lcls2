"""
Class :py:class:`DragRect` - for draggable shape item
=======================================================

Created on 2016-10-10 by Mikhail Dubrovin
"""
#-----------------------------

from math import atan2, degrees #, radians, sin, cos
import logging
logger = logging.getLogger(__name__)

from PyQt5.QtCore import QRectF #Qt, QPointF#, QRect, QRectF
from psana.graphqt.DragBase import FROZEN, ADD, MOVE, EDIT, DELETE, RECT
from psana.graphqt.DragPoint import * # DragPoint, DragBase, Qt, QPen, QBrush, QCursor
from PyQt5.QtWidgets import QGraphicsRectItem

#-----------------------------

class DragRect(QGraphicsRectItem, DragBase) :
                # QRectF, QGraphicsItem, QGraphicsScene
    def __init__(self, obj, parent=None, scene=None,\
                 brush=QBrush(), pen=QPen(Qt.blue, 0, Qt.SolidLine)) :
        """Adds QGraphics(Rect)Item to the scene. 

        Parameters

        obj : QPointF or shape type e.g. QRectF
              obj is QPointF - shape parameters are defined at first mouse click
              obj is QRectF - it will be drawn as is
        """
        logger.debug('In DragRect')

        rect = obj if isinstance(obj, QRectF) else\
               QRectF(obj, obj + QPointF(5,5)) if isinstance(obj, QPointF) else\
               None
        if rect is None :
            logger.warning('DragRect - wrong init object type:', str(obj))
            return

        self._dragtype = RECT
        parent_for_base = None
        QGraphicsRectItem.__init__(self, rect, parent_for_base)
        #DragBase.__init__(self, parent, brush, pen) # is called inside QGraphicsRectItem

        logger.debug('In DragRect - superclass initialization is done')
        
        if isinstance(obj, QPointF) :
            self._drag_mode = ADD
            logger.debug('set elf._drag_mode = ADD, ADD:%d  _drag_mode:%d' % (ADD, self._drag_mode))

        if scene is not None: scene.addItem(self)

        if self._drag_mode == ADD :
            self.grabMouse() # makes available mouseMoveEvent 
            logger.debug('In DragRect mode grabMouse()')

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
        logger.debug('in DragRect.set_control_points')

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

        #self.pro = DragPoint(ct+(ct-cb)/5, parent, scene, pshape='r', rsize=6)
        self.pro = DragPoint(0.3*r.topRight()+0.7*r.topLeft(), parent, scene, pshape='r', rsize=6)

        self.lst_ctl_points = [self.ptr, self.ptl, self.pbr, self.pbl,\
                               self.pct, self.pcl, self.pcb, self.pcr, self.ped, self.pro]

        for cpt in self.lst_ctl_points : self.setZValue(100)

        logger.debug('exit DragRect.set_control_points')


    def remove_control_points(self) :
        logger.debug('DragRect.remove_control_points')
        scene=self.scene()
        for item in self.lst_ctl_points : 
            scene.removeItem(item)
            del item
        #scene.update()


    def move_control_points(self) :

        r0 = self.rect0
        r = self.rect().normalized()
        #p = self.pos() 
        #print('DragRect.move_control_points rect:', r)

        tr = r.topRight()    - r0.topRight()     + self.p0_ptr
        tl = r.topLeft()     - r0.topLeft()      + self.p0_ptl
        br = r.bottomRight() - r0.bottomRight()  + self.p0_pbr
        bl = r.bottomLeft()  - r0.bottomLeft()   + self.p0_pbl

        self.ptr.setPos(tr)
        self.ptl.setPos(tl)
        self.pbr.setPos(br)
        self.pbl.setPos(bl)

        ct = 0.5*(tr+tl)
        cb = 0.5*(bl+br)
        cl = 0.5*(tl+bl)
        cr = 0.5*(tr+br)

        self.pct.setPos(ct)
        self.pcl.setPos(cl)
        self.pcb.setPos(cb)
        self.pcr.setPos(cr)

        self.ped.setPos(0.7*tr+0.3*tl)
        self.pro.setPos(0.3*tr+0.7*tl)

        #self.pro.setPos(ct+(ct-cb)/5)


    def itemChange(self, change, value) :
        #print('%s.itemChange' % (self.__class__.__name__), ' change: %d, value:' % change, value)
        valnew = QGraphicsRectItem.itemChange(self, change, value)
        if change == self.ItemSelectedHasChanged :
            #self.set_control_points_visible(visible=True)            
            self.set_control_points_visible(visible=self.isSelected())            
        return valnew


    def mousePressEvent(self, e) :
        ps = e.scenePos()
        pe = e.pos()

        logger.debug('DragRect.mousePressEvent, at point: %6.1f %6.1f on scene: %6.1f %6.1f'%\
                     (pe.x(), pe.y(), ps.x(), ps.y()))
        QGraphicsRectItem.mousePressEvent(self, e) # points would not show up w/o this line
        #print("DragRect is selected: ", self.isSelected())

        #print('%s.mousePressEvent itemAt:' % self.__class__.__name__, self.scene().itemAt(ps))

        t = self.scene().views()[0].transform()
        i = item_sel = self.scene().itemAt(ps.x(), ps.y(), t)
        #item_sel = self.scene().itemAt(ps)

        if self.lst_ctl_points is None : 
            logger.warning('DragRect.lst_ctl_points is None')
            return

        if i in self.lst_ctl_points :

            r = self.rect()

            #print('set mode EDIT')
            self.set_drag_mode(EDIT)
            self.set_child_item_sel(i)
            self.rect0 = self.rect().normalized()
            #self.p0 = self.pos()

            self.p0_ptr = self.ptr.pos()
            self.p0_ptl = self.ptl.pos()
            self.p0_pbr = self.pbr.pos()
            self.p0_pbl = self.pbl.pos()

            if i == self.ped : self.control_point_menu()

            #print('%s.mousePressEvent rect0' % self.__class__.__name__, self.rect0)      
            #print('%s.mousePressEvent: pcb.pos()' % self.__class__.__name__, self.pcb.pos())


    def mouseMoveEvent(self, e) :
        QGraphicsRectItem.mouseMoveEvent(self, e)
        #logger.debug('%s.mouseMoveEvent' % self.__class__.__name__)
        #print('%s.mouseMoveEvent, at point: ' % self.__class__.__name__, e.pos(), ' scenePos: ', e.scenePos())

        dp = e.scenePos() - e.lastScenePos() 
        r = self.rect()

        if self._drag_mode == MOVE and self.isSelected() :
            self.moveBy(dp.x(), dp.y())

        elif self._drag_mode == ADD :
            #print('%s.mouseMoveEvent _drag_mode=ADD' % self.__class__.__name__)
            r.setBottomRight(r.bottomRight() + dp)
            self.setRect(r)

        elif self._drag_mode == EDIT :
            i = self.child_item_sel()

            dp = self.rotate_point(dp)

            if   i == self.pbr : r.setBottomRight(r.bottomRight() + dp)
            elif i == self.ptr : r.setTopRight   (r.topRight()    + dp)
            elif i == self.ptl : r.setTopLeft    (r.topLeft()     + dp)
            elif i == self.pbl : r.setBottomLeft (r.bottomLeft()  + dp)

            elif i == self.pct : r.setTop   (r.top()    + dp.y())
            elif i == self.pcl : r.setLeft  (r.left()   + dp.x())
            elif i == self.pcb : r.setBottom(r.bottom() + dp.y())
            elif i == self.pcr : r.setRight (r.right()  + dp.x())

            elif i == self.pro :
                #c = r.center()
                #c = r.topLeft()
                c = self.transformOriginPoint()
                #print('=== rect center %6.1f %6.1f' % (c.x(), c.y()))
                v = e.scenePos() - self.mapToScene(c.x(), c.y()) # in scene coordinates
                angle = degrees(atan2(v.y(), v.x())) #+ 90
                self.setRotation(angle)

            r = r.normalized()
            self.setRect(r)

            self.move_control_points()


    def mouseReleaseEvent(self, e):
        #logger.debug('DragRect.mouseReleaseEvent') # % self.__class__.__name__)
        QGraphicsRectItem.mouseReleaseEvent(self, e)

        if self._drag_mode == ADD :
            self.ungrabMouse()
            self.setRect(self.rect().normalized())
            self.setTransformOriginPoint(QPointF(0,0))
            self.redefine_rect()
            self.set_control_points()

        if self._drag_mode == EDIT :
            self.set_child_item_sel(None)
            self.redefine_rect()
            self.move_control_points()
        
        self.set_drag_mode()


#    def hoverEnterEvent(self, e) :
#        #print('%s.hoverEnterEvent' % self.__class__.__name__)
#        QGraphicsRectItem.hoverEnterEvent(self, e)
#        #QApplication.setOverrideCursor(QCursor(self.hover_cursor))


#    def hoverLeaveEvent(self, e) :
#        #print('%s.hoverLeaveEvent' % self.__class__.__name__)
#        QGraphicsRectItem.hoverLeaveEvent(self, e)
#        #QApplication.setOverrideCursor(QCursor(self.hover_cursor))
#        #QApplication.restoreOverrideCursor()
        

#    def hoverMoveEvent(self, e) :
#        #print('%s.hoverMoveEvent' % self.__class__.__name__)
#        QGraphicsRectItem.hoverMoveEvent(self, e)


#    def mouseDoubleClickEvent(self, e) :
#        QGraphicsRectItem.hoverLeaveEvent(self, e)
#        print('%s.mouseDoubleClickEvent, at point: ' % self.__class__.__name__, e.pos() #e.globalX(), e.globalY())


#    def wheelEvent(self, e) :
#        QGraphicsRectItem.wheelEvent(self, e)
#        #print('%s.wheelEvent, at point: ' % self.__class__.__name__, e.pos() #e.globalX(), e.globalY())


#    def emit_signal(self, msg='click') :
#        self.emit(QtCore.SIGNAL('event_on_rect(QString)'), msg)
#        #print(msg)

#-----------------------------
if __name__ == "__main__" :
    print('Self test is not implemented...')
    print('use > python FWViewImageShapes.py')
#-----------------------------
