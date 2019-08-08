"""
Class :py:class:`DragPoly` - for draggable shape item
=======================================================

Created on 2019-06-25 by Mikhail Dubrovin
"""
#-----------------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtCore import Qt#, QPointF#, QPolygon, QPolygonF
from PyQt5.QtGui import QPolygonF
from psana.graphqt.DragBase import FROZEN, ADD, MOVE, EDIT, DELETE, POLY
from psana.graphqt.DragPoint import * # DragPoint, DragBase, Qt, QPen, QBrush, QCursor
from PyQt5.QtWidgets import QGraphicsPolygonItem

#-----------------------------

class DragPoly(QGraphicsPolygonItem, DragBase) :
                # QPolygonF, QGraphicsItem, QGraphicsScene
    def __init__(self, obj, parent=None, scene=None,\
                 brush=QBrush(), pen=QPen(Qt.blue, 0, Qt.SolidLine)) :
        """Adds QGraphics(Polygon)Item to the scene. 

        Parameters

        obj : QPointF or shape type e.g. QPolygonF
              obj is QPointF - shape parameters are defined at first mouse click
              obj is QPolygonF - it will be drawn as is
        """
        logger.debug('In DragPoly')

        poly = None
        if isinstance(obj, QPolygonF) : poly = obj
        elif isinstance(obj, QPointF) :
           poly = QPolygonF()
           poly.append(obj)
           #print('XXX DragPoly 0-point QPolygonF() size  = %d' % poly.size())

        if poly is None :
            logger.warning('DragPoly - wrong init object type:', str(obj))
            return

        self._dragtype = POLY
        self._end_of_add = False 
        self.poly0 = None
        parent_for_base = None
        QGraphicsPolygonItem.__init__(self, poly, parent_for_base)
        #DragBase.__init__(self, parent, brush, pen) # is called inside QGraphicsPolygonItem

        logger.debug('In DragPoly - superclass initialization is done')
        
        if isinstance(obj, QPointF) :
            self._drag_mode = ADD
            logger.debug('set elf._drag_mode = ADD, ADD:%d  _drag_mode:%d' % (ADD, self._drag_mode))

        if scene is not None: scene.addItem(self)

        if self._drag_mode == ADD :
            self.grabMouse() # makes available mouseMoveEvent 
            logger.debug('In DragPoly mode grabMouse()')

        self.setAcceptHoverEvents(True)
        #self.setAcceptTouchEvents(True)
        #self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)

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
        logger.debug('In DragPoly.set_control_points - TBE')
        #return
        parent = self # None
        o = self.polygon()
        scene=self.scene()

        points = [o.at(i) for i in range(o.size())]

        msg = 'DragPoly.set_control_points() poygon corner points:'
        for i,p in enumerate(points) : msg += '\n  %2d: x=%6.1f y=%6.1f ' % (i,p.x(),p.y())
        logger.debug(msg)

        #dragp = DragPoint(p, parent, scene, rsize=5, pshape='h')
        self.lst_ctl_points = [DragPoint(p, parent, scene) for p in points]

        if len(points)>1 :
            self.ped = DragPoint(0.7*points[0]+0.3*points[1], parent, scene,\
                                 pen=QPen(Qt.black, 2, Qt.SolidLine),\
                                 brush=QBrush(Qt.yellow, Qt.SolidPattern), pshape='r', rsize=6)
            self.lst_ctl_points.append(self.ped)

        for cpt in self.lst_ctl_points : self.setZValue(100)


    def move_control_points(self) :
        #logger.debug('In DragPoly.move_control_points - TBE')

        poly  = self.polygon()           
        poly0 = self.poly0

        for i,gri in enumerate(self.lst_ctl_points[:-1]) :
            dp = poly.at(i) - poly0.at(i) + self.pos0[i]
            gri.setPos(dp)

        points   = [poly.at(i)  for i in range(2)]
        points0  = [poly0.at(i) for i in range(2)]
        dp0, dp1 = points[0]-points0[0], points[1]-points0[1]
        self.ped.setPos(0.7*dp0+0.3*dp1 + self.pos0[-1])


    def itemChange(self, change, value) :
        #print('%s.itemChange' % (self.__class__.__name__), ' change: %d, value:' % change, value)
        valnew = QGraphicsPolygonItem.itemChange(self, change, value)
        if change == self.ItemSelectedHasChanged :
            #self.set_control_points_visible(visible=True)            
            self.set_control_points_visible(visible=self.isSelected())            
        return valnew


    def mousePressEvent(self, e) :
        #logger.debug('DragPoly.mousePressEvent, at point: %s on scene: %s '%\
        #             (str(e.pos()), str(e.scenePos()))) # self.__class__.__name__
        QGraphicsPolygonItem.mousePressEvent(self, e) # points would not show up w/o this line

        #print("DragPoly is selected: ", self.isSelected())
        #print('XXX DragPoly.mousePressEvent button L/R/M = 1/2/4: ', e.button())
        #print('XXX DragPoly.mousePressEvent Left: ', e.button()==Qt.LeftButton)

        if e.button()==Qt.RightButton : self._end_of_add = True

        ps = e.scenePos()
        #print('%s.mousePressEvent itemAt:' % self.__class__.__name__, self.scene().itemAt(ps))

        t = self.scene().views()[0].transform()
        item_sel = self.scene().itemAt(ps.x(), ps.y(), t)
        #item_sel = self.scene().itemAt(ps)

        if self.lst_ctl_points is None : 
            logger.warning('DragPoly.lst_ctl_points is None')
            return

        if item_sel in self.lst_ctl_points :
            self.indx_sel = self.lst_ctl_points.index(item_sel)
            print('XXX  DragPoly.mousePressEvent index_selected = ', self.indx_sel)

            self.set_drag_mode(EDIT)
            self.set_child_item_sel(item_sel)

            if item_sel == self.ped : self.control_point_menu()

            if self.poly0 is not None : del self.poly0
            self.poly0 = QPolygonF(self.polygon())
            self.pos0 = [gi.pos() for gi in self.lst_ctl_points]

            #print('%s.mousePressEvent poly0' % self.__class__.__name__, self.poly0)      
            #print('%s.mousePressEvent: pcb.pos()' % self.__class__.__name__, self.pcb.pos())


    def mouseMoveEvent(self, e) :
        QGraphicsPolygonItem.mouseMoveEvent(self, e)
        #logger.debug('%s.mouseMoveEvent' % self.__class__.__name__)
        #print('%s.mouseMoveEvent, at point: ' % self.__class__.__name__, e.pos(), ' scenePos: ', e.scenePos())
        #print('XXX mouseMoveEvent scenePos(), e.lastScenePos:', e.scenePos(), e.lastScenePos())
        dp = e.scenePos() - e.lastScenePos() 

        if self._drag_mode == MOVE and self.isSelected() :
            self.moveBy(dp.x(), dp.y())

        elif self._drag_mode == ADD :
            if self._end_of_add : return
            #print('%s.mouseMoveEvent _drag_mode=ADD' % self.__class__.__name__)
            poly = self.polygon()
            point = e.scenePos()
            if poly.size()==1 : poly.append(point)
            else              : poly.replace(poly.size()-1, point)
            self.setPolygon(poly)

        elif self._drag_mode == EDIT :
            o = self.polygon()
            i = self.child_item_sel()
            o.replace(self.indx_sel, e.scenePos() - self.pos()) #!!! # self.pos() in parent???
            self.setPolygon(o)
            self.move_control_points()


    def mouseReleaseEvent(self, e):
        #logger.debug('DragPoly.mouseReleaseEvent') # % self.__class__.__name__)
        QGraphicsPolygonItem.mouseReleaseEvent(self, e)

        if self._drag_mode == ADD :
            poly = self.polygon()
            point = e.scenePos()
            ind_last = poly.count()-1
            #print('XXX polygone ind_last = %d' % ind_last)

            if self._end_of_add :
                poly.replace(ind_last, point)
                self.ungrabMouse()
                self.set_drag_mode()
            else :
                poly.append(point)

            self.setPolygon(poly)

            if self._end_of_add :
                self.set_control_points()
            #self.setSelected(False)

        if self._drag_mode == EDIT :
            self.set_child_item_sel(None)
            self.set_drag_mode()


#    def hoverEnterEvent(self, e) :
#        #print('%s.hoverEnterEvent' % self.__class__.__name__)
#        QGraphicsPolygonItem.hoverEnterEvent(self, e)
#        #QApplication.setOverrideCursor(QCursor(self.hover_cursor))


#    def hoverLeaveEvent(self, e) :
#        #print('%s.hoverLeaveEvent' % self.__class__.__name__)
#        QGraphicsPolygonItem.hoverLeaveEvent(self, e)
#        #QApplication.setOverrideCursor(QCursor(self.hover_cursor))
#        #QApplication.restoreOverrideCursor()
        

#    def hoverMoveEvent(self, e) :
#        #print('%s.hoverMoveEvent' % self.__class__.__name__)
#        QGraphicsPolygonItem.hoverMoveEvent(self, e)


#    def mouseDoubleClickEvent(self, e) :
#        QGraphicsPolygonItem.hoverLeaveEvent(self, e)
#        print('%s.mouseDoubleClickEvent, at point: ' % self.__class__.__name__, e.pos() #e.globalX(), e.globalY())


#    def wheelEvent(self, e) :
#        QGraphicsPolygonItem.wheelEvent(self, e)
#        #print('%s.wheelEvent, at point: ' % self.__class__.__name__, e.pos() #e.globalX(), e.globalY())


#    def emit_signal(self, msg='click') :
#        self.emit(QtCore.SIGNAL('event_on_poly(QString)'), msg)
#        #print(msg)

#-----------------------------
if __name__ == "__main__" :
    print('Self test is not implemented...')
    print('use > python FWViewImageShapes.py')
#-----------------------------
