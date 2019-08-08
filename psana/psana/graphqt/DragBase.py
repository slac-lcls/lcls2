"""
Class :py:class:`DragBase` is a base class for draggable objects
==================================================================

Created on 2016-09-14 by Mikhail Dubrovin
"""
#-----------------------------
import logging
logger = logging.getLogger(__name__)

from math import radians, sin, cos

from PyQt5.QtCore import Qt, QPoint, QPointF, QRect, QRectF
from PyQt5.QtGui import QPen, QBrush, QCursor

from psana.graphqt.DragTypes import * #dic_drag_type_to_name, UNDEF #, POINT, LINE, RECT, CIRC, POLY, WEDG
from psana.graphqt.QWUtils import select_item_from_popup_menu, select_color

#-----------------------------

FROZEN = 0
ADD    = 1
MOVE   = 2
EDIT   = 3
DELETE = 4

mode_types = ( FROZEN,   ADD,   MOVE,   EDIT,   DELETE)
mode_names = ('FROZEN', 'ADD', 'MOVE', 'EDIT', 'DELETE')

dic_mode_type_to_name = dict(zip(mode_types, mode_names))
dic_mode_name_to_type = dict(zip(mode_names, mode_types))

#-----------------------------

def print_warning(o, metframe) :
    wng = 'WARNING: %s.%16s - abstract interface method needs to be re-implemented in derived class.' \
          % (o.__class__.__name__, metframe.f_code.co_name)
    print(wng)
    #raise NotImplementedError(wng)

#-----------------------------

class DragBase(object) :
    
    def __init__(self, parent=None,\
                 brush=QBrush(), pen=QPen(Qt.blue, 0, Qt.SolidLine)) :

        #logger.debug('In DragBase create %s' % self.str_dragtype())

        self.set_drag_mode()
        self.set_child_item_sel()
        self.set_cursor_hover()
        self.set_cursor_grab()
        self.lst_ctl_points = None
        self._parent = parent
        self._brush = brush
        self._pen_pos = pen
        self._pen_inv = QPen(Qt.white, 3, Qt.SolidLine)
        self._pen_pos.setCosmetic(True)
        self._pen_inv.setCosmetic(True)

        #self.style_reddish = "background-color: rgb(220,   0,   0); color: rgb(0, 0, 0);" # Reddish background
        #self.style_transp  = "background-color: rgb(255,   0,   0, 100);"

    
    def set_drag_mode(self, mode=MOVE) :
        #logger.debug('In DragBase.set_drag_mode %s for %s' % (mode_names[mode], self.str_dragtype()))
        self._drag_mode = mode


    def mode(self) :
        return self._drag_mode


    def set_child_item_sel(self, item=None) :
        self._child_item_sel = item


    def child_item_sel(self) :
        return self._child_item_sel


    def set_control_points_visible(self, visible=True) :
        if self.lst_ctl_points is None : return
        for cpt in self.lst_ctl_points :
            cpt.setVisible(visible)
            #cpt.setZValue(100 if visible else 30)
            #cpt.setEnabled(True)

        #self.ped.setVisible(True)
        #self.ped.setZValue(200)

        self.setZValue(40 if visible else 20)


    def str_dragtype(self) :
        return dic_drag_type_to_name[self._dragtype]


    def set_cursor_hover(self, cursor=Qt.CrossCursor) :
        self.hover_cursor = cursor


    def set_cursor_grab(self, cursor=Qt.SizeAllCursor) : # Qt.ClosedHandCursor) :
        self.grub_cursor = cursor


    def control_point_menu(self) :
        lst = ('Invert', 'Delete', 'Color', 'Cancel')
        txt = select_item_from_popup_menu(lst)

        if txt == 'Invert' :
            self.setPen(self._pen_inv if self.pen()==self._pen_pos else self._pen_pos)

        elif txt == 'Delete' :
            #print('ask parent class:', self._parent, ' to kill self:', self)
            self.set_drag_mode(DELETE)
            self.setVisible(False)
            #self._parent.delete_item(self)

        elif txt == 'Cancel' :
            return

        elif txt == 'Color' :
            color = select_color(self._pen_pos.color())
            self._pen_pos = QPen(color, 2, Qt.SolidLine)
            self._pen_pos.setCosmetic(True)
            self.setPen(self._pen_pos)

        else :
            print('DragBase: this point features are not implemented for item "%s"' % txt)


    def rotate_point(self, p, sign=-1) :
        angle = self.rotation()
        angle_rads = sign * radians(angle)
        s,c = sin(angle_rads), cos(angle_rads)
        x,y = p.x(), p.y()
        return QPointF(x*c-y*s, x*s+y*c)


    def redefine_rect(self):
        """Keeps item rect Top-Left corner always (0,0) its position is defined in pos().
        """
        #ps = self.scenePos()
        #t0 = self.transformOriginPoint()
        p0 = self.pos()
        r0 = self.rect().normalized()
        tl = r0.topLeft()
        dp = self.rotate_point(tl, sign=1)

        #print('XXX item pos dx:%6.1f dy:%6.1f' % (p0.x(), p0.y()))
        #print('XXX rect tl    :%6.1f dy:%6.1f' % (tl.x(), tl.y()))
        #print('XXX rot  tl    :%6.1f dy:%6.1f' % (dp.x(), dp.y()))

        self.setRect(QRectF(QPointF(0,0), r0.size()))        
        self.setPos(p0+dp)

#-----------------------------
# Abstract interface methods
#-----------------------------

#    def create(self)            : print_warning(self, sys._getframe()) # ; return None
#    def move(self)              : print_warning(self, sys._getframe())
#    def move_is_completed(self) : print_warning(self, sys._getframe())
#    def contains(self)          : print_warning(self, sys._getframe())
#    def draw(self)              : print_warning(self, sys._getframe())
#    def print_attrs(self)       : print_warning(self, sys._getframe())

#-----------------------------
if __name__ == "__main__" :
    print('Self test is not implemented...')
    print('use > python FWViewImageShapes.py')
#-----------------------------
