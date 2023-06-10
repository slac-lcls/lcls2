
"""
Class :py:class:`GWROIUtils` is a ROI for interactive image GWViewImageROI
============================================================================

ROIPixel <- ROIBase

Usage ::

    from psana.graphqt.GWROIUtils import *
    import psana.graphqt.GWROIUtils as roiu
See:
    - :class:`GWViewImageROI`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2023-06-05 by Mikhail Dubrovin
"""
#import sys

import logging
logger = logging.getLogger(__name__)

import math # radians, cos, sin, ceil, pi
import numpy as np


from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QApplication,\
     QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsLineItem, QGraphicsPolygonItem, QGraphicsPathItem
from PyQt5.QtGui import  QPen, QPainter, QColor, QBrush, QCursor, QTransform, QPolygonF,  QPainterPath
from PyQt5.QtCore import Qt, QPoint, QPointF, QRect, QRectF, QSize, QSizeF, QLineF # , pyqtSignal

QPEN_DEF   = QPen(Qt.yellow, 1, Qt.SolidLine)  # Qt.DashLine
QBRUSH_DEF = QBrush()  # Qt.red
QCOLOR_DEF = QColor(Qt.yellow)

NONE    = 0

PIXEL   = 1
LINE    = 2
RECT    = 4
SQUARE  = 8
POLYGON = 16
POLYREG = 32
ELLIPSE = 64
CIRCLE  = 128
ARCH    = 256
roi_tuple = (
    (NONE,    'NONE',      '0'),
    (PIXEL,   'PIXEL',     '1'),
    (LINE,    'LINE',      '2'),
    (RECT,    'RECT',      '3'),
    (SQUARE,  'SQUARE',    '4'),
    (POLYGON, 'POLYGON',   '5'),
    (POLYREG, 'POLYREG',   '6'),
    (ELLIPSE, 'ELLIPSE',   '7'),
    (CIRCLE,  'CIRCLE',    '8'),
    (ARCH,    'ARCH',      '9'),
)
roi_types = [r[0] for r in roi_tuple]
roi_names = [r[1] for r in roi_tuple]
roi_keys  = [r[2] for r in roi_tuple]
dict_roi_type_name = {t:n for t,n,k in roi_tuple}

UNVISIBLE = 1
VISIBLE   = 2
ADD       = 4
DELETE    = 8
EDIT      = 16
TERMINATE = 32
mode_tuple = (
  (NONE,      'NONE',      'X'),
  (UNVISIBLE, 'UNVISIBLE', 'U'),
  (VISIBLE,   'VISIBLE',   'V'),
  (ADD,       'ADD',       'A'),
  (DELETE,    'DELETE',    'D'),
  (EDIT,      'EDIT',      'E'),
  (TERMINATE, 'TERMINATE', 'T'),
)
mode_types = [t for t,n,k in mode_tuple]
mode_names = [n for t,n,k in mode_tuple]
mode_keys  = [k for t,n,k in mode_tuple]
dict_mode_type_name = {t:n for t,n,k in mode_tuple}

ORIGIN    = 1
CENTER    = 2
TRANSLATE = 4
ROTATE    = 8
SCALE     = 16
MENU      = 32
OTHER     = 64
handle_tuple = (
  (NONE,      'NONE'),
  (ORIGIN,    'ORIGIN'),
  (CENTER,    'CENTER'),
  (TRANSLATE, 'TRANSLATE'),
  (ROTATE,    'ROTATE'),
  (SCALE,     'SCALE'),
  (MENU,      'MENU'),
  (OTHER,     'OTHER'),
)
handle_types = [t for t,n in handle_tuple]
handle_names = [n for t,n in handle_tuple]
dict_handle_type_name = {t:n for t,n in handle_tuple}


class ROIBase():
    def __init__(self, **kwa):
        #self.roi_type   = kwa.get('roi_type', NONE)
        self.roi_name   = dict_roi_type_name[self.roi_type]
        self.view       = kwa.get('view', None) # QGraphicsView()
        self.pos        = QPointF(kwa.get('pos', QPointF(0,0)))
        self.mode       = kwa.get('mode', 0) # 0/1/2/4/8/16 - invisible/visible/add/move/rotate/edit
        self.moveable   = kwa.get('moveable', False)
        self.scaleable  = kwa.get('scaleable', False)
        self.rotateable = kwa.get('rotateable', False)
        self.scitem     = None
        self.aspect_1x1 = False
        self.set_color_pen_brush(**kwa)
        assert isinstance(self.view, QGraphicsView)
        assert isinstance(self.pos, (QPoint, QPointF))

    def set_color_pen_brush(self, color=QCOLOR_DEF, pen=QPEN_DEF, brush=QBRUSH_DEF, **kwa):
        self.pen   = pen
        self.brush = brush
        self.color = color
        self.pen.setCosmetic(kwa.get('pen_is_cosmetic', True))
        if color is not None:
            self.color = color
            self.pen.setColor(color)
            self.brush.setColor(color)

    def scene(self):
        return self.view.scene()

    def add_to_scene(self, pen=None, brush=None):
        item = self.scitem
        item.setPen(self.pen if pen is None else pen)
        if self.roi_type != LINE:
            item.setBrush(self.brush if brush is None else brush)
        self.scene().addItem(item)

    def move_at_add(self, pos):
        logging.warning('ROIBase.move_at_add must be re-implemented in subclasses')


def items_at_point(scene, point):
    items = scene.items(point)
    logging.debug('sc.itemsAt(%s): %s' % (str(point), str(items)))
    return items


class ROIPixel(ROIBase):
    def __init__(self, **kwa):
        self.roi_type = PIXEL
        ROIBase.__init__(self, **kwa)

        self.brush = QCOLOR_DEF  # kwa.get('brush', QBrush(QCOLOR_DEF))
        self.pen = QPen(QPEN_DEF)  # kwa.get('pen', QPen(QPEN_DEF))
        # self.pos = kwa.get('pos', QPoint(0,0))
        print('XXX you are in ROIPixel now')

    def add_to_scene(self, pos=None):
        """Adds pixel as a rect to scene, returns QGraphicsRectItem"""
        print('XXX you are in ROIPixel.add_to_scene')
        if pos is not None: self.pos = pos
        self.scitem = QGraphicsRectItem(QRectF(QPointF(self.pos), QSizeF(1,1)))
        ROIBase.add_to_scene(self, pen=self.pen, brush=self.brush)
        #self.scitem = self.scene().addRect(QRectF(QPointF(pos), QSizeF(1,1)), self.pen, self.brush)


class ROILine(ROIBase):

    def __init__(self, **kwa):
        self.roi_type = LINE
        ROIBase.__init__(self, **kwa)

    def add_to_scene(self, line=None, pen=None):
        """Adds/returns QGraphicsLineItem to scene"""
        if line is None: line = QLineF(self.pos, self.pos+QPointF(10,10))
        self.scitem = QGraphicsLineItem(QLineF(line))
        ROIBase.add_to_scene(self, pen, brush=None)
        #self.scitem = self.scene().addLine(QLineF(line), pen, brush)


    def move_at_add(self, pos):
        logger.debug('ROILine.move_at_add')
        line = self.scitem.line()
        line.setP2(pos)
        self.scitem.setLine(line)


class ROIRect(ROIBase):
    def __init__(self, **kwa):
        self.roi_type = RECT
        ROIBase.__init__(self, **kwa)

    def add_to_scene(self, rect=None, pen=None, brush=None, angle_deg=0):
        """Adds/returns QGraphicsRectItem to scene"""
        if rect is None: rect = QRectF(self.pos, QSizeF(10,10))
        item = self.scitem = QGraphicsRectItem(QRectF(rect))
        item.setTransformOriginPoint(item.rect().center())
        if angle_deg != 0: item.setRotation(angle_deg)
        ROIBase.add_to_scene(self, pen, brush)

    def move_at_add(self, pos):
        logger.debug('ROIRect.move_at_add')
        rect = self.scitem.rect()
        rect.setBottomRight(pos)
        self.scitem.setRect(rect)


def rect_to_square(rect, pos):
    dp = pos - rect.topLeft()
    w,h = dp.x(), dp.y()
    v = max(abs(w), abs(h))
    w,h = math.copysign(v,w), math.copysign(v,h)
    rect.setSize(QSizeF(w,h))
    return rect


class ROISquare(ROIRect):
    def __init__(self, **kwa):
        self.roi_type = SQUARE
        ROIRect.__init__(self, **kwa)
        self.aspect_1x1 = True

    def move_at_add(self, pos):
        logger.debug('ROISquare.move_at_add')
        self.scitem.setRect(rect_to_square(self.scitem.rect(), pos))


class ROIPolygon(ROIBase):
    def __init__(self, **kwa):
        self.roi_type = POLYGON
        ROIBase.__init__(self, **kwa)

    def add_to_scene(self, poly=None, pen=QPEN_DEF, brush=QBRUSH_DEF):
        """Adds/returns QGraphicsPolygonItem to scene"""
        if poly is None: poly = QPolygonF((self.pos, self.pos+QPointF(10,0), self.pos+QPointF(0,10)))
        self.scitem = QGraphicsPolygonItem(QPolygonF(poly))
        ROIBase.add_to_scene(self, pen, brush)
        #self.scitem = self.scene().addPolygon(QPolygonF(poly), pen, brush)

    def move_at_add(self, pos):
        logger.warning('TBD ROIPolygon.move_at_add')


class ROIPolyreg(ROIPolygon):
    def __init__(self, **kwa):
        self.roi_type = POLYREG
        ROIPolygon.__init__(self, **kwa)
        self.aspect_1x1 = True


class ROIEllipse(ROIBase):

    def __init__(self, **kwa):
        self.roi_type = ELLIPSE
        ROIBase.__init__(self, **kwa)

    def add_to_scene(self, rect=None, pen=QPEN_DEF, brush=QBRUSH_DEF,\
                     angle_deg=0, start_angle=0, span_angle=360):
        """Adds/returns QGraphicsEllipseItem to scene."""
        if rect is None: rect = QRectF(self.pos, QSizeF(10,10))
        item = self.scitem = QGraphicsEllipseItem(QRectF(rect))
        item.setStartAngle(start_angle*16)
        item.setSpanAngle(span_angle*16)
        item.setTransformOriginPoint(item.rect().center())
        item.setRotation(angle_deg)
        ROIBase.add_to_scene(self, pen, brush)
        #self.scitem = self.scene().addEllipse(QRectF(rect), pen, brush)

    def move_at_add(self, pos):
        logger.debug('ROIEllipse.move_at_add')
        rect = self.scitem.rect()
        rect.setBottomRight(pos)
        self.scitem.setRect(rect)


class ROICircle(ROIEllipse):
    def __init__(self, **kwa):
        self.roi_type = CIRCLE
        ROIEllipse.__init__(self, **kwa)
        self.aspect_1x1 = True

    def move_at_add(self, pos):
        logger.debug('ROICircle.move_at_add')
        self.scitem.setRect(rect_to_square(self.scitem.rect(), pos))


class ROIArch(ROIBase):
    def __init__(self, **kwa):
        self.roi_type = ARCH
        ROIBase.__init__(self, **kwa)

    def add_to_scene(self, rect=None, pen=QPEN_DEF, brush=QBRUSH_DEF,\
                     angle_deg=0, start_angle=0, span_angle=360):
        """Adds/returns QGraphicsEllipseItem to scene."""
        if rect is None: rect = QRectF(self.pos, QSizeF(10,10))
        print('WARNING TBD ROIArch')


def select_roi(roi_type, view=None, pos=QPointF(1,1), **kwa):
    o = ROIPixel  (view=view, pos=pos, **kwa) if roi_type == PIXEL else\
        ROILine   (view=view, pos=pos, **kwa) if roi_type == LINE else\
        ROIRect   (view=view, pos=pos, **kwa) if roi_type == RECT else\
        ROISquare (view=view, pos=pos, **kwa) if roi_type == SQUARE else\
        ROIPolygon(view=view, pos=pos, **kwa) if roi_type == POLYGON else\
        ROIPolyreg(view=view, pos=pos, **kwa) if roi_type == POLYREG else\
        ROICircle (view=view, pos=pos, **kwa) if roi_type == CIRCLE else\
        ROIEllipse(view=view, pos=pos, **kwa) if roi_type == ELLIPSE else\
        ROIArch   (view=view, pos=pos, **kwa) if roi_type == ARCH else\
        None

    roi_name = dict_roi_type_name[roi_type]
    if o is None:
       logger.warning('ROI of type %s is not defined' % roi_name)
    else:
       logger.info('create new ROI %s in scene position x: %.1f y: %.1f' %\
                   (roi_name, pos.x(), pos.y()))
    return o



def circle_pointe(p, rx=5, ry=5, npoints=8, astart=0, aspan=360):
    start, span = math.radians(astart), math.radians(aspan)
    angs = np.linspace(start, start+span, num=npoints, endpoint=True)
    return [QPointF(p.x()+rx*c, p.y()+ry*s)\
            for s,c in zip(tuple(np.sin(angs)), tuple(np.cos(angs)))]


def size_points_on_scene(view, rsize):
    t = view.transform()
    rx, ry = rsize/t.m11(), rsize/t.m22()
    return QPointF(rx,0), QPointF(0,ry)


class HandleBase(QGraphicsPathItem):
    """See: DragPoint.py"""
    def __init__(self, **kwa):
        self.roi         = kwa.get('roi', None)  # any derrived from ROIBase < ... < QGraphicsItem
        self.pos         = QPointF(kwa.get('pos', QPointF(0,0)))
        self.rsize       = kwa.get('rsize', 7)
        self.handle_name = dict_handle_type_name[self.handle_type]
        self.view        = kwa.get('view', self.roi.view if self.roi is not None else None)
        self.scene       = kwa.get('scene', self.view.scene() if self.view is not None else None)
        QGraphicsPathItem.__init__(self, parent=self.roi)
        self.setPath(self.path())

    def path(self):
        return QGraphicsPathItem.path(self)

    def add_to_scene(self, pen=QPEN_DEF, brush=QBRUSH_DEF):
        """Adds/returns QGraphicsPathItem to scene"""
        #self.scitem = QGraphicsPathItem(path)
        ROIBase.add_to_scene(self, pen, brush)

    def add_to_scene(self, pen=QPEN_DEF, brush=QBRUSH_DEF):
        self.setBrush(brush)
        self.setPen(pen)
        self.scene.addItem(self)
        return self


class HandleCenter(HandleBase):
    def __init__(self, **kwa):
        self.handle_type = CENTER
        HandleBase.__init__(self, **kwa)

    def path(self):
        """Returns QPainterPath for HandleCenter in scene coordinates around self.pos"""
        p, view, rsize = self.pos, self.view, self.rsize
        dx, dy = size_points_on_scene(view, rsize)
        path = QPainterPath(p-dy) # Designers sign of center
        path.lineTo(p+dy)
        path.lineTo(p-dx/5+dy/5)
        path.lineTo(p-dx)
        path.lineTo(p+dx)
        path.lineTo(p)
        path.closeSubpath()
        return path


class HandleOrigin(HandleBase):
    def __init__(self, **kwa):
        self.handle_type = ORIGIN
        HandleBase.__init__(self, **kwa)

    def path(self):
        """Returns QPainterPath for HandleOrigin in scene coordinates around self.pos"""
        p, view, rsize = self.pos, self.view, self.rsize
        dx, dy = size_points_on_scene(view, rsize)
        path = QPainterPath(p+dx) #vertical cross
        path.lineTo(p-dx)
        path.moveTo(p+dy)
        path.lineTo(p-dy)
        path.closeSubpath()
        return path


class HandleTranslate(HandleBase):
    def __init__(self, **kwa):
        self.handle_type = TRANSLATE
        HandleBase.__init__(self, **kwa)

    def path(self):
        """Returns QPainterPath for HandleTranslate in scene coordinates around self.pos"""
        p, view, rsize = self.pos, self.view, self.rsize
        dx, dy = size_points_on_scene(view, rsize)
        path = QPainterPath(p+dx+dy)  #horizantal rectangle
        path.lineTo(p-dx+dy)
        path.lineTo(p-dx-dy)
        path.lineTo(p+dx-dy)
        path.closeSubpath()
        return path


class HandleRotate(HandleBase):
    def __init__(self, **kwa):
        self.handle_type = ROTATE
        HandleBase.__init__(self, **kwa)

    def path(self):
        """Returns QPainterPath for HandleRotate in scene coordinates around self.pos"""
        p, view, rsize = self.pos, self.view, self.rsize
        dx, dy = size_points_on_scene(view, rsize)
        path = QPainterPath() #Circle/Ellipse - shape
        #path.addEllipse(p, dx.x(), dx.x())
        path.addPolygon(QPolygonF(circle_pointe(p, rx=dx.x(), ry=dx.x(), npoints=16))) # astart=0, aspan=360
        path.closeSubpath()
        return path


class HandleScale(HandleBase):
    def __init__(self, **kwa):
        self.handle_type = SCALE
        HandleBase.__init__(self, **kwa)

    def path(self):
        """Returns QPainterPath for Handle in scene coordinates around self.pos"""
        p, view, rsize = self.pos, self.view, self.rsize
        dx, dy = size_points_on_scene(view, rsize)
        path = QPainterPath(p+dx) #rombic - shape#
        path.lineTo(p+dy)
        path.lineTo(p-dx)
        path.lineTo(p-dy)
        path.closeSubpath()
        return path


class HandleMenu(HandleBase):
    def __init__(self, **kwa):
        self.handle_type = MENU
        HandleBase.__init__(self, **kwa)

    def path(self):
        """Returns QPainterPath for HandleMenu in scene coordinates around self.pos"""
        p, view, rsize = self.pos, self.view, self.rsize
        dx, dy = size_points_on_scene(view, rsize)
        dx06, dy03 = 0.6*dx, 0.3*dy
        path = QPainterPath(p+dx+dy)  #horizantal rectangle
        path.lineTo(p-dx+dy)
        path.lineTo(p-dx-dy)
        path.lineTo(p+dx-dy)
        path.lineTo(p+dx+dy)
        path.moveTo(p-dx06+dy03)
        path.lineTo(p+dx06+dy03)
        path.moveTo(p-dx06-dy03)
        path.lineTo(p+dx06-dy03)
        path.closeSubpath()
        return path


class HandleOther(HandleBase):
    def __init__(self, **kwa):
        self.handle_type = OTHER
        self.shhand = kwa.get('shhand', 1)
        HandleBase.__init__(self, **kwa)

    def path(self):
        """Returns QPainterPath for HandleOther in scene coordinates around self.pos"""
        p, view, rsize = self.pos, self.view, self.rsize
        dx, dy = size_points_on_scene(view, rsize)
        path = None

        print('ZZZ shhand:', self.shhand)

        if self.shhand == 1:
            path = QPainterPath(p+dx+dy) # W-M-shape
            path.lineTo(p-dx-dy)
            path.lineTo(p-dx+dy)
            path.lineTo(p+dx-dy)

        elif self.shhand == 2:
            path = QPainterPath(p+dx) # Z-shape
            path.lineTo(p-dx)
            path.lineTo(p+dy)
            path.lineTo(p-dy)

        else:
            path = QPainterPath(p+dx)  #X-shape
            path.lineTo(p+dy)
            path.lineTo(p-dy)
            path.lineTo(p-dx)

        path.closeSubpath()
        return path


def select_handle(handle_type, roi=None, pos=QPointF(1,1), **kwa):
    o = HandleCenter   (roi=roi, pos=pos, **kwa) if handle_type == CENTER else\
        HandleOrigin   (roi=roi, pos=pos, **kwa) if handle_type == ORIGIN else\
        HandleTranslate(roi=roi, pos=pos, **kwa) if handle_type == TRANSLATE else\
        HandleRotate   (roi=roi, pos=pos, **kwa) if handle_type == ROTATE else\
        HandleScale    (roi=roi, pos=pos, **kwa) if handle_type == SCALE else\
        HandleMenu     (roi=roi, pos=pos, **kwa) if handle_type == MENU else\
        HandleOther    (roi=roi, pos=pos, **kwa) if handle_type == OTHER else\
        None

    handle_name = dict_handle_type_name[handle_type]
    if o is None:
       logger.warning('ROI of type %s is not defined' % handle_name)
    else:
       logger.info('create new handle %s in scene position x: %.1f y: %.1f' %\
                   (handle_name, pos.x(), pos.y()))
    return o


#def qtransform_rotate_around_center(center=QPointF(0,0), angle_deg=20):
#    return QTransform().translate(center.x(), center.y()).\
#           rotate(angle_deg).\
#           translate(-center.x(), -center.y())

#        t = roiu.qtransform_rotate_around_center(center=r0.center(), angle_deg=20)
#        p1 = t.mapToPolygon(QRect(r0))
#        self.scitem = self.scene().addRect(QRectF(rect), pen, brush)

# EOF
