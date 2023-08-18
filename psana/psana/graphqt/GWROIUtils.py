
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

import logging
logger = logging.getLogger(__name__)

import math # radians, cos, sin, ceil, pi
import numpy as np

from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QApplication,\
     QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsLineItem, QGraphicsPolygonItem, QGraphicsPathItem
from PyQt5.QtGui import  QPen, QPainter, QColor, QBrush, QCursor, QTransform, QPolygonF,  QPainterPath
from PyQt5.QtCore import Qt, QPoint, QPointF, QRect, QRectF, QSize, QSizeF, QLineF # , pyqtSignal

QCOLOR_DEF = QColor(Qt.yellow)
QCOLOR_SEL = QColor('#ffeeaaee')
QCOLOR_EDI = QColor(Qt.white) # Qt.magenta
QCOLOR_HID = QColor('#00eeaaee')
QBRUSH_DEF = QBrush()
QBRUSH_ROI = QBrush(QCOLOR_DEF, Qt.SolidPattern)
QBRUSH_EDI = QBrush(QCOLOR_EDI, Qt.SolidPattern)
QPEN_DEF   = QPen(QCOLOR_DEF, 1, Qt.SolidLine)  # Qt.DashLine
QPEN_EDI   = QPen(QCOLOR_EDI, 1, Qt.SolidLine)
QPEN_HID   = QPen(QCOLOR_HID, 1, Qt.SolidLine)
QPEN_EDI.setCosmetic(True)
QPEN_HID.setCosmetic(True)

NONE = 0

PIXEL    = 1
LINE     = 2
RECT     = 4
SQUARE   = 8
POLYGON  = 16
POLYREG  = 32
ELLIPSE  = 64
CIRCLE   = 128
ARCH     = 256
PIXGROUP = 512
roi_tuple = (
    (NONE,     'NONE',      '0'),
    (PIXEL,    'PIXEL',     '1'),
    (LINE,     'LINE',      '2'),
    (RECT,     'RECT',      '3'),
    (SQUARE,   'SQUARE',    '4'),
    (POLYGON,  'POLYGON',   '5'),
    (POLYREG,  'POLYREG',   '6'),
    (ELLIPSE,  'ELLIPSE',   '7'),
    (CIRCLE,   'CIRCLE',    '8'),
    (ARCH,     'ARCH',      '9'),
    (PIXGROUP, 'PIXGROUP',  'X'),
)
roi_types = [r[0] for r in roi_tuple]
roi_names = [r[1] for r in roi_tuple]
roi_keys  = [r[2] for r in roi_tuple]
dict_roi_type_name = {t:n for t,n,k in roi_tuple}

UNVISIBLE = 1
VISIBLE   = 2
ADD       = 4
REMOVE    = 8
SELECT    = 16
EDIT      = 32
mode_tuple = (
  (NONE,      'NONE',      'Q'),
  (UNVISIBLE, 'UNVISIBLE', 'U'),
  (VISIBLE,   'VISIBLE',   'V'),
  (ADD,       'ADD',       'A'),
  (REMOVE,    'REMOVE',    'R'),
  (SELECT,    'SELECT',    'S'),
  (EDIT,      'EDIT',      'E'),
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

def regular_polygon_qpoints(p, rx=5, ry=5, npoints=8, astart=0, aspan=360, endpoint=False):
    start, span = math.radians(astart), math.radians(aspan)
    angs = np.linspace(start, start+span, num=npoints, endpoint=endpoint)
    return [QPointF(p.x()+rx*c, p.y()+ry*s)\
            for s,c in zip(tuple(np.sin(angs)), tuple(np.cos(angs)))]

def size_points_on_scene(view, rsize):
    t = view.transform()
    rx, ry = rsize/t.m11(), rsize/t.m22()
    return QPointF(rx,0), QPointF(0,ry)

def angle_between_points(p0, p1):
    d = p1 - p0
    return math.degrees(math.atan2(d.y(), d.x()))

def distance_between_points(p0, p1):
    d = p1 - p0
    x,y = d.x(), d.y()
    return math.sqrt(x*x + y*y)

def items_at_point(scene, point):
    items = scene.items(point)
    logging.debug('sc.itemsAt(%s): %s' % (str(point), str(items)))
    return items

def int_scpos(scpos):
    return None if scpos is None else QPoint(int(scpos.x()), int(scpos.y()))

def json_point_int(p):
    return p.x(), p.y()
    #return '(%d, %d)' % (p.x(), p.y())

def json_point(p, prec=2):
    return round(p.x(), prec), round(p.y(), prec)
    #return  '(%.2f, %.2f)' % (p.x(), p.y())

def rect_to_square(rect, pos):
    dp = pos - rect.topLeft()
    w,h = dp.x(), dp.y()
    v = max(abs(w), abs(h))
    w,h = math.copysign(v,w), math.copysign(v,h)
    rect.setSize(QSizeF(w,h))
    return rect

def cartesian_distance(x, y):
    return np.sqrt(x*x + y*y)


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
        self.tolerance  = kwa.get('tolerance', 5.0)
        self.inverted   = kwa.get('inverted', False)
        self.is_busy_iscpos = kwa.get('is_busy_iscpos', False)
        self.list_of_handles = []

        self.is_finished = False
        self.scitem = None
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

    def add_to_scene(self, pos=None, pen=None, brush=None):
        #self.pos = pos
        item = self.scitem
        item.setPen(self.pen if pen is None else pen)
        if self.roi_type != LINE:
            item.setBrush(self.brush if brush is None else brush)
        self.scene().addItem(item)

    def move_at_add(self, pos, left_is_pressed=False):
        """pos (QPointF) - scene position of the mouse"""
        logging.debug('ROIBase.move_at_add to be re-implemented in subclasses, if necessary')

    def set_point_at_add(self, pos, clicknum):
        logging.debug('ROIBase.set_point_at_add must be re-implemented in SOME of subclasses')

    def is_last_point(self, scpos, clicknum):
        """Returns boolean answer if input is completed.
           Default responce for all 2-click ROIs like line, rect, ellips, square, circle
        """
        if clicknum > 1:
           self.is_finished = True
        return self.is_finished

    def finish_add_roi(self):
        logging.debug('ROIBase.finish_add_roi should be re-implemented if necessary, e.g ROIPolygon')
        self.is_finished = True

    def show_handles(self):
        """in re-implementetion
           - make self.list_of_handles
           - self.add_handles_to_scene()
        """
        logging.info('ROIBase.show_handles TBRe-implemented for ROI %s' % self.roi_name)

    def hide_handles(self):
        logging.debug('ROIBase.hide_handles for ROI %s' % self.roi_name)
        for o in self.list_of_handles:
            o.setPen(QPEN_HID)
            #logger.debug('begin remove handle: %s' % str(o))
        #self.remove_handles_from_scene()

    def add_handles_to_scene(self):
        logging.debug('ROIBase.add_handles_to_scene for ROI %s' % self.roi_name)
        logging.warning('  handles might be already added to scene at instatiation... by parent=...')
        for o in self.list_of_handles:
            #o.add_handle_to_scene(pen=QPEN_EDI)
            o.setPen(QPEN_EDI)

    def remove_handles_from_scene(self):
        """remove handle objects from the self.list_of_handles and scene, reset self.list_of_handles"""
        logging.debug('ROIBase.remove_handles_from_scene for ROI %s' % self.roi_name)
        for o in self.list_of_handles.copy():
            logger.debug('begin remove handle: %s' % str(o))
            self.scene().removeItem(o)
            self.list_of_handles.remove(o)
            #del o
            logger.debug('   -- removed')
        #self.list_of_handles = []

    def handles_at_point(self, p):
        """returns list of Handle objects found at QPointF p"""
        items = self.scene().items(p)
        handles = [o for o in self.list_of_handles if o in items]
        logger.debug('handles_at_point - point %s: list of handles: %s' % (str(p), str(handles)))
        return handles

    def handle_at_point(self, p):
        handles = self.handles_at_point(p)
        logger.debug('handle_at_point - point %s: found handles, return [0]: %s' % (str(p), str(handles)))
        if handles in (None, []): return None
        return None if handles in (None, []) else handles[0]

    def roi_pars(self):
        logging.debug('ROIBase.roi_pars - dict of roi parameters')
        return {'roi_type':self.roi_type,
                'roi_name': self.roi_name,
                'points':[]
                }
                #'angle_deg': self.scitem.rotation(),

    def set_from_roi_pars(self, d):
        logging.warning('ROIBase.set_from_roi_pars - dict of roi parameter NEEEDS TO BE RE-EMPLEMENTED IN DERIVED SUBCLASS')
        return False

    def set_point(self, n, p):
        """HandleBase.on_move calls this method"""
        logging.warning('TB RE-IMPLEMENTED ROIBase.set_point - set point:%d to point %s' % (n,p))

    def cr_meshgrid(self, shape):
        logging.debug('XXX cr_meshgrid shape %s' % str(shape))
        return np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    def good_bad_pixels(self):
        """accounting inversion"""
        return (False, True) if self.inverted else (True, False)

    def mask(self, shape):
        """generic mask for roi using self.scitem.contains(QPointF)"""
        x, y = self.cr_meshgrid(shape)
        list_xy = list(zip(x.ravel(), y.ravel()))
        # mask = np.array(Path(poly_verts).contains_points(ij))
        print('list_xy:', list_xy[:10])
        cond = np.array([self.scitem.contains(QPointF(*xy)) for xy in list_xy])
        cond.shape = shape
        good, bad = self.good_bad_pixels()
        return np.select([cond], [bad], default=good)


class ROIPixel(ROIBase):
    def __init__(self, **kwa):
        self.roi_type = PIXEL
        ROIBase.__init__(self, **kwa)
        self.pos = int_scpos(self.pos)
        logger.debug('ROIPixel.__init__')
        self.brush = QBRUSH_ROI
        self.pen = QPEN_DEF
        print('ROIPixel.roi_pars: %s' % str(self.roi_pars()))

    def pixel_rect(self, pos=None):
        if pos is not None: self.pos = int_scpos(pos)
        return QRectF(QPointF(self.pos), QSizeF(1,1))

    def add_to_scene(self, pos=None):
        """Adds pixel as a rect to scene"""
        logger.debug('ROIPixel.add_to_scene')
        if not self.is_busy_iscpos:
            self.scitem = QGraphicsRectItem(self.pixel_rect(pos))
            ROIBase.add_to_scene(self, pen=self.pen, brush=self.brush)
        self.finish_add_roi()

    def is_last_point(self, scpos, clicknum):
        return True

    def roi_pars(self):
        d = ROIBase.roi_pars(self)
        d['points'] = [json_point_int(self.pos),]
        return d

    def set_from_roi_pars(self, d):
        logger.info('ROIPixel.set_from_roi_pars dict: %s' % str(d))
        self.add_to_scene() # x,y = d['points'][0]
        return True


class ROIPixGroup(ROIBase):
    def __init__(self, **kwa):
        self.roi_type = PIXGROUP
        ROIBase.__init__(self, **kwa)
        self.pos = int_scpos(self.pos)
        logger.info('ROIPixGroup.__init__')
        self.brush = QBRUSH_ROI
        self.pen = QPEN_DEF
        self.pixpos = []
        self.iscpos_last = None

    def pixel_rect(self, pos=None):
        if pos is not None: self.pos = pos
        self.pixpos.append(self.pos)
        return QRectF(QPointF(self.pos), QSizeF(1,1))

    def is_busy_pos(self, iscpos=None):
        is_busy = any([iscpos == p for p in self.pixpos])
        return is_busy

    def remove_pixel_from_path(self, iscpos):
        """removes pixel position from the list and from the path."""
        self.pixpos.remove(iscpos)
        path = QPainterPath()
        tuple_pos = tuple(self.pixpos)
        self.pixpos = []
        for p in tuple_pos:
            path.addRect(self.pixel_rect(p))
        self.scitem.setPath(path)

    def add_to_scene(self, pos=None):
        """Adds pixel as a rect to scene"""
        iscpos = int_scpos(pos)
        #logger.info('ROIPixGroup.add_to_scene iscpos: %s iscpos_last: %s' % (str(iscpos), (self.iscpos_last)))

        if iscpos == self.iscpos_last\
        or self.is_busy_iscpos:
           #print('XXX ROIPixGroup.add_to_scene - DO NOT ADD - return conditional')
           return

        self.iscpos_last = iscpos

        if self.scitem is None:
           #self.set_color_pen_brush(color=QColor('#ffaaffee'))
           path = QPainterPath()
           path.addRect(self.pixel_rect(iscpos))
           self.scitem = QGraphicsPathItem(path)
           ROIBase.add_to_scene(self, pen=self.pen, brush=self.brush)
           #print('XXX ROIPixGroup.add_to_scene - pixel self.scitem is created and added to scene at 1st click')
        else:
           if self.is_busy_pos(iscpos):
               print('XXX ROIPixGroup   is_busy_pos:', iscpos)
               self.remove_pixel_from_path(iscpos)
               return
           item = self.scitem
           #item.setPen(self.pen)
           #item.setBrush(self.brush)
           path = item.path()
           path.addRect(self.pixel_rect(iscpos))
           item.setPath(path)

    def set_point_at_add(self, pos, clicknum):
        self.add_to_scene(pos)

    def move_at_add(self, pos, left_is_pressed=False):
        #logger.debug('ROIPixGroup.move_at_add')
        if left_is_pressed:
            self.add_to_scene(pos=pos)

    def is_last_point(self, p, clicknum, clicknum_max=200):
        """returns boolean answer if input is completed"""
        return clicknum > clicknum_max

    def roi_pars(self):
        d = ROIBase.roi_pars(self)
        d['points'] = [json_point_int(p) for p in self.pixpos]
        return d

    def set_from_roi_pars(self, d):
        logger.info('ROIPixGroup.set_from_roi_pars dict: %s' % str(d))
        for x,y in d['points']:
            self.add_to_scene(pos=QPoint(x,y))
        return True


class ROILine(ROIBase):

    def __init__(self, **kwa):
        self.roi_type = LINE
        ROIBase.__init__(self, **kwa)

    def add_to_scene(self, pos=None, line=None, pen=None):
        """Adds QGraphicsLineItem to scene"""
        if line is None:
            t = self.tolerance
            line = QLineF(self.pos, self.pos+QPointF(t,t))
        self.scitem = QGraphicsLineItem(QLineF(line))
        ROIBase.add_to_scene(self, pen=pen, brush=None)
        #self.scitem = self.scene().addLine(QLineF(line), pen, brush)

    def move_at_add(self, pos, left_is_pressed=False):
        #logger.debug('ROILine.move_at_add')
        line = self.scitem.line()
        line.setP2(pos)
        self.scitem.setLine(line)

    def roi_pars(self):
        o = self.scitem.line()
        d = ROIBase.roi_pars(self)
        d['points'] = [json_point(p) for p in (o.p1(), o.p2())]
        return d

    def set_from_roi_pars(self, d):
        logger.info('ROILine.set_from_roi_pars dict: %s' % str(d))
        p1, p2 = [QPointF(*xy) for xy in d['points']]
        self.add_to_scene(pos=p1, line=QLineF(p1, p2))
        return True

    def show_handles(self):
        logging.info('ROILine.show_handles for ROI %s' % self.roi_name)
        o = self.scitem.line()
        self.list_of_handles = [
            select_handle(TRANSLATE, view=self.view, roi=self, pos=o.p1(), poinum=0),
            select_handle(TRANSLATE, view=self.view, roi=self, pos=o.p2(), poinum=1),
        ]
        self.add_handles_to_scene()

    def set_point(self, n, p):
        #logging.debug('ROILine.set_point - set point number: %d to position: %s' % (n, str(p)))
        line = self.scitem.line()
        if   n==0: line.setP1(p)
        elif n==1: line.setP2(p)
        else: return
        self.scitem.setLine(line)


class ROIRect(ROIBase):
    def __init__(self, **kwa):
        self.roi_type = RECT
        ROIBase.__init__(self, **kwa)
        self.angle = 0

    def add_to_scene(self, pos=None, rect=None, pen=None, brush=None, angle_deg=0):
        """Adds QGraphicsRectItem to scene"""
        if rect is None:
            t = self.tolerance
            rect = QRectF(self.pos, QSizeF(t,t))
        item = self.scitem = QGraphicsRectItem(QRectF(rect))
        item.setTransformOriginPoint(item.rect().topLeft())  # center()
        if angle_deg != 0:
            self.angle = angle_deg
            item.setRotation(angle_deg)
        ROIBase.add_to_scene(self, pen=pen, brush=brush)

    def move_at_add(self, pos, left_is_pressed=False):
        #logger.debug('ROIRect.move_at_add')
        rect = self.scitem.rect()
        rect.setBottomRight(pos)
        self.scitem.setRect(rect)

    def roi_pars(self):
        o = self.scitem.rect()
        d = ROIBase.roi_pars(self)
        d['angle'] = self.angle
        d['points'] = [json_point(p) for p in (o.topLeft(), o.bottomRight())]  #  list(o.getCoords())
        return d

    def set_from_roi_pars(self, d):
        logger.info('ROIRect.set_from_roi_pars dict: %s' % str(d))
        p1, p2 = [QPointF(*xy) for xy in d['points']]
        self.add_to_scene(pos=p1, rect=QRectF(p1, p2), angle_deg=d['angle'])
        return True

    def show_handles(self):
        logging.info('ROIRect.show_handles for ROI %s' % self.roi_name)
        o = self.scitem.rect()
        self.list_of_handles = [
            select_handle(TRANSLATE, view=self.view, roi=self, pos=o.topLeft(),     poinum=0),
            select_handle(SCALE,     view=self.view, roi=self, pos=o.bottomRight(), poinum=1),
            select_handle(ROTATE,    view=self.view, roi=self, pos=o.topRight(),    poinum=2),
        ]
        self.add_handles_to_scene()

    def set_point(self, n, p):
        #logging.debug('ROIRect.set_point - set point number: %d to position: %s' % (n, str(p)))
        h0, h1, h2 = self.list_of_handles
        r = self.scitem.rect()
        self.scitem.setTransformOriginPoint(r.topLeft())
        pt = self.scitem.mapFromScene(p)
        if n==0:
            r.moveTo(pt)
            h0.set_handle_pos(pt)
        elif n==1: r.setBottomRight(pt)
        elif n==2:
            self.angle = angle_between_points(r.topLeft(), p)
            self.scitem.setRotation(self.angle)
        else: return
        self.scitem.setRect(r)
        h1.set_handle_pos(r.bottomRight())
        h2.set_handle_pos(r.topRight())


class ROISquare(ROIRect):
    def __init__(self, **kwa):
        ROIRect.__init__(self, **kwa)
        self.roi_type = SQUARE
        self.roi_name = dict_roi_type_name[self.roi_type]

    def move_at_add(self, pos, left_is_pressed=False):
        #logger.debug('ROISquare.move_at_add')
        self.scitem.setRect(rect_to_square(self.scitem.rect(), pos))

    def show_handles(self):
        logging.info('ROISquare.show_handles for ROI %s' % self.roi_name)
        o = self.scitem.rect()
        self.list_of_handles = [
            select_handle(TRANSLATE, view=self.view, roi=self, pos=o.topLeft(), poinum=0),
            select_handle(ROTATE,    view=self.view, roi=self, pos=o.bottomRight(), poinum=1),
        ]
        self.add_handles_to_scene()

    def set_point(self, n, p):
        #logging.debug('ROISquare.set_point - set point number: %d to position: %s' % (n, str(p)))
        h0, h1 = self.list_of_handles
        r = self.scitem.rect()
        self.scitem.setTransformOriginPoint(r.topLeft())
        pt = self.scitem.mapFromScene(p)
        if n==0:
            r.moveTo(pt)
            h0.set_handle_pos(pt)
        elif n==1:
            self.angle = angle_between_points(r.topLeft(), p) - 45
            #logging.debug('rotation angle: %.2f' % self.angle)
            self.scitem.setRotation(self.angle)
            r.setBottomRight(pt)
        else: return
        sq = rect_to_square(r, r.bottomRight())
        self.scitem.setRect(sq)
        h1.set_handle_pos(sq.bottomRight())


class ROIPolygon(ROIBase):
    def __init__(self, **kwa):
        self.roi_type = POLYGON
        ROIBase.__init__(self, **kwa)

    def add_to_scene(self, pos=None, poly=None, pen=QPEN_DEF, brush=QBRUSH_DEF):
        """Adds QGraphicsPolygonItem to scene"""
        if poly is None: poly = QPolygonF((self.pos,)) # self.pos+QPointF(10,0), self.pos+QPointF(0,10)))
        self.scitem = QGraphicsPolygonItem(QPolygonF(poly))
        ROIBase.add_to_scene(self, pen=pen, brush=brush)
        #self.scitem = self.scene().addPolygon(QPolygonF(poly), pen, brush)

    def move_at_add(self, pos, left_is_pressed=False):
        self.move_vertex(pos)

    def move_vertex(self, pos):
        poly = self.scitem.polygon()
        i = poly.size()
        logger.debug('polygon size: %d' % i)
        if i==1: poly.append(pos)
        else:    poly.replace(i-1, pos)
        self.scitem.setPolygon(poly)

    def add_vertex(self, pos):
        poly = self.scitem.polygon()
        poly.append(pos)
        self.poly_selected = poly
        self.scitem.setPolygon(poly)

    def set_point_at_add(self, pos, clicknum):
        self.add_vertex(pos)

    def is_last_point(self, scpos, clicknum):
        d = (scpos - self.pos).manhattanLength()
        logging.info('POLYGON manhattanLength(last-first): %.1f closing distance: %.1f'%\
                      (d, self.tolerance))
        return clicknum > 2 and d < self.tolerance

    def finish_add_roi(self):
        if not self.is_finished:
            self.set_poly()
            self.is_finished = True

    def set_poly(self, poly=None):
        self.scitem.setPolygon(self.poly_selected if poly==None else poly)

    def polygon_points(self):
        """returns a list of QPointsF for polygon vertices"""
        o = self.scitem.polygon()
        return [o.value(i) for i in range(o.size())]

    def roi_pars(self):
        d = ROIBase.roi_pars(self)
        d['points'] = [json_point(p) for p in self.polygon_points()]
        return d

    def set_from_roi_pars(self, d):
        logger.info('ROIPolygon.set_from_roi_pars dict: %s' % str(d))
        pxy = [QPointF(*xy) for xy in d['points']]
        self.add_to_scene(pos=pxy[0], poly=QPolygonF(pxy))
        return True

    def show_handles(self):
        logging.info('ROIPolygon.show_handles for ROI %s' % self.roi_name)
        self.list_of_handles = [
            select_handle(TRANSLATE, view=self.view, roi=self, pos=p, poinum=i) for i,p in enumerate(self.polygon_points())]
        self.add_handles_to_scene()

    def set_point(self, n, p):
        #logging.debug('ROIPolygon.set_point - set point number: %d to position: %s' % (n, str(p)))
        poly = self.scitem.polygon()
        if n>poly.size(): return
        poly.replace(n, p)
        self.scitem.setPolygon(poly)


class ROIPolyreg(ROIBase):
    def __init__(self, **kwa):
        self.roi_type = POLYREG
        ROIBase.__init__(self, **kwa)
        self.radius = None # kwa.get('radius', 5.0)
        self.angle  = None # kwa.get('angle', 0)
        self.nverts = 3 # kwa.get('angle', 3)
        self.clicknum = None
        self.scpos_rad = None

    def add_to_scene(self, pos=None, poly=None, pen=QPEN_DEF, brush=QBRUSH_DEF):
        """Adds QGraphicsPolygonItem to scene"""
        self.pos = pos
        t = self.tolerance
        poly = QPolygonF(regular_polygon_qpoints(self.pos, rx=t, ry=t, npoints=self.nverts)) if poly is None\
               else poly  # astart=0, aspan=360
        self.scitem = QGraphicsPolygonItem(poly)
        ROIBase.add_to_scene(self, pen=pen, brush=brush)

    def polyreg_dxy(self, pos):
        d = pos - self.pos
        x, y = d.x(), d.y()
        return d, x, y

    def move_at_add(self, scpos, left_is_pressed=False):
        #logger.debug('ROILine.move_at_add')
        d, x, y = self.polyreg_dxy(scpos)
        angle = math.degrees(math.atan2(y, x)) if self.angle is None else self.angle
        r = math.sqrt(x*x + y*y) if self.radius is None else self.radius
        if self.clicknum != 3: self.set_nverts(scpos)
        poly = QPolygonF(regular_polygon_qpoints(self.pos, rx=r, ry=r, npoints=self.nverts, astart=angle)) # aspan=360
        self.scitem.setPolygon(poly)

    def set_radius_and_angle(self, scpos):
        self.scpos_rad = scpos
        d, x, y = self.polyreg_dxy(scpos)
        self.pradius = QPointF(x, y)
        self.radius = math.sqrt(x*x + y*y)
        self.angle = math.degrees(math.atan2(y, x))

    def set_nverts(self, scpos):
        if self.scpos_rad is not None:
            d = (scpos-self.scpos_rad).manhattanLength()
            self.nverts = 3 + int(16*d/self.radius)

    def set_point_at_add(self, p, clicknum):
        self.clicknum = clicknum
        if   clicknum == 2: self.set_radius_and_angle(p)
        elif clicknum == 3: self.set_nverts(p)

    def is_last_point(self, p, clicknum):
        """returns boolean answer if input is completed"""
        if clicknum > 2:
           self.is_finished = True
        return self.is_finished

    def roi_pars(self):
        d = ROIBase.roi_pars(self)
        d['points'] = [json_point(p) for p in (self.pos, self.pradius)]
        d['nverts'] = self.nverts
        d['radius'] = self.radius
        d['angle']  = self.angle
        return d

    def set_from_roi_pars(self, d):
        logger.info('ROIPolyreg.set_from_roi_pars dict: %s' % str(d))
        pos, pradius = [QPointF(*xy) for xy in d['points']]
        nverts = self.nverts = d['nverts']
        radius = self.radius = d['radius']
        angle  = self.angle  = d['angle']
        self.pradius = pradius
        poly = QPolygonF(regular_polygon_qpoints(pos, rx=radius, ry=radius, npoints=nverts, astart=angle))
        self.add_to_scene(pos=pos, poly=poly)
        return True

    def show_handles(self):
        logging.info('ROIPolyreg.show_handles for ROI %s' % self.roi_name)
        self.list_of_handles = [
            select_handle(TRANSLATE, view=self.view, roi=self, pos=self.pos, poinum=0),
            select_handle(ROTATE,    view=self.view, roi=self, pos=self.scpos_rad, poinum=1),
          #select_handle(MENU,      view=self.view, roi=self, pos=o.topRight()),
        ]
        self.add_handles_to_scene()


    def set_point(self, n, p):
        #logging.debug('ROIPolyreg.set_point - set point number: %d to position: %s' % (n, str(p)))
        h0, h1 = self.list_of_handles
        if n==0:
            self.pos = p   #r.moveTo(p)  # rect.moveCenter(p)
        elif n==1:
            self.set_radius_and_angle(p)
        else: return
        self.move_at_add(p)
        h1.set_handle_pos(self.pos + self.pradius)
        #poly = self.scitem.polygon()
        #self.scitem.setPolygon(poly)


class ROIEllipse(ROIBase):

    def __init__(self, **kwa):
        self.roi_type = ELLIPSE
        ROIBase.__init__(self, **kwa)
        self.angle = 0

    def add_to_scene(self, pos=None, rect=None, pen=QPEN_DEF, brush=QBRUSH_DEF,\
                     angle_deg=0, start_angle=0, span_angle=360):
        """Adds QGraphicsEllipseItem to scene."""
        if rect is None:
            t = self.tolerance
            rect = QRectF(self.pos-QPointF(t/2,t/2), QSizeF(t,t))
            rect.moveCenter(self.pos)
        item = self.scitem = QGraphicsEllipseItem(QRectF(rect))
        item.setStartAngle(start_angle*16)
        item.setSpanAngle(span_angle*16)
        item.setTransformOriginPoint(self.pos) # item.rect().center())
        item.setRotation(angle_deg)
        ROIBase.add_to_scene(self, pen=pen, brush=brush)
        #self.scitem = self.scene().addEllipse(QRectF(rect), pen, brush)

    def move_at_add(self, pos, left_is_pressed=False):
        #logger.debug('ROIEllipse.move_at_add')
        c = self.pos
        dp = pos-c # rect.center()
        self.scitem.setRect(QRectF(c-dp, c+dp))

    def roi_pars(self):
        o = self.scitem.rect()
        d = ROIBase.roi_pars(self)
        d['points'] = [json_point(p) for p in (o.topLeft(), o.bottomRight())]  # list(o.getCoords())
        return d

    def set_from_roi_pars(self, d):
        logger.info('ROIEllipse.set_from_roi_pars dict: %s' % str(d))
        p1, p2 = [QPointF(*xy) for xy in d['points']]
        self.add_to_scene(pos=p1, rect=QRectF(p1, p2))
        return True

    def show_handles(self):
        logging.info('ROIEllipse.show_handles for ROI %s' % self.roi_name)
        o = self.scitem.rect()
        self.list_of_handles = [
            select_handle(TRANSLATE, view=self.view, roi=self, pos=o.center(),      poinum=0),
            select_handle(SCALE,     view=self.view, roi=self, pos=o.bottomRight(), poinum=1),
            select_handle(ROTATE,    view=self.view, roi=self, pos=o.topRight(),    poinum=2),
        ]
        self.add_handles_to_scene()

    def set_point(self, n, p):
        #logging.debug('ROIEllipse.set_point - set point number: %d to position: %s' % (n, str(p)))
        h0, h1, h2 = self.list_of_handles
        r = self.scitem.rect()
        self.scitem.setTransformOriginPoint(r.center())
        pt = self.scitem.mapFromScene(p)
        if n==0: r.moveCenter(pt)
        elif n==1: r.setBottomRight(pt)
        elif n==2:
            self.angle = angle_between_points(r.topLeft(), p)
            self.scitem.setRotation(self.angle)
        else: return
        self.scitem.setRect(r)
        h0.set_handle_pos(r.center())
        h1.set_handle_pos(r.bottomRight())
        h2.set_handle_pos(r.topRight())


class ROICircle(ROIEllipse):
    def __init__(self, **kwa):
        ROIEllipse.__init__(self, **kwa)
        self.roi_type = CIRCLE
        self.roi_name = dict_roi_type_name[self.roi_type]

    def move_at_add(self, pos, left_is_pressed=False):
        #logger.debug('ROIEllipse.move_at_add')
        c = self.pos # center
        d = pos-c
        d = max(d.x(), d.y())
        dp = QPointF(d,d)
        self.scitem.setRect(QRectF(c-dp, c+dp))

    def show_handles(self):
        logging.info('ROICircle.show_handles for ROI %s' % self.roi_name)
        o = self.scitem.rect()
        self.list_of_handles = [
            select_handle(TRANSLATE, view=self.view, roi=self, pos=o.center(), poinum=0),
            select_handle(SCALE,     view=self.view, roi=self, pos=QPointF(o.right(), o.center().y()), poinum=1),
        ]
        self.add_handles_to_scene()

    def set_point(self, n, p):
        #logging.debug('ROICircle.set_point - set point number: %d to position: %s' % (n, str(p)))
        h0, h1 = self.list_of_handles
        r = self.scitem.rect()
        if n==0:
            d = p - r.center()
            r.moveCenter(p)
            h0.set_handle_pos(p)
            h1.set_handle_pos(h1.hpos + d)
        elif n==1:
            c = r.center()
            d = distance_between_points(c, p)
            rp = QPointF(d,d)
            r.setBottomRight(c+rp)
            r.setTopLeft(c-rp)
            h1.set_handle_pos(p)
        else: return
        self.scitem.setRect(r)

    def mask(self, shape):
        r = self.scitem.rect()
        r0 = r.width()/2
        c = r.center()
        x, y = self.cr_meshgrid(shape)
        r = self.scitem.rect()
        d = cartesian_distance(x-c.x(), y-c.y())
        good, bad = self.good_bad_pixels()
        return np.select([d>r0], [good], default=bad)
        #return np.select([d>r0], [False], default=True)


class ROIArch(ROIBase):
    def __init__(self, **kwa):
        self.roi_type = ARCH
        ROIBase.__init__(self, **kwa)
        self.npoints  = kwa.get('npoints', 32)

    def add_to_scene(self, pos=None, rect=None, pen=QPEN_DEF, brush=QBRUSH_DEF,\
                     angle_deg=0, start_angle=0, span_angle=360):
        """Adds QGraphicsPathItem to scene."""
        logger.info('ROIArch.add_to_scene on 1-st click center: %s' % str(self.pos))
        self.clicknum = 1
        t = self.tolerance
        self.set_p1(self.pos + QPointF(t, t))
        self.set_p2(self.pos + QPointF(t, 0))
        self.scitem = QGraphicsPathItem(self.path())
        ROIBase.add_to_scene(self, pen=pen, brush=brush)

    def point_vraxy(self, p):
        v = p - self.pos # defines v relative center
        x, y = v.x(), v.y()
        r = math.sqrt(x*x + y*y)
        a = math.degrees(math.atan2(y, x))
        return v, r, a, x, y

    def set_point_at_add(self, p, clicknum):
        self.clicknum = clicknum
        if   clicknum == 2: self.set_p1(p)
        elif clicknum == 3: self.set_p2(p)

    def set_p1(self, p):
        self.p1 = p
        self.v1, self.r1, self.a1, self.x1, self.y1 = self.point_vraxy(p)

    def set_p2(self, p):
        self.p2 = p
        self.v2, self.r2, self.a2, self.x2, self.y2 = self.point_vraxy(p)

    def path(self):
        """Returns QPainterPath for ROIArch in scene coordinates around self.pos"""
        p0, r1, r2, a1, a2 = self.pos, self.r1, self.r2, self.a1, self.a2
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        span = math.degrees(math.atan2(x1*y2 - y1*x2, x1*x2 + y1*y2))
        if span<0: span += 360
        path = QPainterPath()
        nverts = int(self.npoints * (abs(span)//90+1))
        self.poly_qpoints = \
               regular_polygon_qpoints(p0, rx=r1, ry=r1, npoints=nverts, astart=a1, aspan=span, endpoint=True)\
             + regular_polygon_qpoints(p0, rx=r2, ry=r2, npoints=nverts, astart=a1+span, aspan=-span, endpoint=True)
        path.addPolygon(QPolygonF(self.poly_qpoints))
        path.closeSubpath()
        return path

#    def boundingRect(self):
#        #self.prepareGeometryChange()
#        p, r = self.hpos, self.rsize
#        v = QPointF(r,r)
#        return QRectF(p-v,p+v)

    def move_at_add(self, p, left_is_pressed=False):
        #logger.debug('ROIArch.move_at_add')
        if (p-self.pos).manhattanLength() < self.tolerance: return
        if   self.clicknum == 1: self.set_p1(p)
        elif self.clicknum == 2: self.set_p2(p)
        self.scitem.setPath(self.path())

    def is_last_point(self, p, clicknum):
        """returns boolean answer if input is completed"""
        if clicknum > 2:
           self.is_finished = True
        return self.is_finished

    def roi_pars(self):
        d = ROIBase.roi_pars(self)
        d['points'] = [json_point(p) for p in (self.pos, self.p1, self.p2)]
        return d

    def set_from_roi_pars(self, d):
        logger.info('ROIArch.set_from_roi_pars dict: %s' % str(d))
        p0, p1, p2 = [QPointF(*xy) for xy in d['points']]
        self.add_to_scene(pos=p0)
        self.set_p1(p1)
        self.set_p2(p2)
        self.clicknum = 3
        self.is_finished = True
        self.scitem.setPath(self.path())
        return True

    def show_handles(self):
        logging.info('ROIArch.show_handles for ROI %s' % self.roi_name)
        self.list_of_handles = [
            select_handle(CENTER,    view=self.view, roi=self, pos=self.pos, poinum=0),
            select_handle(TRANSLATE, view=self.view, roi=self, pos=self.p1, poinum=1),
            select_handle(TRANSLATE, view=self.view, roi=self, pos=self.p2, poinum=2),
        ]
        self.add_handles_to_scene()

    def set_point(self, n, p):
        #logging.debug('ROIArch.set_point - set point number: %d to position: %s' % (n, str(p)))
        h0, h1, h2 = self.list_of_handles
        if n==0:
            d = p - self.pos
            self.pos = p
            h0.set_handle_pos(p)
            h1.set_handle_pos(h1.hpos + d)
            h2.set_handle_pos(h2.hpos + d)
        elif n==1:
            d = p - self.p1
            self.set_p1(p)
            h1.set_handle_pos(p)
        elif n==2:
            d = p - self.p2
            self.set_p2(p)
            h2.set_handle_pos(p)
        else: return
        self.scitem.setPath(self.path())


    def mask(self, shape):
        """generic mask for roi using self.scitem.contains(QPointF)"""
        x, y = self.cr_meshgrid(shape)
        list_xy = list(zip(x.ravel(), y.ravel()))
        #poly_verts = self.scitem.polygon().data() # self.scitemit is path, not a polygon
        poly_verts = [(p.x(), p.y()) for p in self.poly_qpoints]
        logging.debug('poly_verts: %s' % str(poly_verts))
        from matplotlib.path import Path
        cond = np.array(Path(poly_verts).contains_points(list_xy))
        cond.shape = shape
        #print('list_xy:', list_xy[:10])
        ##cond = np.array([self.scitem.contains(QPointF(*xy)) for xy in list_xy])
        ##cond.shape = shape
        good, bad = self.good_bad_pixels()
        return np.select([cond], [bad], default=good)



def create_roi(roi_type, view=None, pos=QPointF(1,1), **kwa):
    o = ROIPixel   (view=view, pos=pos, **kwa) if roi_type == PIXEL else\
        ROILine    (view=view, pos=pos, **kwa) if roi_type == LINE else\
        ROIRect    (view=view, pos=pos, **kwa) if roi_type == RECT else\
        ROISquare  (view=view, pos=pos, **kwa) if roi_type == SQUARE else\
        ROIPolygon (view=view, pos=pos, **kwa) if roi_type == POLYGON else\
        ROIPolyreg (view=view, pos=pos, **kwa) if roi_type == POLYREG else\
        ROICircle  (view=view, pos=pos, **kwa) if roi_type == CIRCLE else\
        ROIEllipse (view=view, pos=pos, **kwa) if roi_type == ELLIPSE else\
        ROIArch    (view=view, pos=pos, **kwa) if roi_type == ARCH else\
        ROIPixGroup(view=view, pos=pos, **kwa) if roi_type == PIXGROUP else\
        None

    roi_name = dict_roi_type_name[roi_type]
    if o is None:
       logger.warning('ROI of type %s is not defined' % roi_name)
    else:
       logger.info('create new ROI %s in scene position x: %.1f y: %.1f' %\
                   (roi_name, pos.x(), pos.y()))
    return o


class HandleBase(QGraphicsPathItem):
    """See: DragPoint.py"""
    def __init__(self, **kwa):
        self.roi    = kwa.get('roi', None)  # any derrived from ROIBase < ... < QGraphicsItem
        self.hpos   = QPointF(kwa.get('pos', QPointF(0,0)))
        self.rsize  = kwa.get('rsize', 7)
        self.hname  = dict_handle_type_name[self.htype]
        self.hview  = kwa.get('view', None) # self.roi.view if self.roi is not None else None)
        self.hscene = kwa.get('scene', self.hview.scene() if self.hview is not None else None)
        self.hcolor = kwa.get('color', QCOLOR_EDI)
        self.hpen   = kwa.get('pen', QPEN_EDI)
        self.poinum = kwa.get('poinum', None)

        QGraphicsPathItem.__init__(self, parent=None if self.roi is None else self.roi.scitem)
        #self.setPos(self.hpos)
        self.setPath(self.path())

    def path(self):
        return QGraphicsPathItem.path(self)

    def add_handle_to_scene(self, pen=QPEN_EDI, brush=QBRUSH_DEF):
        self.setBrush(brush)
        self.setPen(pen)
        #self.hscene.addItem(self)  # ALREDAY SET ??? by parent=self.roi
        return self

    def boundingRect(self):
        #self.prepareGeometryChange()
        p, r = self.hpos, self.rsize
        v = QPointF(r,r)
        return QRectF(p-v,p+v)

    def set_handle_pos(self, p):
        #d = p - self.hpos
        self.hpos = p
        self.setPath(self.path()) # WORKS
        #self.setPos(p) # DOES NOT WORK
        #self.translate(d.x(), d.y())
        #print('XXX dir(self):', dir(self))

    def on_move(self, p):
        #print('XXX on_move p: %s' % str(p))
        self.set_handle_pos(p)
        self.roi.set_point(self.poinum, p)


class HandleCenter(HandleBase):
    def __init__(self, **kwa):
        self.htype = CENTER
        HandleBase.__init__(self, **kwa)

    def path(self):
        """Returns QPainterPath for HandleCenter in scene coordinates around self.hpos"""
        p, view, rsize = self.hpos, self.hview, self.rsize
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
        self.htype = ORIGIN
        HandleBase.__init__(self, **kwa)

    def path(self):
        """Returns QPainterPath for HandleOrigin in scene coordinates around self.hpos"""
        p, view, rsize = self.hpos, self.hview, self.rsize
        dx, dy = size_points_on_scene(view, rsize)
        path = QPainterPath(p+dx) #vertical cross
        path.lineTo(p-dx)
        path.moveTo(p+dy)
        path.lineTo(p-dy)
        path.closeSubpath()
        return path


class HandleTranslate(HandleBase):
    def __init__(self, **kwa):
        self.htype = TRANSLATE
        HandleBase.__init__(self, **kwa)

    def path(self):
        """Returns QPainterPath for HandleTranslate in scene coordinates around self.hpos"""
        p, view, rsize = self.hpos, self.hview, self.rsize
        dx, dy = size_points_on_scene(view, rsize)
        path = QPainterPath(p+dx+dy)  #horizantal rectangle
        path.lineTo(p-dx+dy)
        path.lineTo(p-dx-dy)
        path.lineTo(p+dx-dy)
        path.closeSubpath()
        return path


class HandleRotate(HandleBase):
    def __init__(self, **kwa):
        self.htype = ROTATE
        HandleBase.__init__(self, **kwa)

    def path(self):
        """Returns QPainterPath for HandleRotate in scene coordinates around self.hpos"""
        p, view, rsize = self.hpos, self.hview, self.rsize
        dx, dy = size_points_on_scene(view, rsize)
        path = QPainterPath() #Circle/Ellipse - shape
        #path.addEllipse(p, dx.x(), dx.x())
        path.addPolygon(QPolygonF(regular_polygon_qpoints(p, rx=dx.x(), ry=dx.x(), npoints=16))) # astart=0, aspan=360
        path.closeSubpath()
        return path


class HandleScale(HandleBase):
    def __init__(self, **kwa):
        self.htype = SCALE
        HandleBase.__init__(self, **kwa)

    def path(self):
        """Returns QPainterPath for Handle in scene coordinates around self.hpos"""
        p, view, rsize = self.hpos, self.hview, self.rsize
        dx, dy = size_points_on_scene(view, 1.41*rsize)
        path = QPainterPath(p+dx) #rombic - shape#
        path.lineTo(p+dy)
        path.lineTo(p-dx)
        path.lineTo(p-dy)
        path.closeSubpath()
        return path


class HandleMenu(HandleBase):
    def __init__(self, **kwa):
        self.htype = MENU
        HandleBase.__init__(self, **kwa)

    def path(self):
        """Returns QPainterPath for HandleMenu in scene coordinates around self.hpos"""
        p, view, rsize = self.hpos, self.hview, self.rsize
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
        self.htype = OTHER
        self.shhand = kwa.get('shhand', 1)
        HandleBase.__init__(self, **kwa)

    def path(self):
        """Returns QPainterPath for HandleOther in scene coordinates around self.hpos"""
        p, view, rsize = self.hpos, self.hview, self.rsize
        dx, dy = size_points_on_scene(view, rsize)
        path = None

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


def select_handle(htype, roi=None, pos=QPointF(1,1), **kwa):
    _roi = roi
    o = HandleCenter   (roi=_roi, pos=pos, **kwa) if htype == CENTER else\
        HandleOrigin   (roi=_roi, pos=pos, **kwa) if htype == ORIGIN else\
        HandleTranslate(roi=_roi, pos=pos, **kwa) if htype == TRANSLATE else\
        HandleRotate   (roi=_roi, pos=pos, **kwa) if htype == ROTATE else\
        HandleScale    (roi=_roi, pos=pos, **kwa) if htype == SCALE else\
        HandleMenu     (roi=_roi, pos=pos, **kwa) if htype == MENU else\
        HandleOther    (roi=_roi, pos=pos, **kwa) if htype == OTHER else\
        None

    hname = dict_handle_type_name[htype]
    if o is None:
       logger.warning('ROI of type %s is not defined' % hname)
    else:
       logger.info('create new handle %s in scene position x: %.1f y: %.1f' %\
                   (hname, pos.x(), pos.y()))
    return o

# EOF
