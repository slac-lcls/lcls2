
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

from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QApplication,\
     QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsLineItem, QGraphicsPolygonItem
from PyQt5.QtGui import  QPen, QPainter, QColor, QBrush, QCursor, QTransform, QPolygonF
from PyQt5.QtCore import Qt, QPoint, QPointF, QRect, QRectF, QSize, QSizeF, QLineF # , pyqtSignal

QPEN_DEF   = QPen(Qt.yellow, 1, Qt.SolidLine)  # Qt.DashLine
QBRUSH_DEF = QBrush()  # Qt.red
QCOLOR_DEF = QColor(Qt.yellow)

PIXEL   = 1
LINE    = 2
RECT    = 4
POLYGON = 8
ELLIPSE = 16


class ROIBase():
    def __init__(self, **kwa):
        self.view       = kwa.get('view', None) # QGraphicsView()
        self.pos        = kwa.get('pos', QPoint(0,0))
        self.roitype    = kwa.get('roitype', None)
        self.mode       = kwa.get('mode', 0) # 0/1/2/4/8/16 - invisible/visible/add/move/rotate/edit
        self.moveable   = kwa.get('moveable', False)
        self.scaleable  = kwa.get('scaleable', False)
        self.ratateable = kwa.get('ratateable', False)
        self.scitem     = None
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

#    def add_to_scene(self, pos=QPoint(1,1), **kwa):
#        logger.warning('add_to_scene must be implemented in derived class')

    def add_to_scene(self, pen=None, brush=None):
        item = self.scitem
        item.setPen(self.pen if pen is None else pen)
        if self.roitype != LINE:
            item.setBrush(self.brush if brush is None else brush)
        self.scene().addItem(item)


def items_at_point(scene, point):
    items = scene.items(point)
    logging.debug('sc.itemsAt(%s): %s' % (str(point), str(items)))
    return items


#def qtransform_rotate_around_center(center=QPointF(0,0), angle_deg=20):
#    return QTransform().translate(center.x(), center.y()).\
#           rotate(angle_deg).\
#           translate(-center.x(), -center.y())


class ROIPixel(ROIBase):

    def __init__(self, **kwa):
        ROIBase.__init__(self, **kwa)
        self.roitype = PIXEL
        self.brush = kwa.get('brush', QBrush(QCOLOR_DEF))
        # self.pos = kwa.get('pos', QPoint(0,0))

    def add_to_scene(self, pos=QPoint(5,1)):
        """Adds pixel as a rect to scene, returns QGraphicsRectItem"""
        self.pos = pos
        self.scitem = QGraphicsRectItem(QRectF(QPointF(pos), QSizeF(1,1)))
        ROIBase.add_to_scene(self, pen=None, brush=None)
        #self.scitem = self.scene().addRect(QRectF(QPointF(pos), QSizeF(1,1)), self.pen, self.brush)


class ROILine(ROIBase):

    def __init__(self, **kwa):
        ROIBase.__init__(self, **kwa)
        self.roitype = LINE

    def add_to_scene(self, line, pen=None):
        """Adds/returns QGraphicsLineItem to scene"""
        self.scitem = QGraphicsLineItem(QLineF(line))
        ROIBase.add_to_scene(self, pen, brush=None)
        #self.scitem = self.scene().addLine(QLineF(line), pen, brush)


class ROIRect(ROIBase):

    def __init__(self, **kwa):
        ROIBase.__init__(self, **kwa)
        self.roitype = RECT

    def add_to_scene(self, rect, pen=None, brush=None, angle_deg=0):
        """Adds/returns QGraphicsRectItem to scene"""
        item = self.scitem = QGraphicsRectItem(QRectF(rect))
        item.setTransformOriginPoint(item.rect().center())
        if angle_deg != 0: item.setRotation(angle_deg)
        ROIBase.add_to_scene(self, pen, brush)

#        t = roiu.qtransform_rotate_around_center(center=r0.center(), angle_deg=20)
#        p1 = t.mapToPolygon(QRect(r0))
#        self.scitem = self.scene().addRect(QRectF(rect), pen, brush)


class ROIPolygon(ROIBase):

    def __init__(self, **kwa):
        ROIBase.__init__(self, **kwa)
        self.roitype = POLYGON

    def add_to_scene(self, poly, pen=QPEN_DEF, brush=QBRUSH_DEF):
        """Adds/returns QGraphicsPolygonItem to scene"""
        self.scitem = QGraphicsPolygonItem(QPolygonF(poly))
        ROIBase.add_to_scene(self, pen, brush)
        #self.scitem = self.scene().addPolygon(QPolygonF(poly), pen, brush)


class ROIEllipse(ROIBase):

    def __init__(self, **kwa):
        ROIBase.__init__(self, **kwa)
        self.roitype = ELLIPSE

    def add_to_scene(self, rect, pen=QPEN_DEF, brush=QBRUSH_DEF,\
                     angle_deg=0, start_angle=0, span_angle=360):
        """Adds/returns QGraphicsEllipseItem to scene."""
        item = self.scitem = QGraphicsEllipseItem(QRectF(rect))
        item.setStartAngle(start_angle*16)
        item.setSpanAngle(span_angle*16)
        item.setTransformOriginPoint(item.rect().center())
        item.setRotation(angle_deg)
        ROIBase.add_to_scene(self, pen, brush)
        #self.scitem = self.scene().addEllipse(QRectF(rect), pen, brush)


# EOF
