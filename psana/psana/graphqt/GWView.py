"""
Class :py:class:`GWView` for interactively scaleable and moveble view
=====================================================================
GW - Graphical Widget
GWView - is a QGraphicsView - minimal class for scaleable and moveble view

Usage ::
    from psana.graphqt.GWView import *

See:
    - graphqt/examples/ex_FWView.py
    - :class:`GWView`
    - :class:`FWViewImage`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.


Created as FWView on 2017-01-03 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-16
Refactored/split FWView to GWView and GWViewExt on 2022-07-12
"""

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QApplication
from PyQt5.QtGui import QBrush, QPen, QCursor, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF, QTimer
import psana.graphqt.QWUtils as qu

class GWView(QGraphicsView):
    """Bare minimum to move and zoom viewport rect."""

    def __init__(self, parent=None, rscene=QRectF(0, 0, 100, 100),\
                 origin='UL', scale_ctl='HV',\
                 show_mode=0o0, **kwa):
        sc = QGraphicsScene()
        QGraphicsView.__init__(self, sc, parent)

        self.kwa = kwa
        self.set_scale_control(scale_ctl)
        self.set_style()
        self.fit_in_view(rscene, mode=Qt.KeepAspectRatio)
        self.add_test_items_to_scene(show_mode)
        self.click_pos = None


    def scene_rect(self):
        return self.scene().sceneRect()
        #return self.sceneRect()  # SAME AS ABOVE?


    def set_scene_rect(self, r):
        if r is None: return
        self.scene().setSceneRect(r)
        #self.setSceneRect(r)  # WORKS DIFFERENTLY
        #logger.debug('GWView.set_scene_rect rect: %s transform m11: %.2f m22: %.2f'%\
        #             (qu.info_rect_xywh(r), self.transform().m11(), self.transform().m22()))


    def fit_in_view(self, rs=None, mode=Qt.IgnoreAspectRatio):
        """Fits visible part of the scene in the rect rs (scene units).
           possible mode Qt.KeepAspectRatioByExpanding, Qt.KeepAspectRatio, Qt.IgnoreAspectRatio.
        """
        r = self.scene_rect() if rs is None else rs
        self.set_scene_rect(rs)
        self.fitInView(r, mode)


    def set_style(self):
        logger.debug('GWView.set_style')
        self.set_background_brush()


    def set_background_brush(self):
        self.setBackgroundBrush(QBrush(\
            self.kwa.get('bkg_color', QColor(50,5,50)),\
            self.kwa.get('bkg_pattern', Qt.SolidPattern)))


    def set_scale_control(self, scale_ctl='HV'):
        """Sets scale control bit-word
           = 0 - x, y frozen scales
           + 1 - x is interactive
           + 2 - y is interactive
           bit value 0/1 frozen/interactive
        """
        self.str_scale_ctl = scale_ctl
        self._scale_ctl = 0
        if 'H' in scale_ctl: self._scale_ctl += 1
        if 'V' in scale_ctl: self._scale_ctl += 2
        logging.debug('set_scale_control to %d' % self._scale_ctl)


    def update_my_scene(self):
        """should be re-implemented in derived classes, if needed..."""
        pass


    def resizeEvent(self, e):
        """important method to make zoom and pan working correctly..."""
        logger.debug('FWView.resizeEvent')
        QGraphicsView.resizeEvent(self, e)
        self.fit_in_view()
        self.update_my_scene()


    def mousePressEvent(self, e):
        logger.debug('FWView.mousePressEvent')
        if e.button() == Qt.LeftButton:  # and e.modifiers() & Qt.ControlModifier
            #self.click_pos = self.mapToScene(e.pos())
            self.click_pos = e.pos()
            self.rs_center = self.scene_rect().center()
        QGraphicsView.mousePressEvent(self, e)
        #self.setDragMode(self.ScrollHandDrag) # ScrollHandDrag, RubberBandDrag, NoDrag DOES NOT WORK ???


    def mouseMoveEvent(self, e):
        """Move rect CENTER.
        """
        QGraphicsView.mouseMoveEvent(self, e)
        if self._scale_ctl == 0: return
        if self.click_pos is None: return

        dp = e.pos() - self.click_pos
        dx = dp.x() / self.transform().m11() if self._scale_ctl & 1 else 0
        dy = dp.y() / self.transform().m22() if self._scale_ctl & 2 else 0
        rs = self.scene_rect()
        rs.moveCenter(self.rs_center - QPointF(dx, dy))
        self.set_scene_rect(rs)


    def mouseReleaseEvent(self, e):
        logger.debug('mouseReleaseEvent')
        QGraphicsView.mouseReleaseEvent(self, e)
        if self.click_pos is not None:
           self.click_pos = None


    def wheelEvent(self, e):
        #logger.debug('wheelEvent e.angleDelta: %.3f' % e.angleDelta().y()) #+/-120 on each step
        QGraphicsView.wheelEvent(self, e)

        if self._scale_ctl == 0: return

        p = self.mapToScene(e.pos())
        px, py = p.x(), p.y()  # / self.transform().m22()

        rs = self.scene_rect()
        x,y,w,h = rs.x(), rs.y(), rs.width(), rs.height()

        # zoom scene rect relative to mouse position
        f = 1 + 0.3 * (1 if e.angleDelta().y()>0 else -1)
        dxc = (f-1)*(px-x)
        dyc = (f-1)*(py-y)
        dx, sx = (dxc, f*w) if self._scale_ctl & 1 else (0, w)
        dy, sy = (dyc, f*h) if self._scale_ctl & 2 else (0, h)

        rs.setRect(x-dx, y-dy, sx, sy)
        #self.set_scene_rect(rs)
        self.fit_in_view(rs, Qt.IgnoreAspectRatio)


    def add_rect_to_scene_v1(self, rect, brush=QBrush(), pen=QPen(Qt.yellow, 4, Qt.DashLine), pen_is_cosmetic=True):
        """Adds rect to scene, returns QGraphicsRectItem"""
        pen.setCosmetic(pen_is_cosmetic)
        return self.scene().addRect(rect, pen, brush)


    def add_test_items_to_scene(self, show_mode=3, colori=Qt.red, colfld=Qt.magenta):
        if show_mode & 1:
            rs=QRectF(0, 0, 10, 10)
            self.rsi = self.add_rect_to_scene_v1(rs, pen=QPen(Qt.NoPen), brush=QBrush(colfld))
        if show_mode & 2:
            ror=QRectF(-1, -1, 2, 2)
            self.rori = self.add_rect_to_scene_v1(ror,\
                 pen=QPen(Qt.red, 0, Qt.SolidLine),\
                 brush=QBrush(colori))


if __name__ == "__main__":
    import sys
    import psana.graphqt.QWUtils as qu # print_rect
    sys.exit(qu.msg_on_exit())

# EOF
