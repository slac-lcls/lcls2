
import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QApplication
from PyQt5.QtGui import QBrush, QPen, QCursor, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF, QTimer


class GWView(QGraphicsView):
    """Bare minimum to move and zoom viewport rect."""

    def __init__(self, parent=None, rscene=QRectF(0, 0, 100, 100), scale_ctl='HV', show_mode=0o0, **kwa):
        QGraphicsView.__init__(self, QGraphicsScene(rscene), parent)
        self.fit_in_view(rscene, Qt.KeepAspectRatio)
        self.set_scale_control(scale_ctl)
        self.set_style(**kwa)
        self.add_test_items_to_scene(show_mode, **kwa)
        self.click_pos = None


    def fit_in_view(self, rs=None, mode=Qt.KeepAspectRatio):
        """ Fits visible part of the scene in the rect rs (scene units).
            possible mode KeepAspectRatioByExpanding, KeepAspectRatio, IgnoreAspectRatio.
        """
        r = self.scene().sceneRect() if rs is None else rs
        #self.scene().setSceneRect(r)
        self.fitInView(r, mode)


#    def set_scene_rect(self, r):
#        """ Set scene bounding box rect."""
#        self.scene().setSceneRect(r)


    def set_style(self, **kwa):
        self.brudf = kwa.get('brudf', QBrush())
        self.brubx = kwa.get('brubx', QBrush(Qt.black, Qt.SolidPattern))
        self.pendf = kwa.get('pendf', QPen())
        self.pendf.setStyle(kwa.get('pendf_style', Qt.NoPen))
        self.penbx = kwa.get('penbx', QPen(Qt.black, 6, Qt.SolidLine))
        self.set_background_brush(**kwa)


    def set_background_brush(self, color=QColor(50,5,50), pattern=Qt.SolidPattern, **kwa):
        self.setBackgroundBrush(QBrush(color, pattern))


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


    def mousePressEvent(self, e):
        logger.debug('mousePressEvent')
        if e.button() == Qt.LeftButton:  # and e.modifiers() & Qt.ControlModifier
            self.click_pos = e.pos()
            self.rs_center = self.sceneRect().center()
        QGraphicsView.mousePressEvent(self, e)
        #self.setDragMode(self.ScrollHandDrag) # ScrollHandDrag, RubberBandDrag, NoDrag DOES NOT WORK ???


    def mouseMoveEvent(self, e):
        """ Move CENTER othervise zoom at scroll works incorrectly.
        """
        QGraphicsView.mouseMoveEvent(self, e)
        if self._scale_ctl == 0: return
        if self.click_pos is None: return

        dp = e.pos() - self.click_pos
        dx = dp.x() / self.transform().m11() if self._scale_ctl & 1 else 0
        dy = dp.y() / self.transform().m22() if self._scale_ctl & 2 else 0

        sc = self.scene()
        rs = sc.sceneRect()
        rs.moveCenter(self.rs_center - QPointF(dx, dy))
        sc.setSceneRect(rs)
        #self.centerOn(self.rs_center - QPointF(dx, dy)) # DOES NOT WORK ???


    def mouseReleaseEvent(self, e):
        logger.debug('mouseReleaseEvent')
        QGraphicsView.mouseReleaseEvent(self, e)
        if self.click_pos is not None:
           self.click_pos = None


    def wheelEvent(self, e):
        logger.debug('wheelEvent e.angleDelta: %.3f' % e.angleDelta().y()) #+/-120 on each step
        QGraphicsView.wheelEvent(self, e)

        if self._scale_ctl == 0: return

        p = self.mapToScene(e.pos())
        px, py = p.x(), p.y()

        rs = self.scene().sceneRect()
        x,y,w,h = rs.x(), rs.y(), rs.width(), rs.height()

        # zoom scene rect relative to mouse position
        f = 1 + 0.3 * (1 if e.angleDelta().y()>0 else -1)
        dxc = (f-1)*(px-x)
        dyc = (f-1)*(py-y)
        dx, sx = (dxc, f*w) if self._scale_ctl & 1 else (0, w)
        dy, sy = (dyc, f*h) if self._scale_ctl & 2 else (0, h)

        rs.setRect(x-dx, y-dy, sx, sy)
        self.scene().setSceneRect(rs)
        self.fit_in_view(rs)


    def add_rect_to_scene_v1(self, rect, brush=QBrush(), pen=QPen(Qt.yellow, 4, Qt.DashLine)):
        """Adds rect to scene, returns QGraphicsRectItem"""
        pen.setCosmetic(True)
        return self.scene().addRect(rect, pen, brush)


    def add_test_items_to_scene(self, show_mode=3, colori=Qt.red, colfld=Qt.magenta, **kwa):
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
