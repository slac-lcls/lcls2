#!/usr/bin/env python

import sys  # used in subclasses

import logging
logger = logging.getLogger(__name__)

import os
import numpy as np

from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QApplication, QGraphicsPixmapItem
from PyQt5.QtGui import QBrush, QPen, QCursor, QColor, QImage, QPixmap, QPainter
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF, QPoint, QTimer, QSizeF, QRect, QEvent
import psana2.graphqt.QWUtils as qu
import psana2.graphqt.ColorTable as ct
from psana2.pyalgos.generic.NDArrUtils import info_ndarr


def draw_something_rects(painter):
    #from random import randint
    #painter = QPainter()
    pen = QPen(QColor("#376F9F"), 0, Qt.SolidLine) #Qt.black
    pen.setWidth(3)
    painter.setPen(pen)

    brush = QBrush(QColor("#FFD141"))
    brush.setStyle(Qt.Dense1Pattern)
    painter.setBrush(brush)

    painter.drawRects(
        QRect(50, 50, 100, 100),
        QRect(60, 60, 150, 100),
        QRect(70, 70, 100, 150),
        QRect(80, 80, 150, 100),
        QRect(90, 90, 100, 150),
    )
    painter.end()


class TestQGraphicsScene(QGraphicsScene):
    def __init__(self, **kwa):
        QGraphicsScene.__init__(self, **kwa)
        print('XXX TestQGraphicsScene')

    def drawForeground(self, painter, rect):
        QGraphicsScene.drawForeground(self, painter, rect)
        print('XXX TestQGraphicsScene.drawForeground')
        draw_something_rects(painter)

#    def paintEvent(self, e):
#        QGraphicsScene.paintEvent(self, e)
#        print('XXX in TestQGraphicsScene.paintEvent')

    def event(self, e,  *args, **kwa):
        res = QGraphicsScene.event(self, e, *args, **kwa)
        evtype = e.type()
        print('XXX in TestQGraphicsScene.event', evtype, end='')
        if evtype == QEvent.Paint:       print(' QEvent.Paint')
        elif evtype == QEvent.Move:      print(' QEvent.Move')
        elif evtype == QEvent.Resize:    print(' QEvent.Resize')
        elif evtype == QEvent.Leave:     print(' QEvent.Leave')
        elif evtype == QEvent.Enter:     print(' QEvent.Enter')
        elif evtype == QEvent.FocusOut:  print(' QEvent.FocusOut')
        else:                            print(' QEvent. other')

        #painter = QPainter()
        return res

    def items_of_type(self, qgsitem=QGraphicsPixmapItem):
        return [item for item in self.items() if isinstance(item, qgsitem)]

    def mousePressEvent(self, e):
        QGraphicsScene.mousePressEvent(self, e)

        if e.button() == Qt.LeftButton:  # and e.modifiers() & Qt.ControlModifier
            pos = e.scenePos()
            #rs_center = self.scene_rect().center()
            print('TestQGraphicsScene.mousePressEvent on LeftButton pos:', pos.x(), pos.y())
            #self.click_pos = self.mapToScene(e.pos())

            pixmap_items = self.items_of_type(qgsitem=QGraphicsPixmapItem)
            if pixmap_items is []:
                print('TestQGraphicsScene list of pixmap items is empty')
                return
            pixmap = pixmap_items[0].pixmap()
            print('dir(pixmap):', dir(pixmap))


class GWView(QGraphicsView):
    """Bare minimum to move and zoom viewport rect."""

    def __init__(self, parent=None, rscene=QRectF(0, 0, 100, 100),\
                 origin='UL', scale_ctl='HV',\
                 show_mode=0o0, **kwa):
        sc = TestQGraphicsScene()
        QGraphicsView.__init__(self, sc, parent)

        self.kwa = kwa
        self.set_scale_control(scale_ctl)
        self.set_style()
        self.fit_in_view(rscene, mode=Qt.KeepAspectRatio)
        self.click_pos = None
        self.ang_wheel_old = None
        if show_mode > 0: self.add_test_items_to_scene(show_mode)


    def scene_rect(self):
        return self.scene().sceneRect()


    def set_scene_rect(self, r):
        if r is not None:
            self.scene().setSceneRect(r)  # self.setSceneRect(r)  # WORKS DIFFERENTLY!
            if logging.root.level == logging.DEBUG:
                print('GWView.set_scene_rect:  %s' % qu.info_rect_xywh(r), end='\r')


    def fit_in_view(self, rs=None, mode=Qt.IgnoreAspectRatio):
        """Fits visible part of the scene in the rect rs (scene units).
           possible mode Qt.KeepAspectRatioByExpanding, Qt.KeepAspectRatio, Qt.IgnoreAspectRatio.
        """
        r = self.scene_rect() if rs is None else rs
        self.set_scene_rect(rs)
        self.fitInView(r, mode)


    def set_style(self):
        #logger.debug('GWView.set_style')
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
        #logging.debug('set_scale_control to %d' % self._scale_ctl)


    def update_my_scene(self):
        """should be re-implemented in derived classes, if needed..."""
        pass


    def resizeEvent(self, e):
        """important method to make zoom and pan working correctly..."""
        QGraphicsView.resizeEvent(self, e)
        #logger.debug(sys._getframe().f_code.co_name)
        self.fit_in_view()
        self.update_my_scene()


    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:  # and e.modifiers() & Qt.ControlModifier
            logger.debug('GWView.mousePressEvent on LeftButton')
            #self.click_pos = self.mapToScene(e.pos())
            self.click_pos = e.pos()
            self.rs_center = self.scene_rect().center()
        QGraphicsView.mousePressEvent(self, e)


    def _move_scene_rect_by_mouse(self, e):
        #logger.debug('_move_scene_rect_by_mouse')
        dp = e.pos() - self.click_pos
        dx = dp.x() / self.transform().m11() if self._scale_ctl & 1 else 0
        dy = dp.y() / self.transform().m22() if self._scale_ctl & 2 else 0
        rs = self.scene_rect()
        rs.moveCenter(self.rs_center - QPointF(dx, dy))
        self.set_scene_rect(rs)


    def mouseMoveEvent(self, e):
        """Move rect CENTER."""
        QGraphicsView.mouseMoveEvent(self, e)
        if self._scale_ctl == 0: return
        if self.click_pos is None: return
        #logger.debug('mouseMoveEvent at valid click_pos and _scale_ctl')
        self._move_scene_rect_by_mouse(e)


    def mouseReleaseEvent(self, e):
        QGraphicsView.mouseReleaseEvent(self, e)
        logger.debug('mouseReleaseEvent')
        self._move_scene_rect_by_mouse(e)
        self.click_pos = None


    def wheelEvent(self, e):
        ang = e.angleDelta().x()
        if ang != self.ang_wheel_old:
           logger.debug('wheelEvent new direction of e.angleDelta().x(): %.6f' % ang) #+/-120 on each step
           self.ang_wheel_old = ang
        QGraphicsView.wheelEvent(self, e)

        if self._scale_ctl == 0: return

        p = self.mapToScene(e.pos())
        px, py = p.x(), p.y()  # / self.transform().m22()

        rs = self.scene_rect()
        x,y,w,h = rs.x(), rs.y(), rs.width(), rs.height()

        # zoom scene rect relative to mouse position
        f = 1 + 0.3 * (1 if ang>0 else -1)
        dxc = (f-1)*(px-x)
        dyc = (f-1)*(py-y)
        dx, sx = (dxc, f*w) if self._scale_ctl & 1 else (0, w)
        dy, sy = (dyc, f*h) if self._scale_ctl & 2 else (0, h)

        rs.setRect(x-dx, y-dy, sx, sy)
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





class GWViewImage(GWView):

    image_pixmap_changed = pyqtSignal()

    def __init__(self, parent=None, arr=None,\
                 coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20),\
                 origin='UL', scale_ctl='HV', show_mode=0, signal_fast=True):

        h, w = arr.shape
        rscene = QRectF(0, 0, w, h)
        GWView.__init__(self, parent, rscene, origin, scale_ctl, show_mode, wheel_fast=signal_fast, move_fast=signal_fast)
        self._name = self.__class__.__name__
        self.set_coltab(coltab)
        self.pmi = None
        self.arr_limits_old = None
        self.arr_in_rect = None
        self.set_pixmap_from_arr(arr)
        #self.connect_mouse_move_event(self.on_mouse_move_event)


    def set_coltab(self, coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)):
        self.coltab = coltab


    def set_style(self):
        GWView.set_style(self)
        self.setWindowTitle('GWViewImage%s' %(30*' '))
        self.setAttribute(Qt.WA_TranslucentBackground)
        #self.layout().setContentsMargins(0,0,0,0)
        #self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)


    def add_pixmap_to_scene(self, pixmap):
        if self.pmi is None: self.pmi = self.scene().addPixmap(pixmap)
        else               : self.pmi.setPixmap(pixmap)


    def paintEvent(self, e):
        print('XXX in GWViewImage.paintEvent') # , dir(e))
        #painter = QPainter(self)
        GWView.paintEvent(self, e)
#        self.draw_something()



    def mousePressEvent(self, e):
        scpoint = self.mapToScene(e.pos())

        #print('XXX GWViewImage.mousePressEvent but=%d %s scene x=%.1f y=%.1f'%\
        #             (e.button(), str(e.pos()), scpoint.x(), scpoint.y())) # self.__class__.__name__

        #print('XXX DragPoly.mousePressEvent button L/R/M = 1/2/4: ', e.button())
        #print('XXX DragPoly.mousePressEvent Left: ', e.button()==Qt.LeftButton)

        GWView.mousePressEvent(self, e)




    def add_point_to_scene(self, point, brush=QBrush(Qt.red), pen=QPen(Qt.red, 1, Qt.SolidLine), color=QColor(Qt.yellow),  pen_is_cosmetic=True):
        """Adds rect to scene, returns QGraphicsRectItem"""
        pen.setCosmetic(pen_is_cosmetic)
        if color is not None:
          pen.setColor(color)
          brush.setColor(color)
        rect = QRectF(point, QSizeF(1,1))
        return self.scene().addRect(rect, pen, brush)



    def draw_something(self):
        for i in range(100):
            self.add_point_to_scene(QPoint(2+i, 2+i))






    def set_pixmap_from_arr(self, arr, set_def=True, amin=None, amax=None, frmin=0.00001, frmax=0.99999):
        """Input array is scailed by color table. If color table is None arr set as is."""
        self.arr = arr
        h, w = arr.shape

        mask = np.ones_like(arr, dtype=np.uint32)  * 0xffffffff
        mask[int(h/4):int(h/2), int(w/4):int(w/2)] = 0x00ffffff
        #print(info_ndarr(mask, 'XXXX mask:'))

        anorm = arr if self.coltab is None else\
                ct.apply_color_table(arr, ctable=self.coltab, amin=amin, amax=amax, frmin=frmin, frmax=frmax)

        anorm &= mask

        image = QImage(anorm, w, h, QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(image)
        self.add_pixmap_to_scene(pixmap)

        if set_def:
            rs = QRectF(0, 0, w, h)
            self.set_scene_rect(rs)
            self.rs_def = rs
            self.fit_in_view()

        self.draw_something()
        self.image_pixmap_changed.emit()



        arr = ct.image_to_arrcolors(image, channels=4)
        #print(info_ndarr(arr, 'XXXX image_to_arrcolors:'))

        arrB = ct.pixmap_channel(pixmap, channel=0)
        arrG = ct.pixmap_channel(pixmap, channel=1)
        arrR = ct.pixmap_channel(pixmap, channel=2)
        arrA = ct.pixmap_channel(pixmap, channel=3)

        #print(info_ndarr(arrA, 'XXXX alpha channel'))


        #print('dir(pixmap)', dir(pixmap))
        #print('dir(image)', dir(image))

        #mask = pixmap.createMaskFromColor()
        #arr = pixmap_to_arrcolors(pixmap, channels=4)
        #alphach = pixmap.alphaChannel()
        #arr = pixmap_to_arrcolors(alphach, channels=1)


    def array_in_rect(self, rect=None):
        if rect is None: rect=self.scene().sceneRect()
        x1,y1,x2,y2 = rect.getCoords()
        h,w = self.arr.shape
        arr_limits = int(max(0, floor(y1))), int(min(h-1, ceil(y2))),\
                     int(max(0, floor(x1))), int(min(w-1, ceil(x2)))

        if self.arr_limits_old is not None and arr_limits == self.arr_limits_old: return self.arr_in_rect
        r1,r2,c1,c2 = self.arr_limits_old = arr_limits

        # allow minimal shape of zoomed in array (2,2)
        if r1>r2-2:
           r2=r1+2
           if r2>=h: r1,r2 = h-3,h-1
        if c1>c2-2:
           c2=c1+2
           if c2>=w: c1,c2 = w-3,w-1

        self.arr_in_rect = self.arr[r1:r2,c1:c2]
        return self.arr_in_rect


    def connect_image_pixmap_changed(self, recip):
        self.image_pixmap_changed.connect(recip)


    def disconnect_image_pixmap_changed(self, recip):
        self.image_pixmap_changed.disconnect(recip)


    def cursor_on_image_pixcoords_and_value(self, p):
        """Returns cursor pointing pixel coordinates and value,
           - p (QPoint) - cursor on scene position
        """
        #p = self.mapToScene(e.pos())
        ix, iy = int(floor(p.x())), int(floor(p.y()))
        v = None
        arr = self.arr
        if ix<0\
        or iy<0\
        or iy>arr.shape[0]-1\
        or ix>arr.shape[1]-1: pass
        else: v = self.arr[iy,ix]
        return ix, iy, v



SCRNAME = sys.argv[0].split('/')[-1]
USAGE = '\nUsage: python %s <tname [0,3]>' %SCRNAME\
      + '\n   where tname=0/1/2/3 stands for scale_ctl "HV"/"H"/"V"/"", respectively'

def image_with_random_peaks(shape=(500, 500)):
import psana2.pyalgos.generic.NDArrGenerators as ag

    #logger.info('image_with_random_peaks shape: %s' % str(shape))
    img = ag.random_standard(shape, mu=0, sigma=10)
    #logger.info(info_ndarr(img, 'image_with_random_peaks  ag.random_standard'))

    peaks = ag.add_random_peaks(img, npeaks=50, amean=100, arms=50, wmean=1.5, wrms=0.3)
    ag.add_ring(img, amp=20, row=500, col=500, rad=300, sigma=50)
    return img


#def test_gfviewimage(tname):
#    logger.info(sys._getframe().f_code.co_name)
#    #arr = np.random.random((1000, 1000))
#    arr = image_with_random_peaks((1000, 1000))
#    #ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)
#    ctab = ct.color_table_monochr256()
#    #ctab = ct.color_table_interpolated()



logging.basicConfig(format='[%(levelname).1s] %(name)s L:%(lineno)03d %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.INFO)

#os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
app = QApplication(sys.argv)
w=GWViewImage(arr=image_with_random_peaks((1000, 1000)))
w.setGeometry(20, 20, 600, 600)
w.show()
app.exec_()
app.quit()
del w
del app

# EOF
