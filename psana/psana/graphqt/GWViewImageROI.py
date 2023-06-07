
"""
Class :py:class:`GWViewImageROI` is a GWView for interactive image
====================================================================

GWView <- QGraphicsView <- ... <- QWidget

Usage ::

    # Test
    #-----
    import sys
    from psana.graphqt.GWViewImageROI import *
    import psana.graphqt.ColorTable as ct
    app = QApplication(sys.argv)
    ctab = ct.color_table_monochr256()
    w = GWViewImageROI(None, arr, origin='UL', scale_ctl='HV', coltab=ctab)
    w.show()
    app.exec_()

    # Main methods in addition to GWView
    #------------------------------------
    w.set_pixmap_from_arr(arr, set_def=True)
    w.set_coltab(self, coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20))

    w.connect_mouse_press_event(w.test_mouse_press_event_reception)
    w.connect_mouse_move_event(w.test_mouse_move_event_reception)
    w.connect_scene_rect_changed(w.test_scene_rect_changed_reception)

    # Methods
    #--------
    w.set_style()
    ix, iy, v = w.cursor_on_image_pixcoords_and_value(p)

    # Call-back slots
    #----------------
    w.mousePressEvent(e)
    # w.mouseMoveEvent(e)
    # w.closeEvent(e)
    w.key_usage()
    w.keyPressEvent(e)

    # Overrides method from GWView
    #-----------------------------
    w.test_mouse_move_event_reception(e) # signature differs from GWView

    # Global methods for test
    #------------------------
    img = image_with_random_peaks(shape=(500, 500))

See:
    - :class:`GWView`
    - :class:`GWViewImage`
    - :class:`QWSpectrum`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2016-09-09 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-16
"""

from psana.graphqt.GWViewImage import *
from psana.pyalgos.generic.NDArrUtils import np, info_ndarr
from PyQt5.QtGui import  QPen, QPainter, QColor, QBrush, QTransform, QPolygonF
from PyQt5.QtCore import Qt, QPoint, QPointF, QRect, QRectF, QSize, QSizeF, QLineF

import psana.graphqt.GWROIUtils as roiu
QPEN_DEF, QBRUSH_DEF, QCOLOR_DEF  = roiu.QPEN_DEF, roiu.QBRUSH_DEF, roiu.QCOLOR_DEF

class GWViewImageROI(GWViewImage):

    image_pixmap_changed = pyqtSignal()

    def __init__(self, parent=None, arr=None,\
                 coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20),\
                 origin='UL', scale_ctl='HV', show_mode=0, signal_fast=True):

        GWViewImage.__init__(self, parent, arr, coltab, origin, scale_ctl, show_mode, signal_fast)
        self._name = 'GWViewImageROI' # self.__class__.__name__
        self.add_ROI = False
        self.list_of_rois = []
        self.roitype = None
        self.mouse_left_pressed = False
        self.pos_old = None

#    def draw_something(self):
#        #painter = QPainter(self.label.pixmap())
#        pixmap = self.pmi.pixmap()
#        painter = QPainter(pixmap)
#        #painter = QPainter(self)
#        pen = QPen()
#        #pen.setWidth(40)
#        pen.setColor(QColor('white'))  # 'red'))
#        painter.setPen(pen)
#        #painter.drawLine(10, 10, 250, 200)
#        for i in range(100):
#            painter.drawPoint(i+10, 10+i)
#        painter.end()
#        #self.update()



#    def paintEvent(self, e):
#        print('XXX in paintEvent') # , dir(e))
#        GWViewImage.paintEvent(self, e)


    def mouseMoveEvent(self, e):
        GWViewImage.mouseMoveEvent(self, e)

        scpoint = self.mapToScene(e.pos())
        xsc, ysc = int(scpoint.x()), int(scpoint.y())
        pos=QPoint(xsc, ysc)

        if pos == self.pos_old: return

        self.pos_old = pos

        if self.mouse_left_pressed and self.add_ROI and self.roitype == 'pixel':

            #print('XXX GWViewImageROI.mouseMoveEvent but=%d %s scene x=%.1f y=%.1f'%\
            #         (e.button(), str(e.pos()), xsc, ysc)) # self.__class__.__name__

            self.add_roi_pixel(pos)



    def mousePressEvent(self, e):
        GWViewImage.mousePressEvent(self, e)

        scpoint = self.mapToScene(e.pos())
        xsc, ysc = int(scpoint.x()), int(scpoint.y())
        pos=QPoint(xsc, ysc)
        self.pos_old = pos

        print('XXX GWViewImageROI.mousePressEvent but=%d %s scene x=%.1f y=%.1f'%\
                     (e.button(), str(e.pos()), xsc, ysc)) # self.__class__.__name__

        #print('XXX DragPoly.mousePressEvent button L/R/M = 1/2/4: ', e.button())
        #print('XXX DragPoly.mousePressEvent Left: ', e.button()==Qt.LeftButton)

        if e.button() != Qt.LeftButton: return

        self.mouse_left_pressed = True
        self.roitype = 'pixel'

        if self.add_ROI and self.roitype == 'pixel':
            #print('XXX GWViewImageROI > add_pixel_to_scene pos:', pos)
            #self.add_pto_scene(pos=pos)

            self.add_roi_pixel(pos)


    def mouseReleaseEvent(self, e):
        GWViewImage.mouseReleaseEvent(self, e)
        print('XXX GWViewImageROI.mouseReleaseEvent but=%d %s'%\
                     (e.button(), str(e.pos()))) # self.__class__.__name__
        self.roitype = None
        self.mouse_left_pressed = False



    def add_roi_pixel(self, pos):

        for o in self.list_of_rois:
            if pos == o.pos:
                self.scene().removeItem(o.scitem)
                self.list_of_rois.remove(o)
                is_removed = True
                return

        o = roiu.ROIPixel(view=self)
        o.add_to_scene(pos)
        self.list_of_rois.append(o)


    def draw_something(self):
        for i in range(100):
            roiu.ROIPixel(view=self).add_to_scene(QPoint(1+i, 2+i))

        # Pixel
        pi = QPointF(300, 100)
        itpi = roiu.ROIPixel(view=self).add_to_scene(pi)

        # Line
        l0 = QLineF(QPointF(300, 600), QPointF(600, 300))
        itl0 = roiu.ROILine(view=self).add_to_scene(l0)

        # Rect
        r0 = QRectF(100, 200, 200, 100)
        itr0 = roiu.ROIRect(view=self).add_to_scene(r0)
        itr1 = roiu.ROIRect(view=self).add_to_scene(r0, angle_deg=30)

        # Polygone
        p0 = QPolygonF([QPointF(500, 600), QPointF(700, 600), QPointF(700, 500), QPointF(650, 650)])
        itp0 = roiu.ROIPolygon(view=self).add_to_scene(p0)

        # Ellipse
        r0 = QRectF(300, 400, 200, 100)
        itp0 = roiu.ROIEllipse(view=self).add_to_scene(r0)
        itp1 = roiu.ROIEllipse(view=self).add_to_scene(r0, angle_deg=-30, start_angle=-20, span_angle=300)


    def set_pixmap_from_arr(self, arr, set_def=True, amin=None, amax=None, frmin=0.00001, frmax=0.99999):
        """Input array is scailed by color table. If color table is None arr set as is."""

        GWViewImage.set_pixmap_from_arr(self, arr, set_def, amin, amax, frmin, frmax)

        self.draw_something()
        image = self.qimage # QImage
        pixmap = self.qpixmap # QPixmap

        arr = ct.image_to_arrcolors(self.qimage, channels=4)
        print(info_ndarr(arr, 'XXXX image_to_arrcolors:'))

#        arrB = ct.pixmap_channel(self.qpixmap, channel=0)
#        arrG = ct.pixmap_channel(self.qpixmap, channel=1)
#        arrR = ct.pixmap_channel(self.qpixmap, channel=2)
        arrA = ct.pixmap_channel(self.qpixmap, channel=3)

        print(info_ndarr(arrA, 'XXXX alpha channel'))

        #mask = pixmap.createMaskFromColor()
        #arr = pixmap_to_arrcolors(pixmap, channels=4)

if __name__ == "__main__":
    import sys
    sys.exit(qu.msg_on_exit())

# EOF
