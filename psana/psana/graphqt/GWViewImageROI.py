
"""
Class :py:class:`GWViewImageROI` is a GWView for interactive image
===============================================================

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
from PyQt5.QtGui import  QPen, QPainter, QColor, QBrush
from PyQt5.QtCore import Qt, QPoint, QPointF, QRect, QRectF, QSize, QSizeF


class GWViewImageROI(GWViewImage):

    image_pixmap_changed = pyqtSignal()

    def __init__(self, parent=None, arr=None,\
                 coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20),\
                 origin='UL', scale_ctl='HV', show_mode=0, signal_fast=True):

        GWViewImage.__init__(self, parent, arr, coltab, origin, scale_ctl, show_mode, signal_fast)
        self._name = 'GWViewImageROI' # self.__class__.__name__


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



    def paintEvent(self, e):
        print('XXX in paintEvent') # , dir(e))
        #painter = QPainter(self)
        GWViewImage.paintEvent(self, e)
#        self.draw_something()



    def mousePressEvent(self, e):
        scpoint = self.mapToScene(e.pos())

        print('XXX GWViewImageROI.mousePressEvent but=%d %s scene x=%.1f y=%.1f'%\
                     (e.button(), str(e.pos()), scpoint.x(), scpoint.y())) # self.__class__.__name__

        #print('XXX DragPoly.mousePressEvent button L/R/M = 1/2/4: ', e.button())
        #print('XXX DragPoly.mousePressEvent Left: ', e.button()==Qt.LeftButton)

        GWViewImage.mousePressEvent(self, e)




    def add_pixel_to_scene(self, pos=QPoint(5,1),\
                                 brush=QBrush(Qt.red),\
                                 pen=QPen(Qt.red, 1, Qt.SolidLine),\
                                 color=QColor(Qt.yellow),\
                                 pen_is_cosmetic=True):
        """Adds pixel as a rect to scene, returns QGraphicsRectItem"""
        pen.setCosmetic(pen_is_cosmetic)
        if color is not None:
          pen.setColor(color)
          brush.setColor(color)
        rect = QRectF(QPointF(pos), QSizeF(1,1))
        return self.scene().addRect(rect, pen, brush)



    def draw_something(self):
        self.add_pixel_to_scene(QPoint(5, 1))
        for i in range(100):
            self.add_pixel_to_scene(QPoint(1+i, 2+i))


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
