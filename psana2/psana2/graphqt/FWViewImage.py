
"""
Class :py:class:`FWViewImage` is a FWView for interactive image
===============================================================

FWView <- QGraphicsView <- ... <- QWidget

Usage ::

    # Test
    #-----
    import sys
    from psana2.graphqt.FWViewImage import *
import psana2.graphqt.ColorTable as ct
    app = QApplication(sys.argv)
    ctab = ct.color_table_monochr256()
    w = FWViewImage(None, arr, origin='UL', scale_ctl='HV', coltab=ctab)
    w.show()
    app.exec_()

    # Main methods in addition to FWView
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

    # Overrides method from FWView
    #-----------------------------
    w.test_mouse_move_event_reception(e) # signature differs from FWView

    # Global methods for test
    #------------------------
    img = image_with_random_peaks(shape=(500, 500))

See:
    - :class:`FWView`
    - :class:`FWViewImage`
    - :class:`QWSpectrum`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2016-09-09 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-16
"""
from math import floor, ceil
from psana2.graphqt import ColorTable as ct
from psana2.graphqt.FWView import *
from PyQt5.QtGui import QImage, QPixmap


class FWViewImage(FWView):

    image_pixmap_changed = pyqtSignal()

    def __init__(self, parent=None, arr=None,\
                 coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20),\
                 origin='UL', scale_ctl='HV', show_mode=0, signal_fast=True):

        h, w = arr.shape
        rscene = QRectF(0, 0, w, h)
        FWView.__init__(self, parent, rscene, origin, scale_ctl, show_mode, signal_fast)
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
        FWView.set_style(self)
        self.setWindowTitle('FWViewImage%s' %(30*' '))
        self.setAttribute(Qt.WA_TranslucentBackground)
        #self.layout().setContentsMargins(0,0,0,0)
        #self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)


    def add_pixmap_to_scene(self, pixmap):
        if self.pmi is None: self.pmi = self.scene().addPixmap(pixmap)
        else               : self.pmi.setPixmap(pixmap)


    def set_pixmap_from_arr(self, arr, set_def=True, amin=None, amax=None, frmin=0.01, frmax=0.99):
        """Input array is scailed by color table. If color table is None arr set as is.
        """
        self.arr = arr
        anorm = arr if self.coltab is None else\
                ct.apply_color_table(arr, ctable=self.coltab, amin=amin, amax=amax, frmin=frmin, frmax=frmax)
        h, w = arr.shape

        image = QImage(anorm, w, h, QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(image)
        self.add_pixmap_to_scene(pixmap)

        if set_def:
            rs = QRectF(0, 0, w, h)
            self.set_rect_scene(rs, set_def)

        self.image_pixmap_changed.emit()


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


if __name__ == "__main__":
    sys.exit(qu.msg_on_exit())

# EOF
