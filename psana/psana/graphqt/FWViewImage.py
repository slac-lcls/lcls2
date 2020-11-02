"""
Class :py:class:`FWViewImage` is a FWView for interactive image
===============================================================

FWView <- QGraphicsView <- ... <- QWidget

Usage ::

    # Test
    #-----
    import sys
    from psana.graphqt.FWViewImage import *
    import psana.graphqt.ColorTable as ct
    app = QApplication(sys.argv)
    ctab = ct.color_table_monochr256()
    w = FWViewImage(None, arr, origin='UL', scale_ctl='HV', coltab=ctab)
    w.show()
    app.exec_()

    # Main methods in addition to FWView
    #------------------------------------
    w.set_pixmap_from_arr(arr, set_def=True)
    w.set_coltab(self, coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20))

    w.connect_mouse_press_event_to(w.test_mouse_press_event_reception)
    w.connect_mouse_move_event_to(w.test_mouse_move_event_reception)
    w.connect_scene_rect_changed_to(w.test_scene_rect_changed_reception)

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

from math import floor
import ColorTable as ct
from FWView import *
from PyQt5.QtGui import QImage, QPixmap

#----

class FWViewImage(FWView):
    
    def __init__(self, parent=None, arr=None,\
                 coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20),\
                 origin='UL', scale_ctl='HV'):

        h, w = arr.shape
        rscene = QRectF(0, 0, w, h)
        FWView.__init__(self, parent, rscene, origin, scale_ctl)
        self._name = self.__class__.__name__
        self.set_coltab(coltab)
        self.pmi = None
        self.set_pixmap_from_arr(arr)


    def set_coltab(self, coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)):
        self.coltab = coltab


    def set_style(self):
        FWView.set_style(self)
        self.setWindowTitle('FWViewImage%s' %(30*' '))
        self.setAttribute(Qt.WA_TranslucentBackground)
        #self.setContentsMargins(0,0,0,0)
        #self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)


#    def mousePressEvent(self, e):
#        FWView.mousePressEvent(self, e)
#        #print('XXX FWViewImage.mousePressEvent')


#    def mouseMoveEvent(self, e):
#        FWView.mouseMoveEvent(self, e)

 
#    def closeEvent(self, e):
#        FWView.closeEvent(self, e)
#        print('%s.closeEvent' % self._name)


    def add_pixmap_to_scene(self, pixmap):
        if self.pmi is None: self.pmi = self.scene().addPixmap(pixmap)
        else               : self.pmi.setPixmap(pixmap)


    def set_pixmap_from_arr(self, arr, set_def=True):
        """Input array is scailed by color table. If color table is None arr set as is.
        """
        self.arr = arr
        anorm = arr if self.coltab is None else\
                ct.apply_color_table(arr, ctable=self.coltab) 
        h, w = arr.shape

        image = QImage(anorm, w, h, QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(image)
        self.add_pixmap_to_scene(pixmap)

        if set_def:
            rs = QRectF(0, 0, w, h)
            self.set_rect_scene(rs, set_def)


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

#----

    if __name__ == "__main__":

      def test_mouse_move_event_reception(self, e):
        """Overrides method from FWView"""
        p = self.mapToScene(e.pos())
        ix, iy, v = self.cursor_on_image_pixcoords_and_value(p)
        fv = 0 if v is None else v 
        self.setWindowTitle('FWViewImage x=%d y=%d v=%s%s' % (ix, iy, '%.1f'%fv, 25*' '))


      def key_usage(self):
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  R - reset original size'\
               '\n  N - set new pixmap'\
               '\n  W - set new pixmap of random shape, do not change default scene rect'\
               '\n  D - set new pixmap of random shape and change default scene rect'\
               '\n'


      def keyPressEvent(self, e):
        #print('keyPressEvent, key=', e.key())
        if   e.key() == Qt.Key_Escape:
            print('Close app')
            self.close()

        elif e.key() == Qt.Key_R:
            print('Reset original size')
            self.reset_original_size()

        elif e.key() == Qt.Key_N:
            print('Set new pixel map')
            s = self.pmi.pixmap().size()
            img = image_with_random_peaks((s.height(), s.width()))
            self.set_pixmap_from_arr(img, set_def=False)

        elif e.key() in (Qt.Key_W, Qt.Key_D):
            change_def = e.key()==Qt.Key_D
            print('%s: change scene rect %s' % (self._name, 'set new default' if change_def else ''))
            v = ag.random_standard((4,), mu=0, sigma=200, dtype=np.int)
            rs = QRectF(v[0], v[1], v[2]+1000, v[3]+1000)
            print('Set scene rect: %s' % str(rs))
            #self.set_rect_scene(rs, set_def=change_def)
            img = image_with_random_peaks((int(rs.height()), int(rs.width())))
            self.set_pixmap_from_arr(img, set_def=change_def)

        else:
            print(self.key_usage())

#----

if __name__ == "__main__":

  import sys
  sys.path.append('..') # use relative path from parent dir
  import pyalgos.generic.NDArrGenerators as ag
  import numpy as np


  def image_with_random_peaks(shape=(500, 500)):
    from pyalgos.generic.NDArrUtils import print_ndarr

    print('XXX1 shape:', shape)
    img = ag.random_standard(shape, mu=0, sigma=10)
    print_ndarr(img, 'XXX ag.random_standard')

    peaks = ag.add_random_peaks(img, npeaks=50, amean=100, arms=50, wmean=1.5, wrms=0.3)
    ag.add_ring(img, amp=20, row=500, col=500, rad=300, sigma=50)
    return img


  def test_wfviewimage(tname):
    print('%s:' % sys._getframe().f_code.co_name)
    #arr = np.random.random((1000, 1000))
    arr = image_with_random_peaks((1000, 1000))
    #ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)
    ctab = ct.color_table_monochr256()
    #ctab = ct.color_table_interpolated()

    app = QApplication(sys.argv)
    w = None
    if   tname == '0': w = FWViewImage(None, arr, coltab=ctab, origin='UL', scale_ctl='HV')
    elif tname == '1': w = FWViewImage(None, arr, coltab=ctab, origin='UL', scale_ctl='H')
    elif tname == '2': w = FWViewImage(None, arr, coltab=ctab, origin='UL', scale_ctl='V')
    elif tname == '3': w = FWViewImage(None, arr, coltab=ctab, origin='UL', scale_ctl='')
    elif tname == '4':
        arrct = ct.array_for_color_bar(orient='H')
        w = FWViewImage(None, arrct, coltab=None, origin='UL', scale_ctl='H')
        w.setGeometry(50, 50, 500, 40)
    elif tname == '5':
        arrct = ct.array_for_color_bar(orient='V')
        w = FWViewImage(None, arrct, coltab=None, origin='UL', scale_ctl='V')
        w.setGeometry(50, 50, 40, 500)
    elif tname == '6':
        #ctab= ct.color_table_rainbow(ncolors=1000, hang1=0, hang2=360)
        #ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)
        #ctab = ct.color_table_monochr256()
        ctab = ct.color_table_interpolated()
        arrct = ct.array_for_color_bar(ctab, orient='H')
        w = FWViewImage(None, arrct, coltab=None, origin='UL', scale_ctl='H')
        w.setGeometry(50, 50, 500, 40)
    elif tname == '7':
        a = np.arange(15).reshape((5, 3))
        w = FWViewImage(None, a, coltab=ctab, origin='UL', scale_ctl='HV')
    else:
        print('test %s is not implemented' % tname)
        return

    w.connect_mouse_press_event_to(w.test_mouse_press_event_reception)
    w.connect_mouse_move_event_to(w.test_mouse_move_event_reception)
    w.connect_scene_rect_changed_to(w.test_scene_rect_changed_reception)

    w.show()
    app.exec_()

    del w
    del app

#----

if __name__ == "__main__":
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_wfviewimage(tname)
    sys.exit('End of Test %s' % tname)

#----
