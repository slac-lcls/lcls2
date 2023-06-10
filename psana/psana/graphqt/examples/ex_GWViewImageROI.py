#!/usr/bin/env python

from psana.graphqt.GWViewImageROI import *

logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] %(filename)s L:%(lineno)03d %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.INFO)

import inspect
import sys
sys.path.append('..') # use relative path from parent dir
import psana.pyalgos.generic.NDArrGenerators as ag
import numpy as np


class TestGWViewImageROI(GWViewImageROI):

    def KEY_USAGE(self): return 'Keys:'\
               '\n  ESC - exit'\
               '\n  R - reset original size'\
               '\n  N - set new pixmap'\
               '\n  W - set new pixmap of random shape, do not change default scene rect'\
               '\n  H - set new pixmap of random shape and change default scene rect'\
               '\n  ROI = %s select from %s' % (str(self.roi_name), ', '.join(['%s-%s'%(k,n) for t,n,k in roiu.roi_tuple])) + \
               '\n  MODE = %s select from %s' % (str(self.mode_name), ', '.join(['%s-%s'%(k,n) for t,n,k in roiu.mode_tuple])) + \
               '\n'

    def __init__(self, parent=None, arr=None,\
                 coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20),\
                 origin='UL', scale_ctl='HV', show_mode=0, signal_fast=True):
        GWViewImageROI.__init__(self, parent, arr, coltab, origin, scale_ctl, show_mode, signal_fast)
        print(self.KEY_USAGE())


    def test_mouse_move_event_reception(self, e):
        """Overrides method from GWView"""
        p = self.mapToScene(e.pos())
        ix, iy, v = self.cursor_on_image_pixcoords_and_value(p)
        fv = 0 if v is None else v
        self.setWindowTitle('TestGWViewImageROI x=%d y=%d v=%s%s' % (ix, iy, '%.1f'%fv, 25*' '))


    def keyPressEvent(self, e):
        GWViewImageROI.keyPressEvent(self, e)

        key = e.key()
        ckey = chr(key)

        logger.info('keyPressEvent, key = %s' % ckey)
        if key == Qt.Key_Escape:
            logger.info('Close app')
            self.close()

        elif key == Qt.Key_R:
            logger.info('Reset original size')
            self.reset_scene_rect()

        elif key == Qt.Key_N:
            logger.info('Set new pixel map')
            s = self.pmi.pixmap().size()
            img = image_with_random_peaks((s.height(), s.width()))
            self.set_pixmap_from_arr(img, set_def=False)

        elif key in (Qt.Key_W, Qt.Key_H):
            change_def = key==Qt.Key_H
            logger.info('%s: change scene rect %s' % (self._name, 'set new default' if change_def else ''))
            v = ag.random_standard((4,), mu=0, sigma=200, dtype=np.int)
            rs = QRectF(v[0], v[1], v[2]+1000, v[3]+1000)
            logger.info('Set scene rect: %s' % str(rs))
            img = image_with_random_peaks((int(rs.height()), int(rs.width())))
            self.mask = ct.test_mask(img)
            self.set_pixmap_from_arr(img, set_def=change_def)

        if ckey in roiu.roi_keys:
            i = roiu.roi_keys.index(ckey)
            self.roi_type = roiu.roi_types[i]
            self.roi_name = roiu.roi_names[i]
            logger.info('set roi_type: %d roi_name: %s' % (self.roi_type, self.roi_name))

        if ckey in roiu.mode_keys:
            i = roiu.mode_keys.index(ckey)
            self.mode_type = roiu.mode_types[i]
            self.mode_name = roiu.mode_names[i]
            logger.info('set mode_type: %d roi_name: %s' % (self.mode_type, self.mode_name))

            sc = '' if self.mode_type > roiu.VISIBLE else 'HV'
            self.set_scale_control(scale_ctl=sc)

        logger.info(self.KEY_USAGE())


def image_with_random_peaks(shape=(500, 500)):
    from psana.pyalgos.generic.NDArrUtils import info_ndarr

    logger.info('image_with_random_peaks shape: %s' % str(shape))
    img = ag.random_standard(shape, mu=0, sigma=10)
    logger.info(info_ndarr(img, 'image_with_random_peaks  ag.random_standard'))

    peaks = ag.add_random_peaks(img, npeaks=50, amean=100, arms=50, wmean=1.5, wrms=0.3)
    ag.add_ring(img, amp=20, row=500, col=500, rad=300, sigma=50)
    return img


def test_gfviewimageroi(tname):
    logger.info(sys._getframe().f_code.co_name)
    #arr = np.random.random((1000, 1000))
    arr = image_with_random_peaks((1000, 1000))
    #ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)
    ctab = ct.color_table_monochr256()
    #ctab = ct.color_table_interpolated()

    app = QApplication(sys.argv)
    w = None
    if   tname == '0': w = TestGWViewImageROI(None, arr, coltab=ctab, origin='UL', scale_ctl='HV')
    elif tname == '1': w = TestGWViewImageROI(None, arr, coltab=ctab, origin='UL', scale_ctl='H')
    elif tname == '2': w = TestGWViewImageROI(None, arr, coltab=ctab, origin='UL', scale_ctl='V')
    elif tname == '3': w = TestGWViewImageROI(None, arr, coltab=ctab, origin='UL', scale_ctl='')
    elif tname == '4':
        arrct = ct.array_for_color_bar(orient='H')
        w = TestGWViewImageROI(None, arrct, coltab=None, origin='UL', scale_ctl='H')
        w.setGeometry(50, 50, 500, 40)
    elif tname == '5':
        arrct = ct.array_for_color_bar(orient='V')
        w = TestGWViewImageROI(None, arrct, coltab=None, origin='UL', scale_ctl='V')
        w.setGeometry(50, 50, 40, 500)
    elif tname == '6':
        #ctab= ct.color_table_rainbow(ncolors=1000, hang1=0, hang2=360)
        #ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)
        #ctab = ct.color_table_monochr256()
        #ctab = ct.color_table_interpolated()
        ctab = next_color_table(ict=7)
        arrct = ct.array_for_color_bar(ctab, orient='H')
        w = TestGWViewImageROI(None, arrct, coltab=None, origin='UL', scale_ctl='H')
        w.setGeometry(50, 50, 500, 40)
    elif tname == '7':
        a = np.arange(15).reshape((5, 3))
        w = TestGWViewImageROI(None, a, coltab=ctab, origin='UL', scale_ctl='HV')
    else:
        logger.info('test %s IS NOT IMPLEMENTED' % tname)
        return

    w.connect_mouse_press_event(w.test_mouse_press_event_reception)
    w.connect_mouse_move_event(w.test_mouse_move_event_reception)
    w.connect_scene_rect_changed(w.test_scene_rect_changed_reception)

    w.setWindowTitle('ex_GWViewImageROI')
    w.setGeometry(20, 20, 600, 600)
    w.show()
    app.exec_()

    del w
    del app


USAGE = '\nUsage: %s <tname>\n' % sys.argv[0].split('/')[-1]\
      + '\n'.join([s for s in inspect.getsource(test_gfviewimageroi).split('\n') \
                   if "tname ==" in s or 'GWViewImageROI' in s])  # s[9:]


if __name__ == "__main__":
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.info(50*'_' + '\nTest %s' % tname)
    test_gfviewimageroi(tname)
    print(USAGE)
    sys.exit('End of Test %s' % tname)

# EOF
