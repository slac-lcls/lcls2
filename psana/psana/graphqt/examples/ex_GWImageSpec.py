#!/usr/bin/env python

from psana.graphqt.GWImageSpec import *

logger = logging.getLogger(__name__)
#datefmt='%Y-%m-%dT%H:%M:%S'
logging.basicConfig(format='%(asctime)s [%(levelname).1s] %(filename)s L:%(lineno)03d %(message)s', datefmt='%M:%S', level=logging.INFO)

import inspect
import sys
sys.path.append('..') # use relative path from parent dir
import psana.pyalgos.generic.NDArrGenerators as ag
import numpy as np

class TestGWImageSpec(GWImageSpec):

    def KEY_USAGE(self): return 'Keys:'\
               '\n  ESC - exit'\
               '\n  O - reset original size'\
               '\n  N - set new pixmap'\
               '\n  W - set new pixmap of random shape, do not change default scene rect'\
               '\n  H - set new pixmap of random shape and change default scene rect'\
               '\n  D - delete SELECTED ROIs'\
               '\n  C - cancel add_roi command and remove currently drawn roi'\
               '\n  F - finish add_roi command and keep currently drawn roi (for polynom)'\
               '\n  L/J - load/save in jason roi parameters'\
               '\n  M - save mask'\
               '\n  ROI  = %s select from %s' % (str(self.wim.roi_name),  ', '.join(['%s-%s'%(k,n) for t,n,k in roiu.roi_tuple])) + \
               '\n  MODE = %s select from %s' % (str(self.wim.mode_name), ', '.join(['%s-%s'%(k,n) for t,n,k in roiu.mode_tuple])) + \
               '\n'

    def __init__(self, parent=None, arr=None,\
                 coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20),\
                 origin='UL', scale_ctl='HV', show_mode=0, signal_fast=True):
        GWImageSpec.__init__(self, parent=parent, image=arr, ctab=coltab,\
                                origin=origin, scale_ctl=scale_ctl, show_mode=show_mode, signal_fast=signal_fast)
        self.wim = self.wimax.wim
        print(self.KEY_USAGE())


    def test_mouse_move_event_reception(self, e):
        """Overrides method from GWView"""
        p = self.mapToScene(e.pos())
        ix, iy, v = self.cursor_on_image_pixcoords_and_value(p)
        fv = 0 if v is None else v
        self.setWindowTitle('TestGWImageSpec x=%d y=%d v=%s%s' % (ix, iy, '%.1f'%fv, 25*' '))

    def keyPressEvent(self, e):
        GWImageSpec.keyPressEvent(self, e)

        key = e.key()

        if key == Qt.Key_Escape:
            logger.debug('Close app')
            self.close()
            return

        ckey = chr(key)
        logger.debug('keyPressEvent, key = %s' % ckey)

        if key == Qt.Key_O:
            logger.info('Reset original size')
            self.wim.reset_scene_rect()

        elif key == Qt.Key_N:
            logger.info('Set new pixel map')
            s = self.wim.pmi.pixmap().size()
            img = image_with_random_peaks((s.height(), s.width()))
            self.wim.set_pixmap_from_arr(img, set_def=False)

        elif key in (Qt.Key_W, Qt.Key_H):
            change_def = key==Qt.Key_H
            logger.info('%s: change scene rect %s' % (self.wim._name, 'set new default' if change_def else ''))
            v = ag.random_standard((4,), mu=0, sigma=200, dtype=np.int32)
            rs = QRectF(v[0], v[1], v[2]+1000, v[3]+1000)
            logger.info('Set scene rect: %s' % str(rs))
            img = image_with_random_peaks((int(rs.height()), int(rs.width())))
            self.wim.mask = ct.test_mask(img)
            self.wim.set_pixmap_from_arr(img, set_def=change_def)

        elif key == Qt.Key_D:
            self.wim.delete_selected_roi()

        elif key == Qt.Key_C:
            self.wim.cancel_add_roi()

        elif key == Qt.Key_F:
            self.wim.finish()

        elif key == Qt.Key_J:
            self.wim.save_parameters_in_file()

        elif key == Qt.Key_L:
            self.wim.load_parameters_from_file()

        elif key == Qt.Key_M:
            self.wim.save_mask()

        elif ckey in roiu.roi_keys:
            i = roiu.roi_keys.index(ckey)
            self.wim.roi_type = roiu.roi_types[i]
            self.wim.roi_name = roiu.roi_names[i]
            logger.info('set roi_type: %d roi_name: %s' % (self.wim.roi_type, self.wim.roi_name))

        elif ckey in roiu.mode_keys:
            self.wim.finish()
            i = roiu.mode_keys.index(ckey)
            self.wim.mode_type = roiu.mode_types[i]
            self.wim.mode_name = roiu.mode_names[i]
            logger.info('set mode_type: %d roi_name: %s' % (self.wim.mode_type, self.wim.mode_name))

            sc = '' if self.wim.mode_type > roiu.VISIBLE else 'HV'
            self.wim.set_scale_control(scale_ctl=sc)

        logger.info(self.KEY_USAGE())

def image_with_random_peaks(shape=(500, 500)):
    from psana.pyalgos.generic.NDArrUtils import info_ndarr

    img = ag.random_standard(shape, mu=0, sigma=10)
    logger.info(info_ndarr(img, 'image_with_random_peaks  ag.random_standard'))

    peaks = ag.add_random_peaks(img, npeaks=50, amean=100, arms=50, wmean=1.5, wrms=0.3)
    ag.add_ring(img, amp=20, row=500, col=500, rad=300, sigma=50)
    return img

def test_GWImageSpec(tname):
    logger.info(sys._getframe().f_code.co_name)
    arr = image_with_random_peaks((1000, 1000))
    ctab = ct.color_table_interpolated()
    #ctab = ct.color_table_monochr256()
    #arr = np.random.random((1000, 1000))
    #ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)
    #ctab = ct.color_table_monochr256()
    #ctab = ct.color_table_interpolated()

    app = QApplication(sys.argv)
    w = TestGWImageSpec(None, arr, coltab=ctab, origin='UL', scale_ctl='HV')

    w.wim.connect_mouse_press_event(w.wim.test_mouse_press_event_reception)
    w.wim.connect_mouse_move_event(w.wim.test_mouse_move_event_reception)
    #w.connect_scene_rect_changed(w.wim.test_scene_rect_changed_reception)

    w.setWindowTitle('ex_GWImageSpec')

    w.setGeometry(20, 20, 900, 600)
    w.set_splitter_pos()

    #w.move(QCursor().pos())
    w.show()
    app.exec_()

#    del w
#    del app

#USAGE = '\nUsage: %s <tname>\n' % sys.argv[0].split('/')[-1]\
#      + '\n'.join([s for s in inspect.getsource(test_GWImageSpec).split('\n') \
#                   if "tname ==" in s or 'GWImageSpec' in s])  # s[9:]

if __name__ == "__main__":
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.info(50*'_' + '\nTest %s' % tname)
    test_GWImageSpec(tname)
    #print(USAGE)
    sys.exit('End of Test %s' % tname)

# EOF
