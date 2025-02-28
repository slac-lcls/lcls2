#!/usr/bin/env python

from psana.graphqt.GWImageAxesROI import *

logger = logging.getLogger(__name__)
#datefmt='%Y-%m-%dT%H:%M:%S'
logging.basicConfig(format='%(asctime)s [%(levelname).1s] %(filename)s L:%(lineno)03d %(message)s', datefmt='%M:%S', level=logging.INFO)

import inspect
import sys
sys.path.append('..') # use relative path from parent dir
import psana.pyalgos.generic.NDArrGenerators as ag
import numpy as np

class TestGWImageAxesROI(GWImageAxesROI):

    def KEY_USAGE(self): return 'Keys:'\
               '\n  ESC - exit'\
               '\n  O - reset original size'\
               '\n  N - set new pixmap'\
               '\n  W - set new pixmap of random shape, do not change default scene rect'\
               '\n  H - set new pixmap of random shape and change default scene rect'\
               '\n  K - draw test shapes'\
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
        GWImageAxesROI.__init__(self, parent=parent, image=arr, ctab=coltab,\
                                origin=origin, scale_ctl=scale_ctl, show_mode=show_mode, signal_fast=signal_fast)
        print(self.KEY_USAGE())

    def test_mouse_move_event_reception(self, e):
        """Overrides method from GWView"""
        p = self.mapToScene(e.pos())
        ix, iy, v = self.cursor_on_image_pixcoords_and_value(p)
        fv = 0 if v is None else v
        self.setWindowTitle('TestGWImageAxesROI x=%d y=%d v=%s%s' % (ix, iy, '%.1f'%fv, 25*' '))

    def test_draw_shapes(self):
        """Test draw shapes"""
        for i in range(100):
            gap = int(3*i)
            roiu.ROIPixel(view=self).add_to_scene(QPoint(1+gap, 5+gap))

        pi = QPointF(300, 100)
        itpi = roiu.ROIPixel(view=self).add_to_scene(pi)

        l0 = QLineF(QPointF(300, 600), QPointF(600, 300))
        itl0 = roiu.ROILine(view=self).add_to_scene(l0)

        r0 = QRectF(100, 200, 200, 100)
        itr0 = roiu.ROIRect(view=self).add_to_scene(r0)
        itr1 = roiu.ROIRect(view=self).add_to_scene(r0, angle_deg=30)

        p0 = QPolygonF([QPointF(500, 600), QPointF(700, 600), QPointF(700, 500), QPointF(650, 650)])
        itp0 = roiu.ROIPolygon(view=self).add_to_scene(p0)

        r0 = QRectF(300, 400, 200, 100)
        itp0 = roiu.ROIEllipse(view=self).add_to_scene(r0)
        itp1 = roiu.ROIEllipse(view=self).add_to_scene(r0, angle_deg=-30, start_angle=-20, span_angle=300)

    def test_draw_rois(self):
        """Test ROI"""
        itroi1 = roiu.create_roi(roiu.PIXEL,   view=self, pen=roiu.QPEN_DEF).add_to_scene(pos=QPointF(20, 40))
        itroi2 = roiu.create_roi(roiu.LINE,    view=self, pos=QPointF(20, 60)).add_to_scene()
        itroi3 = roiu.create_roi(roiu.RECT,    view=self, pos=QPointF(20, 80)).add_to_scene()
        itroi4 = roiu.create_roi(roiu.SQUARE , view=self, pos=QPointF(20, 100)).add_to_scene()
        itroi5 = roiu.create_roi(roiu.POLYGON, view=self, pos=QPointF(20, 120)).add_to_scene()
        itroi6 = roiu.create_roi(roiu.POLYREG, view=self, pos=QPointF(20, 140)).add_to_scene()
        itroi7 = roiu.create_roi(roiu.ELLIPSE, view=self, pos=QPointF(20, 160)).add_to_scene()
        itroi8 = roiu.create_roi(roiu.CIRCLE,  view=self, pos=QPointF(20, 180)).add_to_scene()
        itroi9 = roiu.create_roi(roiu.ARCH,    view=self, pos=QPointF(20, 200)).add_to_scene()

    def test_draw_handles(self):
        """Test Handle"""
        ithc = roiu.select_handle(roiu.CENTER,    view=self, roi=None, pos=QPointF(50,20)).add_to_scene()
        itho = roiu.select_handle(roiu.ORIGIN,    view=self, roi=None, pos=QPointF(80,20)).add_to_scene()
        itht = roiu.select_handle(roiu.TRANSLATE, view=self, roi=None, pos=QPointF(110,20)).add_to_scene()
        ithr = roiu.select_handle(roiu.ROTATE,    view=self, roi=None, pos=QPointF(140,20)).add_to_scene()
        iths = roiu.select_handle(roiu.SCALE,     view=self, roi=None, pos=QPointF(170,20)).add_to_scene()
        ithm = roiu.select_handle(roiu.MENU,      view=self, roi=None, pos=QPointF(200,20)).add_to_scene()
        ith1 = roiu.select_handle(roiu.OTHER,     view=self, roi=None, pos=QPointF(230,20), shhand=1).add_to_scene()
        ith2 = roiu.select_handle(roiu.OTHER,     view=self, roi=None, pos=QPointF(260,20), shhand=2).add_to_scene()
        ith3 = roiu.select_handle(roiu.OTHER,     view=self, roi=None, pos=QPointF(290,20), shhand=3).add_to_scene()

    def keyPressEvent(self, e):
        GWImageAxesROI.keyPressEvent(self, e)

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

        elif key == Qt.Key_K:
            self.test_draw_shapes()
            self.test_draw_rois()
            self.test_draw_handles()

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

def test_gwimageaxesroi(tname):
    logger.info(sys._getframe().f_code.co_name)
    arr = image_with_random_peaks((1000, 1000))
    ctab = ct.color_table_monochr256()
    #arr = np.random.random((1000, 1000))
    #ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)
    #ctab = ct.color_table_monochr256()
    #ctab = ct.color_table_interpolated()

    app = QApplication(sys.argv)
    w = None
    if   tname == '0': w = TestGWImageAxesROI(None, arr, coltab=ctab, origin='UL', scale_ctl='HV')
    elif tname == '1': w = TestGWImageAxesROI(None, arr, coltab=ctab, origin='UL', scale_ctl='H')
    elif tname == '2': w = TestGWImageAxesROI(None, arr, coltab=ctab, origin='UL', scale_ctl='V')
    elif tname == '3': w = TestGWImageAxesROI(None, arr, coltab=ctab, origin='UL', scale_ctl='')
    elif tname == '4':
        arrct = ct.array_for_color_bar(orient='H')
        w = TestGWImageAxesROI(None, arrct, coltab=None, origin='UL', scale_ctl='H')
        w.setGeometry(50, 50, 500, 40)
    elif tname == '5':
        arrct = ct.array_for_color_bar(orient='V')
        w = TestGWImageAxesROI(None, arrct, coltab=None, origin='UL', scale_ctl='V')
        w.setGeometry(50, 50, 40, 500)
    elif tname == '6':
        ctab = next_color_table(ict=7)
        arrct = ct.array_for_color_bar(ctab, orient='H')
        w = TestGWImageAxesROI(None, arrct, coltab=None, origin='UL', scale_ctl='H')
        w.setGeometry(50, 50, 500, 40)
    elif tname == '7':
        a = np.arange(15).reshape((5, 3))
        w = TestGWImageAxesROI(None, a, coltab=ctab, origin='UL', scale_ctl='HV')
    else:
        logger.info('test %s IS NOT IMPLEMENTED' % tname)
        return

    w.wim.connect_mouse_press_event(w.wim.test_mouse_press_event_reception)
    w.wim.connect_mouse_move_event(w.wim.test_mouse_move_event_reception)
    #w.connect_scene_rect_changed(w.wim.test_scene_rect_changed_reception)

    w.setWindowTitle('ex_GWImageAxesROI')

    w.setGeometry(20, 20, 600, 600)
    #w.move(QCursor().pos())
    w.show()
    app.exec_()

#    del w
#    del app

USAGE = '\nUsage: %s <tname>\n' % sys.argv[0].split('/')[-1]\
      + '\n'.join([s for s in inspect.getsource(test_gwimageaxesroi).split('\n') \
                   if "tname ==" in s or 'GWImageAxesROI' in s])  # s[9:]

if __name__ == "__main__":
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.info(50*'_' + '\nTest %s' % tname)
    test_gwimageaxesroi(tname)
    print(USAGE)
    sys.exit('End of Test %s' % tname)

# EOF
