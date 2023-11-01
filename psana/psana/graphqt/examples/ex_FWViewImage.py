
from psana.graphqt.FWViewImage import *

import sys
sys.path.append('..') # use relative path from parent dir
import psana.pyalgos.generic.NDArrGenerators as ag
import numpy as np


class TestFWViewImage(FWViewImage):

    KEY_USAGE = 'Keys:'\
               '\n  ESC - exit'\
               '\n  R - reset original size'\
               '\n  N - set new pixmap'\
               '\n  W - set new pixmap of random shape, do not change default scene rect'\
               '\n  D - set new pixmap of random shape and change default scene rect'\
               '\n'

    def test_mouse_move_event_reception(self, e):
        """Overrides method from FWView"""
        p = self.mapToScene(e.pos())
        ix, iy, v = self.cursor_on_image_pixcoords_and_value(p)
        fv = 0 if v is None else v
        self.setWindowTitle('TestFWViewImage x=%d y=%d v=%s%s' % (ix, iy, '%.1f'%fv, 25*' '))


    def keyPressEvent(self, e):
        #print('keyPressEvent, key=', e.key())
        if e.key() == Qt.Key_Escape:
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
            print(self.KEY_USAGE)


def image_with_random_peaks(shape=(500, 500)):
    from psana.pyalgos.generic.NDArrUtils import print_ndarr

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
    if   tname == '0': w = TestFWViewImage(None, arr, coltab=ctab, origin='UL', scale_ctl='HV')
    elif tname == '1': w = TestFWViewImage(None, arr, coltab=ctab, origin='UL', scale_ctl='H')
    elif tname == '2': w = TestFWViewImage(None, arr, coltab=ctab, origin='UL', scale_ctl='V')
    elif tname == '3': w = TestFWViewImage(None, arr, coltab=ctab, origin='UL', scale_ctl='')
    elif tname == '4':
        arrct = ct.array_for_color_bar(orient='H')
        w = TestFWViewImage(None, arrct, coltab=None, origin='UL', scale_ctl='H')
        w.setGeometry(50, 50, 500, 40)
    elif tname == '5':
        arrct = ct.array_for_color_bar(orient='V')
        w = TestFWViewImage(None, arrct, coltab=None, origin='UL', scale_ctl='V')
        w.setGeometry(50, 50, 40, 500)
    elif tname == '6':
        #ctab= ct.color_table_rainbow(ncolors=1000, hang1=0, hang2=360)
        #ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)
        #ctab = ct.color_table_monochr256()
        ctab = ct.color_table_interpolated()
        arrct = ct.array_for_color_bar(ctab, orient='H')
        w = TestFWViewImage(None, arrct, coltab=None, origin='UL', scale_ctl='H')
        w.setGeometry(50, 50, 500, 40)
    elif tname == '7':
        a = np.arange(15).reshape((5, 3))
        w = TestFWViewImage(None, a, coltab=ctab, origin='UL', scale_ctl='HV')
    else:
        print('test %s is not implemented' % tname)
        return

    w.connect_mouse_press_event(w.test_mouse_press_event_reception)
    w.connect_mouse_move_event(w.test_mouse_move_event_reception)
    w.connect_scene_rect_changed(w.test_scene_rect_changed_reception)

    w.show()
    app.exec_()

    del w
    del app


if __name__ == "__main__":
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_wfviewimage(tname)
    sys.exit('End of Test %s' % tname)

# EOF
