
from psana.graphqt.GWViewExt import *

logging.basicConfig(format='[%(levelname).1s] %(asctime)s L:%(lineno)03d %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.DEBUG)

import sys
import psana.pyalgos.generic.NDArrGenerators as ag
import numpy as np


class TestGWViewExt(GWViewExt):

    KEY_USAGE = 'Keys:'\
            '\n  ESC - exit'\
            '\n  R - reset original size'\
            '\n  U - update default rect scene to QRectF(-10, -10, 30, 30)'\
            '\n  W - change scene rect, do not change default'\
            '\n  D - change scene rect and its default'\
            '\n'

    def keyPressEvent(self, e):
        #logger.debug('keyPressEvent, key=', e.key())
        if   e.key() == Qt.Key_Escape:
            self.close()

        elif e.key() == Qt.Key_R:
            print('Reset original size')
            #self.reset_original_size()
            self.reset_scene_rect()

        elif e.key() == Qt.Key_U:
            print('Update default rect scene')
            #self.reset_original_size()
            self.reset_scene_rect(rs=QRectF(-10, -10, 30, 30))

        elif e.key() in (Qt.Key_W, Qt.Key_D):
            change_def = e.key()==Qt.Key_D
            print('change scene rect %s' % ('set new default' if change_def else ''))
            v = ag.random_standard((4,), mu=0, sigma=3, dtype=np.int)
            rs = QRectF(v[0]-5, v[1]-5, v[2]+20, v[3]+20)
            print('Set scene rect: %s' % str(rs))
            self.reset_scene_rect(rs)

        else:
            print(self.KEY_USAGE)


SCRNAME = sys.argv[0].split('/')[-1]

USAGE = '\nUsage: python %s <tname [0-8]>' %SCRNAME\
      + ' # then activate graphics window and use keyboad keys R/W/D/<Esc>'


def test_fwview(tname):
    print('%s:' % sys._getframe().f_code.co_name)
    b="background-color:yellow; border: 0px solid green"
    app = QApplication(sys.argv)
    w = None
    if   tname == '0': w=TestGWViewExt(None, rscene=QRectF(-10, -10, 30, 30), origin='DL', show_mode=3, scale_ctl='HV')
    elif tname == '1': w=TestGWViewExt(None, rscene=QRectF(-10, -10, 30, 30), origin='UL', show_mode=3, scale_ctl='HV')
    elif tname == '2': w=TestGWViewExt(None, rscene=QRectF(-10, -10, 30, 30), origin='UR', show_mode=3, scale_ctl='HV')
    elif tname == '3': w=TestGWViewExt(None, rscene=QRectF(-10, -10, 30, 30), origin='DR', show_mode=3, scale_ctl='HV')
    elif tname == '4': w=TestGWViewExt(None, rscene=QRectF(-10, -10, 30, 30), origin='DL', show_mode=3, scale_ctl='')
    elif tname == '5': w=TestGWViewExt(None, rscene=QRectF(-10, -10, 30, 30), origin='DL', show_mode=3, scale_ctl='H')
    elif tname == '6': w=TestGWViewExt(None, rscene=QRectF(-10, -10, 30, 30), origin='DL', show_mode=3, scale_ctl='V')
    elif tname == '7': w=TestGWViewExt(None, rscene=QRectF(-10, -10, 30, 30), origin='DL', show_mode=1, scale_ctl='HV')
    elif tname == '8': w=TestGWViewExt(None, rscene=QRectF(-10, -10, 30, 30), origin='DL', show_mode=3, scale_ctl='HV')
    else:
        print('test %s is not implemented' % tname)
        return

    w.connect_mouse_move_event(w.test_mouse_move_event_reception)
    w.connect_scene_rect_changed(w.test_scene_rect_changed_reception)
    w.connect_mouse_press_event(w.test_mouse_press_event_reception)

    w.setWindowTitle("TestGWViewExt")
    w.setGeometry(20, 20, 600, 600)
    w.show()

    #w.disconnect_mouse_move_event(w.test_mouse_move_event_reception)
    #w.disconnect_scene_rect_changed(w.test_scene_rect_changed_reception)
    #w.disconnect_mouse_press_event(w.test_mouse_press_event_reception)
    #w.close()

    app.exec_()
    app.quit()

    del w
    del app


if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_fwview(tname)
    print(USAGE)
    sys.exit('End of Test %s' % tname)

# EOF
