#!/usr/bin/env python

import sys
import inspect
from psana.graphqt.GWViewHist import *
import psana.pyalgos.generic.NDArrGenerators as ag

logging.basicConfig(format='[%(levelname).1s] %(filename)s L:%(lineno)03d %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.DEBUG)

class TestGWViewHist(GWViewHist):

    KEY_USAGE = 'Keys:'\
            '\n  ESC - exit'\
            '\n  R - reset original/default size'\
            '\n  H - set new histogram'\
            '\n  W - change scene rect, do not change default'\
            '\n  D - change scene rect and its default'\
            '\n'

    def keyPressEvent(self, e):
        #logger.debug('keyPressEvent, key=', e.key())
        if   e.key() == Qt.Key_Escape:
            self.close()

        elif e.key() == Qt.Key_R:
            print('Reset original size')
            self.reset_scene_rect()

        elif e.key() == Qt.Key_H:
            print('Set new histogram')
            self.update_my_scene(hbins=test_histogram())
            #self.reset_scene_rect()

        elif e.key() in (Qt.Key_W, Qt.Key_D):
            change_def = e.key()==Qt.Key_D
            print('change scene rect %s' % ('set new default' if change_def else ''))
            v = ag.random_standard((4,), mu=0, sigma=10, dtype=np.int)
            w, h = (1200, 120) if self.hist.orient == 'H' else (120, 1200)
            rs = QRectF(v[0], v[1], v[2]+w, v[3]+h)
            print('Set scene rect: %s' % str(rs))
            if change_def:
                self.reset_scene_rect(rs)
            else:
                self.fit_in_view(rs)

        else:
            print(self.KEY_USAGE)


def test_gwviewhist(tname):
    logger.info('%s:' % sys._getframe().f_code.co_name)
    app = QApplication(sys.argv)
    w = None

    rsh = QRectF(0, 0, 1200,  200)
    rsv = QRectF(0, 0,  200, 1200)

    if   tname ==  '0': w=TestGWViewHist(None, rsh, origin='UL', scale_ctl='H', fgcolor='white', bgcolor='gray')
    elif tname ==  '1': w=TestGWViewHist(None, rsh, origin='DL', scale_ctl='H', fgcolor='black', bgcolor='yellow')
    elif tname ==  '2': w=TestGWViewHist(None, rsh, origin='DR')
    elif tname ==  '3': w=TestGWViewHist(None, rsh, origin='UR')
    elif tname ==  '4': w=TestGWViewHist(None, rsv, origin='DL', scale_ctl='V', fgcolor='yellow', bgcolor='gray', orient='V')
    elif tname ==  '5': w=TestGWViewHist(None, rsv, origin='DR', scale_ctl='V', fgcolor='white', orient='V', auto_limits='H')
    else:
        logger.info('test %s is not implemented' % tname)
        return

    w.print_attributes()
    w.connect_scene_rect_changed(w.test_scene_rect_changed_reception)
    w.connect_mouse_move_event(w.test_mouse_move_event_reception)
    w.connect_mouse_press_event(w.test_mouse_press_event_reception)
    w.show()
    app.exec_()

    #w.disconnect_mouse_move_event(w.test_mouse_move_event_reception)
    #w.disconnect_scene_rect_changed(w.test_scene_rect_changed_reception)
    #w.disconnect_mouse_press_event(w.test_mouse_press_event_reception)
    #w.close()

    del w
    del app


SCRNAME = sys.argv[0].split('/')[-1]
USAGE = '\nUsage: python %s <tname>\n' % SCRNAME\
      + '\n'.join([s for s in inspect.getsource(test_gwviewhist).split('\n') if "tname ==" in s])


if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.info(50*'_' + '\nTest %s' % tname)
    test_gwviewhist(tname)
    logger.info(USAGE)
    sys.exit('End of Test %s' % tname)

# EOF
