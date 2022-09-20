
from psana.graphqt.GWViewAxis import *
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] %(asctime)s L:%(lineno)03d %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.DEBUG)

import sys
import psana.pyalgos.generic.NDArrGenerators as ag

class TestGWViewAxis(GWViewAxis):

    def key_usage(self):
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  R - set random axis limits'\
               '\n'

    def keyPressEvent(self, e):
        #logger.debug('keyPressEvent, key=', e.key())
        if   e.key() == Qt.Key_Escape:
            self.close()

        elif e.key() == Qt.Key_R:
            print('TestGWViewAxis: set random limits for axis')
            v = ag.random_standard((2,), mu=0, sigma=20, dtype=ag.np.int)
            vmin, vmax = v[0], v[0] + 100
            self.set_axis_limits(vmin, vmax)

        else:
            print(self.key_usage())


def test_GWViewAxis(tname):
    print('%s:' % sys._getframe().f_code.co_name)
    app = QApplication(sys.argv)
    w = None
    rs=QRectF(0, 0, 1000, 100)
    if   tname ==  '0': w=TestGWViewAxis(None, rs, side='D', origin='UL')
    elif tname ==  '1': w=TestGWViewAxis(None, rs, side='U', origin='UL')
    elif tname ==  '2': w=TestGWViewAxis(None, rs, side='L', origin='UL')
    elif tname ==  '3': w=TestGWViewAxis(None, rs, side='R', origin='UL')

    elif tname == '10': w=TestGWViewAxis(None, rs, side='D', origin='DL')
    elif tname == '11': w=TestGWViewAxis(None, rs, side='U', origin='DL')
    elif tname == '12': w=TestGWViewAxis(None, rs, side='L', origin='DL')
    elif tname == '13': w=TestGWViewAxis(None, rs, side='R', origin='DL')

    elif tname == '20': w=TestGWViewAxis(None, rs, side='D', origin='DR')
    elif tname == '21': w=TestGWViewAxis(None, rs, side='U', origin='DR')
    elif tname == '22': w=TestGWViewAxis(None, rs, side='L', origin='DR')
    elif tname == '23': w=TestGWViewAxis(None, rs, side='R', origin='DR')

    elif tname == '30': w=TestGWViewAxis(None, rs, side='D', origin='UR')
    elif tname == '31': w=TestGWViewAxis(None, rs, side='U', origin='UR')
    elif tname == '32': w=TestGWViewAxis(None, rs, side='L', origin='UR')
    elif tname == '33': w=TestGWViewAxis(None, rs, side='R', origin='UR')

    elif tname == '40': w=TestGWViewAxis(None, rs, side='D', origin='UL', fgcolor='red', bgcolor='yellow')
    elif tname == '41': w=TestGWViewAxis(None, rs, side='U', origin='UL', fgcolor='red', bgcolor='yellow')
    elif tname == '42': w=TestGWViewAxis(None, rs, side='L', origin='UL', fgcolor='red', bgcolor='yellow')
    elif tname == '43': w=TestGWViewAxis(None, rs, side='R', origin='UL', fgcolor='red', bgcolor='yellow')
    else:
        print('test %s is not implemented' % tname)
        return

    print(w.info_attributes())
    #w.connect_axes_limits_changed(w.test_axes_limits_changed_reception)
    #w.disconnect_axes_limits_changed(w.test_axes_limits_changed_reception)
    w.show()
    app.exec_()

    del w
    del app


if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_GWViewAxis(tname)
    sys.exit('End of Test %s' % tname)

# EOF
