#!/usr/bin/env python

import sys
import inspect
from psana.graphqt.GWViewHist import *

logging.basicConfig(format='[%(levelname).1s] %(filename)s L:%(lineno)03d %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.DEBUG)


def test_GWViewHist(tname):
    logger.info('%s:' % sys._getframe().f_code.co_name)
    app = QApplication(sys.argv)
    w = None

    rs = QRectF(0, 0, 100, 1000)

    if   tname ==  '0': w=GWViewHist(None, rs, origin='UL', scale_ctl='V', fgcolor='white', bgcolor='gray')
    elif tname ==  '1': w=GWViewHist(None, rs, origin='DL', scale_ctl='H', fgcolor='black', bgcolor='yellow')
    elif tname ==  '2': w=GWViewHist(None, rs, origin='DR')
    elif tname ==  '3': w=GWViewHist(None, rs, origin='UR')
    elif tname ==  '4': w=GWViewHist(None, rs, origin='DR', scale_ctl='V', fgcolor='yellow', bgcolor='gray', orient='V')
    elif tname ==  '5': w=GWViewHist(None, rs, origin='DR', scale_ctl='V', fgcolor='white', orient='V')
    else:
        logger.info('test %s is not implemented' % tname)
        return

    w.print_attributes()
    w.connect_scene_rect_changed(w.test_scene_rect_changed_reception)
    w.show()
    app.exec_()

    del w
    del app


SCRNAME = sys.argv[0].split('/')[-1]
USAGE = '\nUsage: python %s <tname>\n' % SCRNAME\
      + '\n'.join([s for s in inspect.getsource(test_GWViewHist).split('\n') if "tname ==" in s])


if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.info(50*'_' + '\nTest %s' % tname)
    test_GWViewHist(tname)
    logger.info(USAGE)
    sys.exit('End of Test %s' % tname)

# EOF
