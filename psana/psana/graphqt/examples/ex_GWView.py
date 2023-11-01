#!/usr/bin/env python

import sys
from psana.graphqt.GWView import *

logging.basicConfig(format='[%(levelname).1s] %(name)s L:%(lineno)03d %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.DEBUG)

SCRNAME = sys.argv[0].split('/')[-1]
USAGE = '\nUsage: python %s <tname [0,3]>' %SCRNAME\
      + '\n   where tname=0/1/2/3 stands for scale_ctl "HV"/"H"/"V"/"", respectively'

def test_gwview(tname):
    #print('%s:' % sys._getframe().f_code.co_name)
    app = QApplication(sys.argv)
    w=GWView(rscene=QRectF(-10, -10, 30, 30),\
             scale_ctl=('HV', 'H', 'V', '')[int(tname)],\
             show_mode=0o377)
    w.setWindowTitle('ex_GWView')
    w.setGeometry(20, 20, 600, 600)
    w.show()
    app.exec_()
    app.quit()
    del w
    del app

if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_gwview(tname)
    print(USAGE)
    sys.exit('End of Test %s' % tname)

# EOF
