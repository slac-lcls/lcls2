#!/usr/bin/env python

"""Class :py:class:`ex_GWImageAxes` - test
==========================================

Usage ::

    # Test: lcls2/psana/psana/graphqt/GWImageAxes.py

Created on 2023-05-11 by Mikhail Dubrovin
"""

from psana.graphqt.GWImageAxes import *

logging.basicConfig(format='[%(levelname).1s] %(filename)s L:%(lineno)03d %(message)s', level=logging.DEBUG)

SCRNAME = sys.argv[0].split('/')[-1]
USAGE = '\nUsage: %s' % SCRNAME

if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    app = QApplication(sys.argv)
    w = GWImageAxes(signal_fast=False) # True)
    w.setGeometry(100, 50, 800, 800)
    w.setWindowTitle('Image with two axes')
    w.show()
    app.exec_()
    del w
    del app

# EOF
