#------------------------------
"""Class :py:class:`QWTree` is a QTreeView->QWidget for tree model
===================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/QWTree.py

    from psana.graphqt.QWTree import QWTree
    w = QWTree()

Created on 2019-11-12 by Mikhail Dubrovin
"""
#------------------------------
import logging
logger = logging.getLogger(__name__)

import sys
from PyQt5.QtWidgets import QApplication
from psana.graphqt.H5VQWTree import Qt, H5VQWTree
#------------------------------

#fname = '/reg/g/psdm/detector/data_test/hdf5/amox27716-r0100-e060000-single-node.h5'
FNAME_TEST = '/reg/g/psdm/detector/calib/jungfrau/jungfrau-171113-154920171025-3d00fb.h5'

#------------------------------

def hdf5explorer(parser) :
    import sys
    fmt = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=fmt, datefmt='%H:%M:%S', level=logging.DEBUG)

    (popts, pargs) = parser.parse_args() # TRICK! this line allows -h or --help potion !!!
    opts = vars(popts)
    kwargs = opts

    fname = pargs[0] if len(pargs) else FNAME_TEST
    kwargs['fname'] = fname

    print('Open file', fname)

    #sys.exit('TEST EXIT')

    app = QApplication(sys.argv)
    w = H5VQWTree(**kwargs)
    #w.setGeometry(10, 25, 200, 600)
    w.setWindowTitle('HDF5 explorer')

    w.move(50,20)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------

#if __name__ == "__main__" :
#    hdf5explorer()

#------------------------------
