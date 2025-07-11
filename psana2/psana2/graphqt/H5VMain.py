
"""Class :py:class:`H5VMain` is a QWidget for main window of hdf5viewer
========================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/H5VMain.py

    from psana2.graphqt.H5VMain import H5VMain

See method: hdf5explorer

Created on 2019-11-12 by Mikhail Dubrovin
"""

import logging
#logger = logging.getLogger(__name__)

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit
from psana2.graphqt.QWLoggerStd import QWLoggerStd

from psana2.graphqt.H5VControl import H5VControl
from psana2.graphqt.H5VQWTree import Qt, H5VQWTree
from psana2.graphqt.CMConfigParameters import cp
from psana2.pyalgos.generic.Utils import print_kwargs, is_in_command_line
from psana2.detector.RepoManager import RepoManager

SCRNAME = sys.argv[0].rsplit('/')[-1]

class H5VMain(QWidget):

    def __init__(self, **kwargs):
        QWidget.__init__(self, parent=None)

        cp.h5vmain = self

        self.proc_kwargs(**kwargs)
        logdir = cp.log_prefix.value()

        kwargs['parent'] = self

        self.wlog = kwargs.get('wlog', cp.wlog)
        if self.wlog is None: self.wlog = QWLoggerStd(cp, show_buttons=False)

        self.wtree = H5VQWTree(**kwargs)
        self.wctrl = H5VControl(**kwargs)
        self.wtree.wctrl = self.wctrl

        self.hspl = QSplitter(Qt.Horizontal)
        self.hspl.addWidget(self.wtree)
        if cp.wlog is None: self.hspl.addWidget(self.wlog)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.wctrl)
        self.vbox.addWidget(self.hspl)

        self.setLayout(self.vbox)

        self.set_style()
        self.set_tool_tips()

        if kwargs.get('rec_at_start', False):
            RepoManager(dirrepo=logdir).save_record_at_start(SCRNAME)

        #self.connect_signals_to_slots()


    def proc_kwargs(self, **kwargs):
        print_kwargs(kwargs)
        loglevel   = kwargs.get('loglevel', 'DEBUG').upper()
        logdir     = kwargs.get('logdir', './')
        savelog    = kwargs.get('savelog', False)
        if is_in_command_line('-l', '--loglevel'): cp.log_level.setValue(loglevel)
        cp.log_prefix.setValue(logdir)
        cp.save_log_at_exit.setValue(savelog)


    def connect_signals_to_slots(self):
        pass
        #self.connect(self.wbut.but_reset, QtCore.SIGNAL('clicked()'), self.on_but_reset)
        #self.connect(self.wbut.but_save,  QtCore.SIGNAL('clicked()'), self.on_but_save)


    def set_tool_tips(self):
        self.setToolTip('hdf5 explorer')


    def set_style(self):
        self.setGeometry(50, 50, 500, 600)
        self.layout().setContentsMargins(0,0,0,0)

        self.wctrl.setFixedHeight(50)

    def closeEvent(self, e):
        QWidget.closeEvent(self, e)
        cp.h5vmain = None


def hdf5explorer(**kwargs):
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
    #fmt = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    #logging.basicConfig(format=fmt, datefmt='%H:%M:%S', level=logging.DEBUG)

    a = QApplication(sys.argv)
    w = H5VMain(**kwargs)
    w.setGeometry(10, 25, 900, 700)
    w.setWindowTitle('HDF5 explorer')
    w.move(50,20)
    w.show()
    a.exec_()
    del w
    del a


if __name__ == "__main__":
    import os
    kwargs = {\
      'fname':'/reg/g/psdm/detector/calib/DEPRECATED/jungfrau/jungfrau-171113-154920171025-3d00fb.h5',\
      'loglevel':'INFO',\
      'logdir':'%s/hdf5explorer-log' % os.path.expanduser('~'),\
      'savelog':True}
    hdf5explorer(**kwargs)

# EOF
