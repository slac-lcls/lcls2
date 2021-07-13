
"""Class :py:class:`IVMain` is a QWidget for main window of hdf5viewer 
========================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/IVMain.py

    from psana.graphqt.IVMain import IVMain

See method:

Created on 2021-06-14 by Mikhail Dubrovin
"""

import logging
#logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit
from PyQt5.QtCore import Qt
from psana.graphqt.IVControl import IVControl
from psana.graphqt.CMConfigParameters import cp
from psana.graphqt.FWViewImage import FWViewImage, ct
from psana.graphqt.IVImageAxes import IVImageAxes, test_image
from psana.graphqt.IVSpectrum import IVSpectrum

class IVMain(QWidget):

    def __init__(self, **kwargs):
        QWidget.__init__(self, parent=None)
        #self._name = self.__class__.__name__

        cp.ivmain = self

        self.proc_kwargs(**kwargs)

        image = kwargs.get('image', test_image(shape=(50,50)))
        ctab = kwargs.get('ctab', ct.color_table_default())

        self.wimage= IVImageAxes(parent=self, image=image, origin='UL', scale_ctl='HV', coltab=ctab, signal_fast=self.signal_fast)
        self.wspec = IVSpectrum(signal_fast=self.signal_fast, image=image, ctab=ctab)
        self.wctrl = IVControl(**kwargs)
        self.hspl = QSplitter(Qt.Horizontal)
        self.hspl.addWidget(self.wimage)
        self.hspl.addWidget(self.wspec)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.wctrl)
        self.vbox.addWidget(self.hspl)
        self.setLayout(self.vbox)

        self.set_style()
        self.set_tool_tips()

        #self.connect_signals_to_slots()


    def proc_kwargs(self, **kwargs):
        #print_kwargs(kwargs)
        loglevel   = kwargs.get('loglevel', 'DEBUG').upper()
        logdir     = kwargs.get('logdir', './')
        savelog    = kwargs.get('savelog', False)
        self.wlog  = kwargs.get('wlog', None)
        self.signal_fast = kwargs.get('signal_fast', True)
        #if is_in_command_line('-l', '--loglevel'): cp.log_level.setValue(loglevel)
        #if is_in_command_line('-S', '--saveloglogdir'):
        #if is_in_command_line('-L', '--logdir'):
        #cp.log_prefix.setValue(logdir)
        #cp.save_log_at_exit.setValue(savelog)


    def connect_signals_to_slots(self):
        pass
        #self.connect(self.wbut.but_reset, QtCore.SIGNAL('clicked()'), self.on_but_reset)
        #self.connect(self.wbut.but_save,  QtCore.SIGNAL('clicked()'), self.on_but_save)


    def set_tool_tips(self):
        self.setToolTip('Image Viewer')


    def set_style(self):
        self.layout().setContentsMargins(0,0,0,0)
        self.wctrl.setFixedHeight(80)
        self.wspec.setMaximumWidth(300)


    def closeEvent(self, e):
        #logger.debug('closeEvent')
        QWidget.closeEvent(self, e)
        cp.ivmain = None


def do_main(**kwargs):

    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d : %(message)s', level=logging.DEBUG)#INFO)

    import sys
    from PyQt5.QtWidgets import QApplication

    a = QApplication(sys.argv)
    w = IVMain(**kwargs)
    w.setGeometry(10, 100, 1000, 800)
    w.setWindowTitle('Image Viewer')
    #if __name__ == "__main__": w.wctrl.but_tabs_is_visible(False)

    w.move(50,20)
    w.show()
    a.exec_()
    del w
    del a


if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    kwargs = {\
      'fname':'/reg/g/psdm/detector/calib/jungfrau/jungfrau-171113-154920171025-3d00fb.h5',\
      'loglevel':'INFO',\
      'logdir':'%s/hdf5explorer-log' % os.path.expanduser('~'),\
      'savelog':True}
    do_main(**kwargs)

# EOF
