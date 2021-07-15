
"""Class :py:class:`IVMain` is a QWidget for main window of Image Viewer
========================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/IVMain.py

    from psana.graphqt.IVMain import IVMain

See method:

Created on 2021-06-14 by Mikhail Dubrovin
"""

import logging
#logger = logging.getLogger(__name__)
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit
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

        fname = kwargs.get('fname', None)
        image = kwargs.get('image', test_image(shape=(32,32)))
        ctab  = kwargs.get('ctab', ct.color_table_default())

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

        if fname is not None: self.wctrl.on_changed_fname_nda(fname)
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


def image_viewer(**kwargs):

    loglevel = kwargs.get('loglevel', 'DEBUG').upper()
    intlevel = logging._nameToLevel[loglevel]
    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d : %(message)s', level=intlevel)

    a = QApplication(sys.argv)
    w = IVMain(**kwargs)
    w.setGeometry(10, 100, 1000, 800)
    w.move(50,20)
    w.show()
    w.setWindowTitle('Image Viewer')
    #if __name__ == "__main__": w.wctrl.but_tabs_is_visible(False)
    a.exec_()
    del w
    del a
    #return a,w


def do_main(**kwargs):
    #logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d : %(message)s', level=logging.INFO)
    image_viewer(**kwargs)


def do_work(dt_sec=1, nloops=20):
    from time import sleep
    for i in range(nloops):
        print('do_work loop: %3d' % i)
        sleep(dt_sec)


def do_python_threads(**kwargs):
    import threading
    #logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d : %(message)s', level=logging.INFO)

    #tg = threading.Thread(target=image_viewer)
    #tg.daemon = True
    #tg.start()

    tw = threading.Thread(target=do_work)
    #tw.daemon = True
    tw.start()

    image_viewer(**kwargs)


if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
    kwargs = {\
      'fname':'/cds/group/psdm/detector/data2_test/misc/cspad2x2.1-ndarr-ave-meca6113-r0028.npy',\
      'loglevel':'INFO',\
    }
    #do_main(**kwargs)
    do_python_threads(**kwargs)

# EOF
