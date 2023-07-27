
"""Class :py:class:`IVMain` is a QWidget for main window of Image Viewer
========================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/IVMain.py

    from psana.graphqt.IVMain import IVMain

See method:

Created on 2021-06-14 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit
from PyQt5.QtCore import Qt
from psana.graphqt.IVControl import IVControl, image_from_ndarray
from psana.graphqt.CMConfigParameters import cp
from psana.graphqt.FWViewImage import FWViewImage, ct
from psana.graphqt.IVImageAxes import IVImageAxes, test_image
from psana.graphqt.IVSpectrum import IVSpectrum

SCRNAME = sys.argv[0].rsplit('/')[-1]

class IVMain(QWidget):

    def __init__(self, **kwargs):
        QWidget.__init__(self, parent=None)

        cp.ivmain = self

        self.proc_kwargs(**kwargs)

        fname = kwargs.get('fname', None)
        last_selected_fname = cp.last_selected_fname.value()
        last_selected_data = cp.last_selected_data

        if fname is None and last_selected_fname is not None: fname = last_selected_fname
        image = kwargs.get('image', None)
        image = image if image is not None else\
                image_from_ndarray(last_selected_data) if last_selected_data is not None else\
                None
        if image is None: image = test_image(shape=(32,32))
        else: fname = None

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

        if fname is not None:
            self.wctrl.wfnm_nda.but.setText(fname)
            self.wctrl.on_changed_fname_nda(fname)


    def proc_kwargs(self, **kwargs):
        #print_kwargs(kwargs)
        loglevel   = kwargs.get('loglevel', 'DEBUG').upper()
        logdir     = kwargs.get('logdir', './')
        savelog    = kwargs.get('savelog', False)
        self.wlog  = kwargs.get('wlog', None)
        self.signal_fast = kwargs.get('signal_fast', True)


    def set_tool_tips(self):
        self.setToolTip('Image Viewer')


    def set_style(self):
        self.layout().setContentsMargins(0,0,0,0)
        self.wctrl.setFixedHeight(80)
        self.wspec.setMaximumWidth(300)


    def closeEvent(self, e):
        logger.debug('closeEvent')
        QWidget.closeEvent(self, e)
        cp.ivmain = None


def image_viewer(**kwargs):
    repodir = kwargs.get('repodir', './work')
    loglevel = kwargs.get('loglevel', 'DEBUG').upper()
    intlevel = logging._nameToLevel[loglevel]
    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d : %(message)s', level=intlevel)

    if kwargs.get('rec_at_start', True):
        from psana.detector.RepoManager import RepoManager
        RepoManager(dirrepo=repodir).save_record_at_start(SCRNAME)

    a = QApplication(sys.argv)
    w = IVMain(**kwargs)
    w.setGeometry(10, 100, 1000, 800)
    w.move(50,20)
    w.show()
    w.setWindowTitle('Image Viewer')
    a.exec_()
    del w
    del a


def do_main(**kwargs):
    image_viewer(**kwargs)


def do_work(dt_sec=1, nloops=20):
    from time import sleep
    for i in range(nloops):
        print('do_work loop: %3d' % i)
        sleep(dt_sec)


def do_python_threads(**kwargs):
    """Test python's multi-threading to run data processing paralel to qt widget.
    """
    import threading
    #tg = threading.Thread(target=image_viewer)
    #tg.daemon = True
    #tg.start()
    tw = threading.Thread(target=do_work)
    tw.start()
    image_viewer(**kwargs)


if __name__ == "__main__":
    from psana.detector.dir_root import DIR_DATA_TEST
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
    kwargs = {\
      'fname': DIR_DATA_TEST + '/misc/cspad2x2.1-ndarr-ave-meca6113-r0028.npy',\
      'loglevel':'DEBUG',\
    }

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if   tname == '0': do_main(**kwargs)
    elif tname == '1': do_python_threads(**kwargs)
    else: logger.debug('Not-implemented test "%s"' % tname)

# EOF
