
"""Class :py:class:`MEDMain` is a QWidget for main window of the Mask Editor
============================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/MEDMain.py

    from psana.graphqt.MEDMain import MEDMain

See method:

Created on 2023-09-06 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit
from PyQt5.QtCore import Qt

from psana.detector.dir_root import DIR_DATA_TEST, DIR_REPO

from psana.graphqt.GWLoggerStd import GWLoggerStd
from psana.graphqt.GWImageSpec import GWImageSpec
from psana.graphqt.MEDControl import MEDControl
from psana.graphqt.MEDControlROI import MEDControlROI
import psana.graphqt.MEDUtils as mu

SCRNAME = sys.argv[0].rsplit('/')[-1]

class MEDMain(QWidget):

    def __init__(self, **kwa):
        QWidget.__init__(self, parent=None)

        self.proc_kwargs(**kwa)

        image = mu.image_from_kwargs(**kwa)
        ctab = mu.color_table(ict=self.ictab)
        self.wisp = GWImageSpec(parent=self, image=image, ctab=ctab, signal_fast=self.signal_fast)
        self.wlog = GWLoggerStd()
        self.wctl = MEDControl(parent=self, **kwa)
        self.wbts = MEDControlROI(parent=self)

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.wbts)
        self.hbox.addWidget(self.wisp)
        self.hbox.setContentsMargins(0,0,0,0)
        self.wbox = QWidget()
        self.wbox.setLayout(self.hbox)

        self.vspl = QSplitter(Qt.Vertical)
        self.vspl.addWidget(self.wbox)
        self.vspl.addWidget(self.wlog)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.wctl)
        self.vbox.addWidget(self.vspl)
        self.setLayout(self.vbox)

        self.set_style()
        self.set_tool_tips()

    def proc_kwargs(self, **kwa):
        #print_kwa(kwa)
        self.kwa   = kwa
        loglevel   = kwa.get('loglevel', 'DEBUG').upper()
        logdir     = kwa.get('logdir', './')
        savelog    = kwa.get('savelog', False)
        self.wlog  = kwa.get('wlog', None)
        self.ictab = kwa.get('ictab', 2)
        self.signal_fast = kwa.get('signal_fast', False)

    def set_tool_tips(self):
        self.setToolTip('Mask Editor')

    def set_style(self):
        self.layout().setContentsMargins(0,0,0,0)
        #self.wisp.setFixedHeight(400)
        #self.setStyleSheet("background-color: rgb(0, 0, 0); color: rgb(220, 220, 220);")

    def set_splitter_pos(self, fr=0.95):
        #self.wlog.setMinimumHeight(100)
        h = self.height()
        s = int(fr*h)
        self.vspl.setSizes((s, h-s)) # spl_pos = self.vspl.sizes()[0]
        self.wisp.set_splitter_pos(fr=0.8)

    def resizeEvent(self, e):
        QWidget.resizeEvent(self, e)
        self.set_splitter_pos()

    def closeEvent(self, e):
        logger.debug('closeEvent')
        QWidget.closeEvent(self, e)


def mask_editor(**kwa):
    repodir  = kwa.get('repodir', './work')
    loglevel = kwa.get('loglevel', 'INFO').upper()
    intlevel = logging._nameToLevel[loglevel]
    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d : %(message)s', level=intlevel)

    from psana.detector.Utils import info_dict  #, info_command_line, info_namespace, info_parser_arguments, str_tstamp
    print('kwargs: %s' % info_dict(kwa))

    if kwa.get('rec_at_start', True):
        from psana.detector.RepoManager import RepoManager
        RepoManager(dirrepo=repodir).save_record_at_start(SCRNAME)

    a = QApplication(sys.argv)
    w = MEDMain(**kwa)
    w.setGeometry(10, 100, 1000, 800)
    w.set_splitter_pos()
    w.move(50,20)
    w.show()
    w.setWindowTitle('Image Viewer')
    a.exec_()
    del w
    del a


if __name__ == "__main__":
    from psana.detector.dir_root import DIR_DATA_TEST
    #import os
    #os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
    kwa = {\
      'ndafname': DIR_DATA_TEST + '/misc/cspad2x2.1-ndarr-ave-meca6113-r0028.npy',\
      'loglevel':'INFO',\
    }

    mask_editor(**kwa)

# EOF
