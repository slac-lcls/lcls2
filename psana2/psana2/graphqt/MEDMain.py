
"""Class :py:class:`MEDMain` is a QWidget for main window of the Mask Editor
============================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/MEDMain.py

    from psana2.graphqt.MEDMain import MEDMain

See method:

Created on 2023-09-06 by Mikhail Dubrovin
"""
import logging
logger = logging.getLogger(__name__)
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit
from PyQt5.QtCore import Qt

from psana2.detector.dir_root import DIR_DATA_TEST, DIR_REPO

from psana2.graphqt.GWLoggerStd import GWLoggerStd
from psana2.graphqt.GWImageSpec import GWImageSpec
from psana2.graphqt.MEDControl import MEDControl
from psana2.graphqt.MEDControlROI import MEDControlROI
import psana2.graphqt.MEDUtils as mu

from psana2.detector.RepoManager import RepoManager

SCRNAME = sys.argv[0].rsplit('/')[-1]

class MEDMain(QWidget):

    def __init__(self, **kwa):
        QWidget.__init__(self, parent=None)

        kwa['repoman'] = repoman = RepoManager(**kwa)
        self.wlog = GWLoggerStd(**kwa)
        repoman.save_record_at_start(SCRNAME)
        self.kwa = kwa

        self.ictab = kwa.get('ctab', 2)
        self.signal_fast = kwa.get('signal_fast', False)

        image, geo, geo_doc = mu.image_from_kwargs(**kwa)
        ctab = mu.color_table(ict=self.ictab)
        self.wisp = GWImageSpec(parent=self, image=image, ctab=ctab, signal_fast=self.signal_fast)
        self.wctl = MEDControl(parent=self, **kwa, geo=geo, geo_doc=geo_doc)
        self.wbts = MEDControlROI(parent=self, **kwa)

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

    def set_tool_tips(self):
        self.setToolTip('Mask Editor')

    def set_style(self):
        self.layout().setContentsMargins(0,0,0,0)
        #self.setStyleSheet("background-color: rgb(0, 0, 0); color: rgb(220, 220, 220);")

    def set_splitter_pos(self, fr=0.95):
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
    #loglevel = kwa.get('logmode', 'INFO').upper()
    #intlevel = logging._nameToLevel[loglevel]
    #logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d : %(message)s', level=intlevel)
    #print('kwargs: %s' % mu.ut.info_dict(kwa))

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
    from psana2.detector.dir_root import DIR_DATA_TEST
    kwa = {\
      'ndafname': DIR_DATA_TEST + '/misc/cspad2x2.1-ndarr-ave-meca6113-r0028.npy',\
      'loglevel':'INFO',\
    }
    mask_editor(**kwa)

# EOF
