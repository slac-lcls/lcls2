
"""Class :py:class:`FMWMainLS1` is a QWidget for File Manager of LCLS1
========================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/FMW1Main.py

    from psana.graphqt.FMW1Main import FMW1Main

See method:

Created on 2021-07-27 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit
from PyQt5.QtCore import Qt
from psana.graphqt.CMConfigParameters import cp, dir_calib, expname_def
from psana.graphqt.FSTree import FSTree
from psana.graphqt.FMW1Control import FMW1Control

class FMW1Main(QWidget):

    def __init__(self, **kwa):

        parent = kwa.get('parent', None)
 
        QWidget.__init__(self, parent=parent)

        cp.fmw1main = self

        self.proc_kwargs(**kwa)

        expname = kwa.get('expname', expname_def())
        tdir = dir_calib(expname)
        logger.debug('topdir: %s' % tdir)

        self.wfstree = FSTree(parent=self,\
               topdir=tdir,\
               is_selectable_dir=False,\
               selectable_ptrns=['.data'],\
               unselectable_ptrns=['HISTORY','.data~']\
              )

        self.wctrl = FMW1Control(parent=self, expname=expname)

        #self.wrhs = QTextEdit('placeholder')

        self.hspl = QSplitter(Qt.Horizontal)
        self.hspl.addWidget(self.wfstree)
        #self.hspl.addWidget(self.wrhs)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.wctrl)
        self.vbox.addWidget(self.hspl)
        self.setLayout(self.vbox)

        self.set_style()
        self.set_tool_tips()


    def proc_kwargs(self, **kwa):
        #print_kwa(kwa)
        loglevel   = kwa.get('loglevel', 'DEBUG').upper()
        logdir     = kwa.get('logdir', './')
        savelog    = kwa.get('savelog', False)


    def set_tool_tips(self):
        self.setToolTip('File Manager for LCLS1')


    def set_style(self):
        self.layout().setContentsMargins(0,0,0,0)
        self.wctrl.setFixedHeight(40)


    def closeEvent(self, e):
        logger.debug('closeEvent')
        QWidget.closeEvent(self, e)
        cp.fmw1main = None


def file_manager_lcls1(**kwa):
    loglevel = kwa.get('loglevel', 'DEBUG').upper()
    intlevel = logging._nameToLevel[loglevel]
    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d: %(message)s', level=intlevel)

    a = QApplication(sys.argv)
    w = FMW1Main(**kwa)
    w.setGeometry(10, 100, 1000, 800)
    w.move(50,20)
    w.show()
    w.setWindowTitle('File Manager')
    a.exec_()
    del w
    del a


def do_main(**kwa):
    file_manager_lcls1(**kwa)


if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
    kwa = {\
      'loglevel':'DEBUG',\
    }

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if   tname == '0': do_main(**kwa)
    else: logger.debug('Not-implemented test "%s"' % tname)

# EOF
