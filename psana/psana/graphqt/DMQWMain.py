
"""Class :py:class:`DMQWMain` is a QWidget for File Manager of LCLS1
========================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/DMQWMain.py

    from psana.graphqt.DMQWMain import DMQWMain

See method:

Created on 2021-08-10 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit
from PyQt5.QtCore import Qt
from psana.graphqt.CMConfigParameters import cp, dir_calib
from psana.graphqt.DMQWList import DMQWList, EXPNAME_TEST
from psana.graphqt.DMQWControl import DMQWControl

class DMQWMain(QWidget):

    def __init__(self, **kwargs):

        parent = kwargs.get('parent', None)
 
        QWidget.__init__(self, parent=parent)

        cp.dmqwmain = self

        self.proc_kwargs(**kwargs)

        #fname = kwargs.get('fname', None)

        self.wdmlist = DMQWList(parent=self, **kwargs)

        self.wctrl = DMQWControl(parent=self)

        self.wrhs = QTextEdit('wrhs')

        self.hspl = QSplitter(Qt.Horizontal)
        self.hspl.addWidget(self.wdmlist)
        self.hspl.addWidget(self.wrhs)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.wctrl)
        self.vbox.addWidget(self.hspl)
        self.setLayout(self.vbox)

        self.set_style()
        self.set_tool_tips()

        #if fname is not None: self.wctrl.on_changed_fname_nda(fname)

#        self.connect_signals_to_slots()
#    def connect_signals_to_slots(self):
#        self.connect(self.wbut.but_reset, QtCore.SIGNAL('clicked()'), self.on_but_reset)
#        self.connect(self.wbut.but_save,  QtCore.SIGNAL('clicked()'), self.on_but_save)


    def proc_kwargs(self, **kwargs):
        #print_kwargs(kwargs)
        loglevel   = kwargs.get('loglevel', 'DEBUG').upper()
        logdir     = kwargs.get('logdir', './')
        savelog    = kwargs.get('savelog', False)


    def set_tool_tips(self):
        self.setToolTip('File Manager for LCLS1')


    def set_style(self):
        self.layout().setContentsMargins(0,0,0,0)
        #self.wctrl.setFixedHeight(80)
        #self.wspec.setMaximumWidth(300)
        self.wctrl.setFixedHeight(40)


    def closeEvent(self, e):
        logger.debug('closeEvent')
        QWidget.closeEvent(self, e)
        cp.dmqwmain = None


def data_manager(**kwargs):
    loglevel = kwargs.get('loglevel', 'DEBUG').upper()
    intlevel = logging._nameToLevel[loglevel]
    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d: %(message)s', level=intlevel)

    a = QApplication(sys.argv)
    w = DMQWMain(**kwargs)
    w.setGeometry(10, 100, 1000, 800)
    w.move(50,20)
    w.show()
    w.setWindowTitle('Data Manager')
    a.exec_()
    del w
    del a


def do_main(**kwargs):
    data_manager(**kwargs)


if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
    kwargs = {\
      'loglevel':'DEBUG',\
      'expname':EXPNAME_TEST,\
    }

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if   tname == '0': do_main(**kwargs)
    else: logger.debug('Not-implemented test "%s"' % tname)

# EOF
