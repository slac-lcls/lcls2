
"""Class :py:class:`CMWControlBase` is a QWidget base class for control buttons
===============================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/CMWControlBase.py

    from psana.graphqt.CMWControlBase import CMWControlBase
    w = CMWControlBase()

Created on 2021-06-16 by Mikhail Dubrovin
"""
import os
import sys

import logging
logger = logging.getLogger(__name__)

#from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,\
                            QPushButton, QLabel, QComboBox, QLineEdit, QTextEdit
from PyQt5.QtCore import QSize, QRectF, pyqtSignal, QModelIndex, QTimer

from psana.graphqt.QWFileNameV2 import QWFileNameV2
from psana.graphqt.CMConfigParameters import cp, dirs_to_search, expname_def, dir_calib
from psana.graphqt.Styles import style
from psana.graphqt.QWIcons import icon
import psana.graphqt.QWUtils as qwu

COMMAND_SET_ENV_LCLS1 = '. /cds/sw/ds/ana/conda1/manage/bin/psconda.sh; echo "PATH: $PATH"; echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"; '
ENV1 = {} #'PATH':'/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/puppetlabs/bin'}


class CMWControlBase(QWidget):
    """QWidget base class with a regular set of buttons for control panels"""

    def __init__(self, **kwa):

        QWidget.__init__(self, kwa.get('parent', None))

        self.wfnm = QWFileNameV2(parent = self,\
          label  = kwa.get('label', 'File:'),\
          path   = kwa.get('path', '/cds/group/psdm/detector/data2_test/misc/Select'),\
          fltr   = kwa.get('fltr', '*.txt *.npy *.data *.dat\n*'),\
          dirs   = kwa.get('dirs', dirs_to_search()))

        self.but_tabs = QPushButton('Tabs %s' % cp.char_expand)
        self.but_save = QPushButton('Save')
        self.but_view = QPushButton('View')

        self.but_tabs.clicked.connect(self.on_but_tabs)
        self.but_save.clicked.connect(self.on_but_save)
        self.but_view.clicked.connect(self.on_but_view)

        if __name__ == "__main__":
            self.box1 = QHBoxLayout()
            self.box1.addWidget(self.wfnm)
            self.box1.addStretch(1)
            self.box1.addWidget(self.but_tabs)
            self.box1.addWidget(self.but_save)
            self.box1.addWidget(self.but_view)
            self.setLayout(self.box1)

            self.wfnm.connect_path_is_changed_to_recipient(self.on_changed_fname)

            self.set_tool_tips()
            self.set_style()


    def set_tool_tips(self):
        self.wfnm.setToolTip('Select file')
        self.but_tabs.setToolTip('Show/hide tabs')
        self.but_save.setToolTip('Save button')
        self.but_view.setToolTip('Use the last selected item to view in IV')


    def set_style(self):
        icon.set_icons()
        self.but_save.setIcon(icon.icon_save)
        self.but_tabs.setStyleSheet(style.styleButtonGood)
        self.but_tabs.setFixedWidth(60)
        self.but_save.setFixedWidth(60)
        self.but_view.setFixedWidth(60)
        self.wfnm.lab.setStyleSheet(style.styleLabel)


    def on_changed_fname(self, fname):
        logger.debug('on_changed_fname: %s' % fname)


    def on_but_tabs(self):
        logger.debug('on_but_tabs switch between visible and invisible tabs')
        self.view_hide_tabs()


    def on_but_save(self):
        logger.debug('on_but_save - NEEDS TO BE RE_IMPLEMENTED')


    def on_but_view(self):
        logger.debug('on_but_view - NEEDS TO BE RE_IMPLEMENTED')


    def view_hide_tabs(self):
        wtabs = cp.cmwmaintabs
        if wtabs is None: return
        is_visible = wtabs.tab_bar_is_visible()
        self.but_tabs.setText('Tabs %s'%cp.char_shrink if is_visible else 'Tabs %s'%cp.char_expand)
        wtabs.set_tabs_visible(not is_visible)


    def but_tabs_is_visible(self, isvisible=True):
        self.but_tabs.setVisible(isvisible)


if __name__ == "__main__":
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d: %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    w = CMWControlBase()
    w.setGeometry(100, 50, 500, 50)
    w.setWindowTitle('CMWControlBase')
    w.show()
    app.exec_()
    del w
    del app

# EOF
