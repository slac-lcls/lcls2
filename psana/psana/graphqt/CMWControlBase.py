
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

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QGridLayout, QPushButton, QLabel, QComboBox, QLineEdit
from PyQt5.QtCore import QSize, QRectF, pyqtSignal, QModelIndex

from psana.graphqt.CMConfigParameters import cp, dirs_to_search
from psana.graphqt.Styles import style
from psana.graphqt.QWIcons import icon


class CMWControlBase(QWidget):
    """QWidget - base class for control buttons"""

    def __init__(self, **kwargs):

        parent = kwargs.get('parent',None)
        QWidget.__init__(self, parent)

        self.but_tabs = QPushButton('Tabs %s' % cp.char_expand)
        self.but_save = QPushButton('Save')
        self.but_view = QPushButton('View')

        self.but_tabs.clicked.connect(self.on_but_tabs)
        self.but_save.clicked.connect(self.on_but_save)
        self.but_view.clicked.connect(self.on_but_view)

        if __name__ == "__main__":
            self.box1 = QHBoxLayout()
            self.box1.addStretch(1) 
            self.box1.addWidget(self.but_tabs)
            self.box1.addWidget(self.but_save)
            self.box1.addWidget(self.but_view)
            self.setLayout(self.box1)

            self.set_tool_tips()
            self.set_style()


    def set_tool_tips(self):
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
