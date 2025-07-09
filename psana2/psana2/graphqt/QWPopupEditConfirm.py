
"""
:py:class:`QWPopupEditConfirm` - Popup GUI
============================================

Usage::

    # Test: python lcls2/psana/psana/graphqt/QWPopupEditConfirm.py

    # Import
    from psana2.graphqt.QWPopupEditConfirm import popup_edit_and_confirm
    r, s = popup_edit_and_confirm(parent, dx=0, dy=0, height=60, width=150, is_frameless=False,\
                                  msg='text to edit',\
                                  win_title='Edit & confirm or cancel')

    # Methods - see test

See:
    - :py:class:`QWPopupEditConfirm`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2021-09-08 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QPushButton, QTextEdit, QFrame, QSizePolicy, QApplication
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QCursor  # , QColor, QBrush


class QWPopupEditConfirm(QDialog):

    def __init__(self, **kwa):
        QDialog.__init__(self, kwa.get('parent', None))
        win_title = kwa.get('win_title', 'Edit and confirm or cancel')
        if win_title: self.setWindowTitle(win_title)
        self.edi_msg = QTextEdit(kwa.get('msg', 'text to edit and confirm'))
        self.but_apply = QPushButton(kwa.get('but_title_apply', 'Apply'))
        self.but_cancel = QPushButton(kwa.get('but_title_cancel', 'Cancel'))
        self.but_cancel.clicked.connect(self.on_cancel)
        self.but_apply.clicked.connect(self.on_apply)

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.but_cancel)
        self.hbox.addWidget(self.but_apply)
        self.hbox.addStretch(1)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.edi_msg)
        self.vbox.addLayout(self.hbox)
        self.setLayout(self.vbox)

        self.but_cancel.setFocusPolicy(Qt.NoFocus)
        self.set_style()

    def set_style(self):
        self.layout().setContentsMargins(2,2,2,2)
        styleGray = "background-color: rgb(230, 240, 230); color: rgb(0, 0, 0);"
        self.but_cancel.setStyleSheet(styleGray)
        self.but_apply.setStyleSheet(styleGray)
        #self.edi_msg.setMinimumSize(500,60)
        #self.setFixedHeight(30)
        #self.setMinimumHeight(30)

    def on_cancel(self):
        logger.debug('on_cancel')
        self.reject()

    def on_apply(self):
        logger.debug('on_apply')
        self.accept()

    def message(self):
        return str(self.edi_msg.toPlainText())

def popup_edit_and_confirm(parent, dx=0, dy=0, height=60, width=150, is_frameless=False,\
                           msg='text to edit',\
                           win_title='Edit & confirm or cancel'):
    w = QWPopupEditConfirm(parent=parent, msg=msg, win_title=win_title)
    if width  is not None: w.setFixedWidth(width)
    if height is not None: w.setFixedHeight(height)
    if is_frameless: w.setWindowFlags(w.windowFlags() | Qt.FramelessWindowHint)
    w.move(QCursor.pos().__add__(QPoint(dx,dy)))
    resp = w.exec_()
    s = w.message()
    del w
    return resp, s

if __name__ == "__main__":
    import os
    import sys
    logging.getLogger('psana.pscalib.geometry').setLevel(logging.WARNING)
    logging.basicConfig(format='[%(levelname).1s] L:%(lineno)03d %(name)s %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    parent = None
    r, s = popup_edit_and_confirm(parent, dx=0, dy=0, height=60, width=150, is_frameless=False,\
                                  msg='my text', win_title='Edit & confirm or cancel')

    logger.debug('QtWidgets.QDialog.Rejected: %d' % QDialog.Rejected)
    logger.debug('QtWidgets.QDialog.Accepted: %d' % QDialog.Accepted)
    logger.debug('resp=%s output: %s' % (r,s))

    #del w
    del app

# EOF
