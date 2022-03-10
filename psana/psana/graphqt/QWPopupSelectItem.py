
"""
:py:class:`QWPopupSelectItem` - Popup GUI for (str) item selection from the list of items
=========================================================================================

Usage::

    # Import
    from psana.graphqt.QWPopupSelectItem import QWPopupSelectItem

    # Methods - see test
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    exp_name = popup_select_item_from_list(None, lst)

See:
    - :py:class:`QWPopupSelectItem`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2017-01-26 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-15
"""

import os
import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QApplication, QDialog, QListWidget, QVBoxLayout, QListWidgetItem# QPushButton
from PyQt5.QtCore import Qt, QPoint, QEvent, QMargins, QSize, QTimer
from PyQt5.QtGui import QCursor


class QWPopupSelectItem(QDialog):

    def __init__(self, parent=None, lst=[], show_frame=False):

        QDialog.__init__(self, parent, flags=Qt.WindowStaysOnTopHint)

        self.name_sel = None
        self.list = QListWidget(parent=self)
        self.show_frame = show_frame

        self.fill_list(lst)

        vbox = QVBoxLayout()
        vbox.addWidget(self.list)
        self.setLayout(vbox)

        self.list.itemClicked.connect(self.on_item_click)

        self.show_tool_tips()
        self.set_style()

        self.dt_msec=1000
        QTimer().singleShot(self.dt_msec, self.on_timeout)


    def on_timeout(self):
        logger.debug('on_timeout - activate popup window, isActive: %s' % self.isActiveWindow())
        self.setFocus(True)
        self.raise_()
        self.activateWindow()
        QTimer().singleShot(self.dt_msec, self.on_timeout)


    def fill_list(self, lst):
        self.list.clear()
        for s in sorted(lst):
            item = QListWidgetItem(s, self.list)
            item.setSizeHint(QSize(4*len(s), 15))
        #self.list.sortItems(Qt.AscendingOrder)


    def set_style(self):
        self.setWindowTitle('Select')
        if not self.show_frame:
          self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setEnabled(True)
        self.layout().setContentsMargins(0,0,0,0)
        parent = self.parentWidget()
        if parent is None:
           self.move(QCursor.pos().__add__(QPoint(-110,-50)))
        logger.debug('use %s position for popup findow' % ('CURSOR' if parent is None else 'BUTTON'))


    def show_tool_tips(self):
        self.setToolTip('Select item from the list')


    def on_item_click(self, item):
        self.name_sel = item.text()
        logger.debug('on_item_click %s' % self.name_sel)
        self.accept()
        self.done(QDialog.Accepted)


    def event(self, e):
        #logger.debug('event.type %s' % str(e.type()))
        if e.type() == QEvent.WindowDeactivate:
            logger.debug('intercepted mouse click outside popup window')
            self.reject()
            self.done(QDialog.Rejected)
        return QDialog.event(self, e)


    def closeEvent(self, e):
        logger.debug('closeEvent')
        self.reject()
        self.done(QDialog.Rejected)


    def selectedName(self):
        return self.name_sel


def popup_select_item_from_list(parent, lst, min_height=200, dx=0, dy=0, show_frame=False):
    w = QWPopupSelectItem(parent, lst, show_frame)
    #w.setMinimumHeight(min_height)
    size = len(lst)
    nchars = max([len(s) for s in lst])
    height = min(min_height, size*16)
    w.setFixedWidth(10*nchars)
    w.setFixedHeight(height)
    if dx or dy: w.move(QCursor.pos().__add__(QPoint(dx,dy)))
    resp=w.exec_()
    return w.selectedName()


if __name__ == "__main__":
  logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG)

  def test_select_exp(tname):
    lst = sorted(os.listdir('/reg/d/psdm/CXI/'))
    logger.debug('lst: %s' % str(lst))
    app = QApplication(sys.argv)
    exp_name = popup_select_item_from_list(None, lst)
    logger.debug('exp_name = %s' % exp_name)


if __name__ == "__main__":
    import sys; global sys
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.debug('%s\nTest %s' % (50*'_', tname))
    if   tname == '0': test_select_exp(tname)
    else: sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of Test %s' % tname)

# EOF
