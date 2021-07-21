
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
from PyQt5.QtCore import Qt, QPoint, QEvent, QMargins, QSize
from PyQt5.QtGui import QCursor


class QWPopupSelectItem(QDialog):

    def __init__(self, parent=None, lst=[]):

        QDialog.__init__(self, parent)

        self.name_sel = None
        self.list = QListWidget(parent)

        self.fill_list(lst)
        #self.fill_list_icons(lst_icons)

        # Confirmation buttons
        #self.but_cancel = QPushButton('&Cancel') 
        #self.but_apply  = QPushButton('&Apply') 
        #cp.setIcons()
        #self.but_cancel.setIcon(cp.icon_button_cancel)
        #self.but_apply .setIcon(cp.icon_button_ok)
        #self.connect(self.but_cancel, QtCore.SIGNAL('clicked()'), self.onCancel)
        #self.connect(self.but_apply,  QtCore.SIGNAL('clicked()'), self.onApply)

        #self.hbox = QVBoxLayout()
        #self.hbox.addWidget(self.but_cancel)
        #self.hbox.addWidget(self.but_apply)
        ##self.hbox.addStretch(1)

        vbox = QVBoxLayout()
        vbox.addWidget(self.list)
        #vbox.addLayout(self.hbox)
        #vbox.addStretch(1)
        self.setLayout(vbox)

        self.list.itemClicked.connect(self.on_item_click)

        self.show_tool_tips()
        self.set_style()


    def fill_list(self, lst):
        self.list.clear()
        for s in sorted(lst):
            item = QListWidgetItem(s, self.list)
            item.setSizeHint(QSize(4*len(s), 15))
        #self.list.sortItems(Qt.AscendingOrder)


#    def fill_list_icons(self, lst_icons):
#        for ind, icon in enumerate(lst_icons):
#            item = QListWidgetItem(icon, '%d'%ind, self.list) #'%d'%ind
#            item.setSizeHint(QtCore.QSize(80,30))
#        #self.list.sortItems(Qt.AscendingOrder)


    def set_style(self):
        self.setWindowTitle('Select')
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
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


#    def mousePressEvent(self, e):
#        QDialog.mousePressEvent(self, e)
#        logger.debug('mousePressEvent')

        
    def event(self, e):
        #logger.debug('event.type %s' % str(e.type()))
        if e.type() == QEvent.WindowDeactivate:
            logger.debug('intercepted mouse click outside popup window')
            self.reject()
            self.done(QDialog.Rejected)
        return QDialog.event(self, e)
    

    def closeEvent(self, e):
        logger.debug('closeEvent', __name__)
        self.reject()
        self.done(QDialog.Rejected)


    def selectedName(self):
        return self.name_sel

 
#    def onCancel(self):
#        logger.debug('onCancel', __name__)
#        self.reject()
#        self.done(QDialog.Rejected)
 
#    def onApply(self):
#        logger.debug('onApply', __name__)  
#        self.accept()
#        self.done(QDialog.Accepted)


def popup_select_item_from_list(parent, lst, min_height=200, dx=0, dy=0):
    w = QWPopupSelectItem(parent, lst)
    #w.setMinimumHeight(min_height)
    size = len(lst)
    nchars = max([len(s) for s in lst])
    height = min(min_height, size*16)
    w.setFixedWidth(10*nchars)
    w.setFixedHeight(height)
    if dx or dy: w.move(QCursor.pos().__add__(QPoint(dx,dy)))
    resp=w.exec_()
    #if   resp == QDialog.Accepted: return w.selectedName()
    #elif resp == QDialog.Rejected: return None    
    #else: return None
    return w.selectedName()

#----------- TESTS ------------

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
