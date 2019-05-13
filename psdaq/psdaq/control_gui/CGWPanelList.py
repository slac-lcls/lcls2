#--------------------
"""Class :py:class:'CGWPanelList` is a QWList->QListView->QWidget for list model
================================================================================

Usage ::

    # Run test: python lcls2/psdaq/psdaq/control_gui/CGWPanelList.py

    from psdaq.control_gui.CGWPanelList import CGWPanelList
    w = CGWPanelList()

Created on 2019-05-01
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

from psdaq.control_gui.QWList import QWList, QStandardItem
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QSize

#--------------------

BIT_CHECKABLE  = 1
BIT_ENABLED    = 2
BIT_EDITABLE   = 4
BIT_SELECTABLE = 8
BIT_CBX_STATUS = 16

#--------------------

class CGWPanelList(QWList) :
    """Widget for List
    """
    def __init__(self, **kwargs) :
        QWList.__init__(self, **kwargs)
        self.disconnect_signals()
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.setMinimumSize(50, 30)

#--------------------

    def sizeHint(self):
        return QSize(50, 30)

#--------------------

    def fill_list_model(self, **kwargs):
        self.clear_model()

        lst   = kwargs.get('list_str', [])
        flags = kwargs.get('list_flags', [2 for i in range(len(lst))])

        for s,f in zip(lst,flags):
            item = QStandardItem(s)
            if f & BIT_CHECKABLE : item.setCheckState({False:0, True:2}[bool(f & BIT_CBX_STATUS)])
            item.setCheckable (f & BIT_CHECKABLE)
            item.setEnabled   (f & BIT_ENABLED)
            item.setEditable  (f & BIT_EDITABLE)
            item.setSelectable(f & BIT_SELECTABLE)
            self.model.appendRow(item)

#--------------------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    lst   = ['drp%d'%i for i in range(5)]
    flags = [(3+16*(i%2)) for i in range(5)]
    app = QApplication(sys.argv)
    #w = CGWPanelList(list_str=lst, list_flags=flags)
    w = CGWPanelList(list_str=lst)
    #w.setFixedSize(80, 300)
    w.setWindowTitle('CGWPanelList')
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

#--------------------
