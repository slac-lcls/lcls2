#--------------------
"""
:py:class:`CGWMainCollection` - widget for control_gui
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWMainCollection import CGWMainCollection

    # Methods - see test

See:
    - :py:class:`CGWMainCollection`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-04-26 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QGridLayout, QLabel, QSizePolicy 
# QWidget, QLabel, QLineEdit, QFileDialog
#from PyQt5.QtCore import pyqtSignal #, Qt, QRectF, QPointF, QTimer
from PyQt5.QtCore import QSize
from psdaq.control_gui.CGWPanelList import CGWPanelList
from psdaq.control_gui.CGJsonUtils  import get_status

#--------------------

class CGWMainCollection(QWidget) :
    """
    """
    TABTITLE_H = ['drp','teb','meb']

    def __init__(self, parent=None):

        QWidget.__init__(self, parent)

        self.grid = None
        self.update_table()

        #self.lab_duration = QLabel('Duration')
        #self.lab_events   = QLabel('Events')
        #self.lab_dameged  = QLabel('Damaged')
        #self.lab_size     = QLabel('Size')
        #self.but_status = QPushButton('Update')

        self.vbox = QVBoxLayout() 
        #self.vbox.addWidget(self.lab_duration)
        #self.vbox.addWidget(self.lab_events  )
        #self.vbox.addWidget(self.lab_dameged )
        #self.vbox.addWidget(self.lab_size    )
        self.vbox.addLayout(self.grid)
        #self.vbox.addWidget(self.but_status)
        #self.vbox.addStretch(1)
        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        #self.but_status.clicked.connect(self.on_but_status)

#--------------------

    def update_table(self) :
        #list2d = [['drp1','teb1','meb1'],\
        #          ['drp2','teb2','meb2'],\
        #          ['drp3','teb3','meb3']]

        list2d = get_status(self.TABTITLE_H)
        logger.debug('list2d processes status:\n%s' % str(list2d))

        nrows = len(list2d)

        if self.grid is None :
            self.grid = QGridLayout()

            for col,s in enumerate(self.TABTITLE_H) :
                self.grid.addWidget(QLabel(s), 0, col)
                self.grid.setColumnStretch(col, 0)
                self.grid.setColumnMinimumWidth(col, 100)
                lst_col = [list2d[r][col] for r in range(nrows)]
                self.grid.addWidget(CGWPanelList(list_str=lst_col), 1, col)

        else :
            for col,s in enumerate(self.TABTITLE_H) :
                w = self.grid.itemAtPosition(1, col).widget()
                lst_col = [list2d[r][col] for r in range(nrows)]
                w.fill_list_model(list_str=lst_col)

#--------------------

    def set_tool_tips(self) :
        self.setToolTip('CGWMainCollection')
        #self.but_status.setToolTip('Click on button.') 

#--------------------

    def sizeHint(self):
        return QSize(250, 60)

#--------------------

    def set_style(self) :
        from psdaq.control_gui.Styles import style
        self.setStyleSheet(style.qgrbox_title)

        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        self.setMinimumSize(100, 40)

        self.layout().setContentsMargins(0,0,0,0)

#--------------------
 
#    def on_but_status(self):
#        logger.debug('on_but_status')
#        self.update_table()

#--------------------

if __name__ == "__main__" :

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWMainCollection(None)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
