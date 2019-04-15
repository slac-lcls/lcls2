#------------------------------
"""
:py:class:`QWPopupTableCheck` - Popup table of str items and/with check-boxes
==================================================================================

Usage::

    # Test: python lcls2/psdaq/psdaq/control_gui/QWPopupTableCheck.py

    # Import
    from psdaq.control_gui.QWPopupTableCheck import QWPopupTableCheck

    # Methods - see test

See:
    - :py:class:`QWPopupTableCheck`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui/>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-03-29 by Mikhail Dubrovin
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QPushButton, QSizePolicy#, QGridLayout, QCheckBox, QTextEdit, QLabel, 
from PyQt5.QtCore import Qt
from psdaq.control_gui.Styles import style
from psdaq.control_gui.QWTableOfCheckBoxes import QWTableOfCheckBoxes

#------------------------------

class QWPopupTableCheck(QDialog) :
    """
    """
    def __init__(self, **kwargs):
        parent = kwargs.get('parent', None)
        QDialog.__init__(self, parent)
 
        win_title = kwargs.get('win_title', None)
        if win_title is not None : self.setWindowTitle(win_title)

        self.wtab = QWTableOfCheckBoxes(**kwargs)
        #self.make_gui_checkbox()

        self.do_ctrl  = kwargs.get('do_ctrl', True)
        self.do_frame = kwargs.get('do_frame', True)

        self.but_cancel = QPushButton('&Cancel') 
        self.but_apply  = QPushButton('&Apply') 
        
        self.but_cancel.clicked.connect(self.onCancel)
        self.but_apply.clicked.connect(self.onApply)

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.but_cancel)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_apply)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.wtab)
        self.vbox.addLayout(self.hbox)
        self.setLayout(self.vbox)

        self.setIcons()
        self.set_style()

#-----------------------------  

    def set_style(self):
        if not self.do_frame :
           self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        styleGray = "background-color: rgb(230, 240, 230); color: rgb(0, 0, 0);" # Gray
        #styleTest = "background-color: rgb(100, 240, 200); color: rgb(0, 0, 0);"
        styleDefault = ""
        self.setStyleSheet(styleDefault)

        self.layout().setContentsMargins(0,0,0,0)

        self.setMinimumWidth(100)
        self.but_cancel.setFixedWidth(70)
        self.but_apply .setFixedWidth(70)

        self.but_cancel.setFocusPolicy(Qt.NoFocus)
        self.but_cancel.setStyleSheet(styleGray)
        self.but_apply.setStyleSheet(styleGray)
        self.but_apply.setEnabled(self.do_ctrl)
        self.but_apply.setFlat(not self.do_ctrl)

        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        self.wtab.setFixedHeight(self.wtab.height()+2)
        self.setFixedWidth(self.wtab.width()+2)
        

    def setIcons(self):
        try :
          from psdaq.control_gui.QWIcons import icon
          icon.set_icons()
          self.but_cancel.setIcon(icon.icon_button_cancel)
          self.but_apply .setIcon(icon.icon_button_ok)
        except : pass
 

    def onCancel(self):
        logger.debug('onCancel')
        self.reject()


    def onApply(self):
        logger.debug('onApply')  
        self.list2d_out = self.wtab.fill_output_object()
        self.accept()


    def table_out(self):
        return self.list2d_out

#------------------------------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)

    title_h = ['proc/pid/host', 'alias']
    tableio = [\
      [[False,'name 1'], 'alias 1'],\
      [[True, 'name 2'], 'alias 2'],\
      [[True, 'name 3'], 'alias 3'],\
      ['name 4', [True, 'alias 4']],\
      ['name 5',         'alias 5'],\
    ]

    print('%s\nI/O  table:' % (50*'_'))
    for rec in tableio : print(rec)

    w = QWPopupTableCheck(tableio=tableio, title_h=title_h, do_ctrl=True, do_edit=True)
    w.move(200,100)
    resp = w.exec_()
    logger.debug('resp: %s' % {QDialog.Rejected:'Rejected', QDialog.Accepted:'Accepted'}[resp])
    #for name,state in dict_in.items() : logger.debug('%s checkbox state %s' % (name.ljust(10), state))

    print('%s\nI/O table:' % (50*'_'))
    for rec in tableio : print(rec)

    print('%s\nOutput table from items:' % (50*'_'))
    for rec in w.table_out() : print(rec)

    del w
    del app

#------------------------------
