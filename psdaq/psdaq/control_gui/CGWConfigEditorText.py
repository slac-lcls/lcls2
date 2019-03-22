#--------------------
"""
:py:class:`CGWConfigEditorText` - widget for configuration editor
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWConfigEditorText import CGWConfigEditorText

    # Methods - see test

See:
    - :py:class:`CGWConfigEditorText`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-03-08 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

from psdaq.control_gui.CGJsonUtils import str_json, json_from_str

from PyQt5.QtWidgets import QWidget,  QVBoxLayout, QTextEdit #, QPushButton, QFileDialog, QLabel, QLineEdit, QGroupBox
#from PyQt5.QtCore import pyqtSignal #, Qt, QRectF, QPointF, QTimer

#--------------------

class CGWConfigEditorText(QWidget) :
#class CGWConfigEditorText(QGroupBox) :
    """Text-like configuration editor widget
    """
    def __init__(self, parent=None, parent_ctrl=None, dictj={'a_test':0,'b_test':1}) :

        self.dictj = dictj # is used in fill_tree_model which is called at superclass initialization

        #QGroupBox.__init__(self, 'Partition', parent)
        QWidget.__init__(self, parent)
        self.parent_ctrl = parent_ctrl

        self.edi_txt = QTextEdit('Json in text is here...')
        #self.but_save = QPushButton('Save')

        self.vbox = QVBoxLayout() 
        self.vbox.addWidget(self.edi_txt)
        #self.vbox.addWidget(self.but_save)
        #self.vbox.addStretch(1)
        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        #self.but_select.clicked.connect(self.on_but_select)
        #self.but_display.clicked.connect(self.on_but_display)
        #self.but_save.clicked.connect(self.on_but_save)

        self.set_content(self.dictj)

#--------------------

    def set_content(self, dictj) :  
        """Interface method
        """
        self.dictj = dictj
        sj = str_json(dictj)
        #self.edi_txt.append(sj)
        self.edi_txt.setText(sj)

#--------------------
 
    def get_content(self):
        """Interface method
        """
        return json_from_str(str(self.edi_txt.toPlainText()))

#--------------------

    def set_tool_tips(self) :
        self.setToolTip('Configuration editor GUI')

#--------------------

    def set_style(self) :

        from psdaq.control_gui.Styles import style
        #self.grb_read_nodes.setStyleSheet(style.qgrbox_title)
        #self.grb_proc_nodes.setStyleSheet(style.qgrbox_title)
        #self.grb_bld       .setStyleSheet(style.qgrbox_title)
        #self.grb_camera_ioc.setStyleSheet(style.qgrbox_title)

        self.setWindowTitle('Configuration Editor')
        self.setMinimumSize(300,500)
        self.layout().setContentsMargins(0,0,0,0)

        #self.but_save.setStyleSheet(style.styleButton) 
        #self.but_save.setVisible(True)
 
        #self.setMinimumWidth(300)
        #self.edi.setMinimumWidth(210)
        #self.setFixedHeight(34) # 50 if self.show_frame else 34)
        #if not self.show_frame : 
        #self.layout().setContentsMargins(0,0,0,0)

        #style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        #style = "background-color: rgb(100, 240, 200); color: rgb(0, 0, 0);" # Greenish
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 0);" # Pinkish
        #style = "background-color: rgb(240, 240, 100); color: rgb(0, 0, 0);" # YellowBkg
        #self.setStyleSheet(style)

        #self.setFixedSize(750,270)
        #self.setMaximumWidth(800)
 
#--------------------
 
#    def on_but_display(self):
#        logger.debug('on_but_display')

#--------------------

#    def closeEvent(self, e):
#        logger.debug('closeEvent')
#        #self.parent_ctrl.w_display = None

#--------------------

if __name__ == "__main__" :
    dictj_test = {"detType": "test","detName": "test1","detId": "serial1234","doc": "No comment","alg": {"alg": "raw", "doc": "", "version": [1, 2, 3] },"aa": -5,"ab": -5192,"ac": -393995,"ad": -51000303030,"ae": 3,"af": 39485,"ak": "A random string!"}

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWConfigEditorText(parent=None, parent_ctrl=None, dictj=dictj_test)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
