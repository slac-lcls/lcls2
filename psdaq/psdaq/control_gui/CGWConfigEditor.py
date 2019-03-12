#--------------------
"""
:py:class:`CGWConfigEditor` - widget for configuration editor
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWConfigEditor import CGWConfigEditor

    # Methods - see test

See:
    - :py:class:`CGWConfigEditor`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-03-08 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

from psdaq.control_gui.CGJsonUtils import load_json_from_file, str_json, json_from_str
from psdaq.control_gui.Utils import save_textfile, path_to_test_data

from PyQt5.QtWidgets import QWidget,  QVBoxLayout, QTextEdit, QPushButton, QFileDialog # , QWidget, QLabel, QLineEdit, QGroupBox
#from PyQt5.QtCore import pyqtSignal #, Qt, QRectF, QPointF, QTimer

#--------------------

def fake_str_json() :
    return '{"detType": "test","detName": "test1","detId": "serial1234","doc": "No comment","alg": {"alg": "raw", "doc": "", "version": [1, 2, 3] },"aa": -5,"ab": -5192,"ac": -393995,"ad": -51000303030,"ae": 3,"af": 39485,"ak": "A random string!"}'

#--------------------

class CGWConfigEditor(QWidget) :
#class CGWConfigEditor(QGroupBox) :
    """
    """
    def __init__(self, parent=None, parent_ctrl=None):

        #QGroupBox.__init__(self, 'Partition', parent)
        QWidget.__init__(self, parent)
        self.parent_ctrl = parent_ctrl

        self.edi_txt = QTextEdit('Json in text is here...')
        self.but_save = QPushButton('Save')

        self.vbox = QVBoxLayout() 
        self.vbox.addWidget(self.edi_txt)
        self.vbox.addWidget(self.but_save)
        #self.vbox.addStretch(1)
        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        #self.but_select.clicked.connect(self.on_but_select)
        #self.but_display.clicked.connect(self.on_but_display)
        self.but_save.clicked.connect(self.on_but_save)

        # I/O files
        self.load_text('%s/json2xtc_test.json' % path_to_test_data())
        self.fname_json = './test.json' # output file

#--------------------

    def load_text(self, fname) :  
        #print('CGWConfigEditor: load json from %s' % fname)
        #jo = load_json_from_file(fname)

        print('CGWConfigEditor: use FAKE json')
        jo = json_from_str(fake_str_json())
        sj = str_json(jo)
        #print('CGWConfigEditor: str json:\n%s' % sj)
        #self.edi_txt.append(sj)
        self.edi_txt.setText(sj)

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

        self.but_save.setStyleSheet(style.styleButton) 
        self.but_save.setVisible(True)
 
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
 
    def on_but_save(self):
        logger.debug('on_but_save')
        if self.select_fname() : 
           self.save_json_in_file()

#--------------------
 
    def select_fname(self):

        logger.info('select_fname %s' % self.fname_json)
        path, ftype = QFileDialog.getSaveFileName(self,
                                               caption   = 'Select the file to save json',
                                               directory = self.fname_json,
                                               filter    = '*.json'
                                               )
        if path == '' :
            logger.debug('Saving is cancelled')
            return False
        self.fname_json = path
        logger.info('Output file: %s' % path)
        return True

#--------------------
 
    def save_json_in_file(self):
        logger.info('save_json_in_file %s' % self.fname_json)
        text = str(self.edi_txt.toPlainText())
        save_textfile(text, self.fname_json, mode='w', verb=True)

#--------------------
 
#    def on_but_display(self):
#        logger.debug('on_but_display')

#--------------------

#    def closeEvent(self, e):
#        logger.debug('closeEvent')
#        #self.parent_ctrl.w_display = None

#--------------------

if __name__ == "__main__" :

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWConfigEditor(None)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
