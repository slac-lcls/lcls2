#------------------------------
"""
:py:class:`QWPopupEditText` - Popup GUI
========================================

Usage::

    # Test: python lcls2/psdaq/psdaq/control_gui/QWPopupEditText.py

    # Import
    from psdaq.control_gui.QWPopupEditText import QWPopupEditText

    # Methods - see test

See:
    - :py:class:`QWPopupEditText`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui/>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-03-18 by Mikhail Dubrovin
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)
from PyQt5.QtWidgets import QDialog, QFileDialog, QHBoxLayout, QVBoxLayout,\
                            QPushButton, QCheckBox, QTextEdit, QFrame, QSizePolicy
from PyQt5.QtCore import Qt

from psdaq.control_gui.QWEditText import QWEditText

#from psdaq.control_gui.Utils import load_textfile, save_textfile
#from psana.pyalgos.generic.Utils import load_textfile, save_textfile

#------------------------------

class QWPopupEditText(QDialog) :
    """Popap text editor QWEditText with extra Apply and Cancel buttons.
    """
    def __init__(self, **kwargs) :

        parent      = kwargs.get('parent', None)
        win_title   = kwargs.get('win_title', 'Text Editor')

        QDialog.__init__(self, parent)
 
        if win_title is not None : self.setWindowTitle(win_title)

        self.edi_text = QWEditText(**kwargs)

        self.but_cancel = QPushButton('&Cancel') 
        self.but_apply  = QPushButton('&Apply') 
        
        self.edi_text.hbox.addWidget(self.but_cancel)
        self.edi_text.hbox.addWidget(self.but_apply)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.edi_text)
        self.setLayout(self.vbox)

        #self.but_cancel.setFocusPolicy(Qt.NoFocus)
        #self.but_apply.setFocusPolicy(Qt.StrongFocus)

        self.but_cancel.clicked.connect(self.on_but_cancel)
        self.but_apply.clicked.connect(self.on_but_apply)

        self.set_style()
        self.set_icons()
        self.set_tool_tips()

#-----------------------------  

    def set_tool_tips(self):
        self.setToolTip('Text Editor')
        self.but_apply.setToolTip('Apply changes')
        self.but_cancel.setToolTip('Cancel changes')
        #self.but_save.setToolTip('Save content in file')
        #self.but_load.setToolTip('Load content from file')
        

    def set_style(self):
        #self.setFixedWidth(200)
        self.setMinimumWidth(200)
        styleGray = "background-color: rgb(230, 240, 230); color: rgb(0, 0, 0);" # Gray
        styleDefault = ""
        #self.setWindowFlags(Qt.FramelessWindowHint)
        self.layout().setContentsMargins(0,0,0,0)

        self.setStyleSheet(styleDefault)
        self.but_cancel.setStyleSheet(styleGray)
        self.but_apply.setStyleSheet(styleGray)
        self.set_style_msg(styleGray)

        #self.but_apply.setEnabled(self.enblctrl)
        #self.but_apply.setFlat(not self.enblctrl)


    def set_style_msg(self, style_bkgd):
        #self.edi_msg.setReadOnly(True)
        #self.edi_msg.setStyleSheet(style_bkgd)
        #self.edi_msg.setFrameStyle(QFrame.NoFrame)
        #self.edi_msg.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding) # Ignored, Fixed, Expanding, Preferred
        #self.edi_msg.updateGeometry()
        #s = self.edi_msg.document().size()
        #print('XXX:document().size()', s.width(), s.height())
        #self.edi_msg.setMinimumSize(200,50)
        #self.edi_msg.setFixedSize(180,50)
        pass


    def set_icons(self):
        try :
          #from psana.graphqt.QWIcons import icon
          from psdaq.control_gui.QWIcons import icon
          icon.set_icons()
          self.but_cancel.setIcon(icon.icon_button_cancel)
          self.but_apply .setIcon(icon.icon_button_ok)
        except : pass
 
    #def resizeEvent(self, e):
        #logger.debug('resizeEvent') 

    #def moveEvent(self, e):
        #logger.debug('moveEvent') 

    #def event(self, event):
        #logger.debug('Event happens...: %s' % str(event))

    
    def closeEvent(self, e):
        logger.debug('closeEvent')
        QDialog.closeEvent(self, e)
        self.on_but_cancel()


    def on_but_cancel(self):
        logger.debug('on_but_cancel')
        self.reject()


    def on_but_apply(self):
        logger.debug('on_but_apply')  
        self.accept()

#--------------------

    def get_content(self):
        return str(self.edi_text.get_content())
        #return str(self.edi_text.toPlainText())

    def set_content(self, text='N/A'):
        self.edi_text.set_content(text)
        #self.edi_text.setPlainText(text)

#--------------------
#--------------------
#--------------------
#--------------------

if __name__ == "__main__" :
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    import sys
    from PyQt5.QtCore import QPoint #Qt
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = QWPopupEditText(parent=None, text="1,2,3,4,5,11,12,30,40")
    w.move(QPoint(100,50))
    #w.show()
    resp=w.exec_()
    t = w.get_content()
    logger.debug('resp: %s' % {QDialog.Rejected:'Rejected', QDialog.Accepted:'Accepted'}[resp])
    print("edited text: %s" % str(t))
    del w
    del app

#------------------------------
