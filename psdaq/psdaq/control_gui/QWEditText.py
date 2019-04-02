#------------------------------
"""
:py:class:`QWEditText` - simple text editor widget
==================================================

Usage::

    # Test: python lcls2/psdaq/psdaq/control_gui/QWEditText.py

    # Import
    from psdaq.control_gui.QWEditText import QWEditText

    # Methods - see test

See:
    - :py:class:`QWEditText`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui/>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-03-20 by Mikhail Dubrovin
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

import os

#from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QFileDialog, QHBoxLayout, QVBoxLayout,\
                            QPushButton, QTextEdit, QComboBox
                            # QSizePolicy, QCheckBox, QFrame
from psdaq.control_gui.Utils import load_textfile, save_textfile
#from psana.pyalgos.generic.Utils import load_textfile, save_textfile

#------------------------------

class QWEditText(QWidget) :
    """ Text editor widget woth Load and Save buttons.
    """
    MORE_OPTIONS = ('More','Load','Save')

    def __init__(self, **kwargs) :
        parent      = kwargs.get('parent', None)
        win_title   = kwargs.get('win_title', 'Text Editor')
        self.ifname = kwargs.get('ifname', 'test.txt')
        self.ofname = kwargs.get('ofname', 'test.txt')
        self.textin = kwargs.get('text', 'N/A')

        QWidget.__init__(self, parent)
        if win_title is not None : self.setWindowTitle(win_title)

        self.vbox = QVBoxLayout()
        self.edi_text = QTextEdit(str(self.textin))
        self.vbox.addWidget(self.edi_text)

        self.box_more = QComboBox(self)
        self.box_more.addItems(self.MORE_OPTIONS)
        self.box_more.setCurrentIndex(0)

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.box_more)
        self.hbox.addStretch(1)

        self.vbox.addLayout(self.hbox)
        self.setLayout(self.vbox)

        self.box_more.currentIndexChanged[int].connect(self.on_box_more)

        self.set_style()
        #self.set_icons()
        self.set_tool_tips()

#-----------------------------  

    def set_tool_tips(self):
        self.setToolTip('Text editor')
        self.box_more.setToolTip('More options:'\
                                +'\n * Load content from file'\
                                +'\n * Save content in file')

    def set_style(self):
        #self.setFixedWidth(200)
        self.setMinimumWidth(200)
        styleGray = "background-color: rgb(230, 240, 230); color: rgb(0, 0, 0);" # Gray
        styleDefault = ""

        #self.setWindowFlags(Qt.FramelessWindowHint)
        self.layout().setContentsMargins(0,0,0,0)

        self.setStyleSheet(styleDefault)
        self.set_style_msg(styleGray)

 
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
          from psdaq.control_gui.QWIcons import icon
          #from psana.graphqt.QWIcons import icon
          icon.set_icons()
          #self.but_cancel.setIcon(icon.icon_button_cancel)
          #self.but_apply .setIcon(icon.icon_button_ok)
        except : pass
 
    #def resizeEvent(self, e):
        #logger.debug('resizeEvent') 

    #def moveEvent(self, e):
        #logger.debug('moveEvent') 

    #def event(self, event):
        #logger.debug('Event happens...: %s' % str(event))

    #def closeEvent(self, event):
    #    logger.debug('closeEvent')

#--------------------

    def get_content(self):
        return str(self.edi_text.toPlainText())

    def set_content(self, text='N/A'):
        self.edi_text.setPlainText(text)

#--------------------

    def select_ifname(self):
        logger.info('select_ifname %s' % self.ifname)
        path, ftype = QFileDialog.getOpenFileName(self,
                        caption   = 'Select the file to load json',
                        directory = self.ifname,
                        filter = 'Text files(*.txt *.text *.dat *.data *.npy)\nAll files (*)'
                        )
        if path == '' :
            logger.info('Loading is cancelled')
            return False
        self.ifname = path
        logger.info('Input file: %s' % path)
        return True

#--------------------
 
    def load_content_from_numpy_file(self, fname):
        logger.debug('TBD: load_content_from_numpy_file %s' % fname)
        import numpy as np
        nda = np.load(fname)              # array([0, 1, 2, 3, 4, 5])
        #lst = nda.tolist()               # [0, 1, 2, 3, 4, 5]
        #lst_str = [str(v)] for v in lst] # ['0', '1', '2', '3', '4', '5']
        #txt = ' '.join(lst)              # '0 1 2 3 4 5'
        txt = np.array2string(nda).lstrip('[').rstrip(']') # '0 1 2 3 4 5'
        return txt

#--------------------
 
    def save_content_in_numpy_file(self, fname):
        logger.warning('save_content_in_numpy_file %s' % fname)
        txt = self.get_content()
        import numpy as np

        #TODO: enter dtype and shape using popup gui, otherwise default

        arr = np.fromstring(txt, sep=' ') # dtype=int
        #arr.astype(int)                   
        #arr.shape = shape
        np.save(fname, arr)

#--------------------
 
    def on_but_load(self):
        logger.debug('on_but_load')
        if self.select_ifname() :
           root, ext = os.path.splitext(self.ifname)
           self.textin = self.load_content_from_numpy_file(self.ifname)\
                         if ext == '.npy' else load_textfile(self.ifname)

           self.set_content(self.textin)

#--------------------
 
    def save_text_in_file(self):
        #logger.info('save_text_in_file %s' % self.ofname)
        txt = self.get_content()
        save_textfile(txt, self.ofname, mode='w', verb=False)

#--------------------
 
    def select_ofname(self):
        logger.info('select_ofname %s' % self.ofname)
        path, ftype = QFileDialog.getSaveFileName(self,
                        caption   = 'Select the file to save',
                        directory = self.ofname,
                        filter    = 'Text files (*.txt *.text *.dat *.data *.npy)\nAll files (*)'
                        )
        if path == '' :
            logger.info('Saving is cancelled')
            return False
        self.ofname = path
        logger.info('Output file: %s' % path)
        return True

#--------------------
 
    def on_but_save(self):
        logger.debug('on_but_save')
        if self.select_ofname() : 
           root, ext = os.path.splitext(self.ofname)
           if ext == '.npy' : self.save_content_in_numpy_file(self.ofname)
           else             : self.save_text_in_file()

#--------------------
 
    def on_box_more(self, ind) :
        opt = self.MORE_OPTIONS[ind]
        logger.info('CGWConfigEditor selected option %s' % opt)
        if   ind==1 : self.on_but_load()
        elif ind==2 : self.on_but_save()
        self.box_more.setCurrentIndex(0)

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
    w = QWEditText(text='1 2 3 4 5 11 12 30 40')
    w.move(QPoint(100,50))
    w.show()
    app.exec_()
    t = w.get_content()
    logger.debug("edited text: %s" % str(t))
    del w
    del app

#------------------------------
