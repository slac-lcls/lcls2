#------------------------------
"""
@version $Id: QWFileName.py 12962 2016-12-09 20:06:16Z dubrovin@SLAC.STANFORD.EDU $

@author Mikhail S. Dubrovin
"""
#------------------------------

import os
import sys
from graphqt.Frame import Frame
from PyQt4 import QtGui, QtCore

#------------------------------

class QWFileName(Frame) : # QtGui.QWidget
    """Widget for file name input
    """
    def __init__(self, parent=None, butname='Browse', label='File:',\
                 path='/reg/neh/home/dubrovin/LCLS/rel-expmon/log.txt',\
                 mode='r',\
                 fltr='*.txt *.data *.png *.gif *.jpg *.jpeg\n *',\
                 show_frame=False) :

        #QtGui.QWidget.__init__(self, parent)
        Frame.__init__(self, parent, mlw=1, vis=show_frame)
        self._name = self.__class__.__name__

        self.mode = mode
        self.path = path
        self.fltr = fltr
        self.show_frame = show_frame

        self.lab = QtGui.QLabel(label)
        self.but = QtGui.QPushButton(butname)
        self.edi = QtGui.QLineEdit(path)
        self.edi.setReadOnly(True) 

        self.hbox = QtGui.QHBoxLayout() 
        self.hbox.addWidget(self.lab)
        self.hbox.addWidget(self.edi)
        self.hbox.addWidget(self.but)
        self.hbox.addStretch(1)
        self.setLayout(self.hbox)

        self.set_tool_tips()
        self.set_style()

        self.connect(self.but, QtCore.SIGNAL('clicked()'), self.on_but)

#------------------------------

    def path(self):
        return self.path

#------------------------------

    def set_tool_tips(self) :
        self.but.setToolTip('Select input file.')
        self.edi.setToolTip('Path to the file (read-only).\nClick on button to change it.') 

#------------------------------

    def set_style(self) :
        self.setWindowTitle('File name selection widget')
        self.setMinimumWidth(300)
        self.edi.setMinimumWidth(210)
        self.setFixedHeight(50 if self.show_frame else 34)
        if not self.show_frame : self.setContentsMargins(-9,-9,-9,-9)

        #style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        #style = "background-color: rgb(100, 240, 200); color: rgb(0, 0, 0);" # Greenish
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 0);" # Pinkish
        #style = "background-color: rgb(240, 240, 100); color: rgb(0, 0, 0);" # YellowBkg
        #self.setStyleSheet(style)

        #self.setMinimumSize(725,360)
        #self.setFixedSize(750,270)
        #self.setMaximumWidth(800)
 
#------------------------------
 
    def on_but(self):
        self.path = str(QtGui.QFileDialog.getSaveFileName(self, 'Output file', self.path, filter=self.fltr)) \
                    if self.mode == 'w' else \
                    str(QtGui.QFileDialog.getOpenFileName(self, 'Input file', self.path, filter=self.fltr))

        dname, fname = os.path.split(self.path)

        if self.mode == 'r' and not os.path.lexists(self.path) :
            return
            #raise IOError('File %s is not available' % self.path)

        elif dname == '' or fname == '' :
            return
            #logger.info('Input directiry name or file name is empty... use default values', __name__)
            #print'Input directiry name or file name is empty... use default values'

        else :
            self.edi.setText(self.path)
            self.emit(QtCore.SIGNAL('path_is_changed(QString)'), self.path)
            #logger.info('Selected file:\n' + self.path, __name__)
            #print 'Selected file: %s' % self.path

#------------------------------

    def connect_path_is_changed_to_recipient(self, recip) :
        self.connect(self, QtCore.SIGNAL('path_is_changed(QString)'), recip)

#------------------------------
 
    def test_signal_reception(self, s) :
        print '%s.%s: str=%s' % (self._name, sys._getframe().f_code.co_name, s)

#------------------------------

if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    w = QWFileName(None, butname='Select', label='Path:',\
                   path='/reg/neh/home/dubrovin/LCLS/rel-expmon/log.txt', show_frame=True)
    w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#------------------------------
