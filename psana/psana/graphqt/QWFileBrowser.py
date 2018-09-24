#------------------------------
#   QWFileBrowser ...
#------------------------------

import os

import logging
logger = logging.getLogger(__name__)

#from PyQt4 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QTextEdit, QComboBox, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt #, QMargins

from psana.graphqt.Styles import style

#from ConfigParametersForApp import cp
#from Logger                 import logger
import psana.graphqt.QWUtils as gu

from psana.pyalgos.generic.Utils import load_textfile

#------------------------------

class QWFileBrowser(QWidget) :
    """GUI for File Browser"""

    def __init__(self, parent=None, list_of_files=['Empty list'], selected_file=None, is_editable=True) :

        QWidget.__init__(self, parent)

        self.setGeometry(200, 400, 900, 500)
        self.setWindowTitle('GUI File Browser')
        #try : self.setWindowIcon(cp.icon_browser)
        #except : pass

        self.box_txt    = QTextEdit()
 
        self.tit_status = QLabel('Status:')
        self.tit_file   = QLabel('File:')
        self.but_brow   = QPushButton('Browse') 
        self.but_close  = QPushButton('Close') 
        self.but_save   = QPushButton('Save As') 

        self.is_editable = is_editable

        self.box_file      = QComboBox(self) 
        self.setListOfFiles(list_of_files)

        self.hboxM = QHBoxLayout()
        self.hboxM.addWidget(self.box_txt)

        self.hboxF = QHBoxLayout()
        self.hboxF.addWidget(self.tit_file)
        self.hboxF.addWidget(self.box_file)
        self.hboxF.addWidget(self.but_brow)

        self.hboxB = QHBoxLayout()
        self.hboxB.addWidget(self.tit_status)
        self.hboxB.addStretch(4)     
        self.hboxB.addWidget(self.but_save)
        self.hboxB.addWidget(self.but_close)

        self.vbox  = QVBoxLayout()
        #self.vbox.addWidget(self.tit_title)
        self.vbox.addLayout(self.hboxF)
        self.vbox.addLayout(self.hboxM)
        self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)
        
        #self.connect(self.but_brow,  QtCore.SIGNAL('clicked()'), self.onBrow)
        #self.connect(self.but_save,  QtCore.SIGNAL('clicked()'), self.onSave)
        #self.connect(self.but_close, QtCore.SIGNAL('clicked()'), self.onClose)
        #self.connect(self.box_file, QtCore.SIGNAL('currentIndexChanged(int)'), self.onBox)

        self.but_brow .clicked.connect(self.onBrow)
        self.but_save .clicked.connect(self.onSave)
        self.but_close.clicked.connect(self.onClose)
        self.box_file.currentIndexChanged['int'].connect(self.onBox)
 
        self.startFileBrowser(selected_file)

        self.showToolTips()
        self.setStyle()

        #self.guifilebrowser = self


    def showToolTips(self):
        #self           .setToolTip('This GUI is intended for run control and monitoring.')
        self.but_close .setToolTip('Close this window.')


    def setStyle(self):
        style.set_styles()

        self.           setStyleSheet(style.styleBkgd)
        self.tit_status.setStyleSheet(style.styleTitle)
        self.tit_file  .setStyleSheet(style.styleTitle)
        self.tit_file  .setFixedWidth(25)
        self.tit_file  .setAlignment (Qt.AlignRight)
        self.box_file  .setStyleSheet(style.styleButton) 
        self.but_brow  .setStyleSheet(style.styleButton)
        self.but_brow  .setFixedWidth(60)
        self.but_save  .setStyleSheet(style.styleButton)
        self.but_close .setStyleSheet(style.styleButton)
        self.box_txt   .setReadOnly  (not self.is_editable)
        self.box_txt   .setStyleSheet(style.styleWhiteFixed) 
        self.layout().setContentsMargins(0,0,0,0)


    def setListOfFiles(self, list):
        self.list_of_files  = ['Click on this box and select file from pop-up-list']
        self.list_of_files += list
        self.box_file.clear()
        self.box_file.addItems(self.list_of_files)
        #self.box_file.setCurrentIndex( 0 )


    def setParent(self,parent) :
        self.parent = parent


    #def resizeEvent(self, e):
        #logger.debug('resizeEvent') 
        #pass


    #def moveEvent(self, e):
        #logger.debug('moveEvent') 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        #pass


    def closeEvent(self, event):
        logger.debug('closeEvent')
        #self.saveLogTotalInFile() # It will be saved at closing of GUIMain

        #try    : cp.guimain.butFBrowser.setStyleSheet(style.styleButtonBad)
        #except : pass

        #try    : cp.guidark.but_browse.setStyleSheet(style.styleButtonBad)
        #except : pass

        self.box_txt.close()

        #cp.guifilebrowser = None

        #try    : del cp.guifilebrowser # QWFileBrowser
        #except : pass


    def onClose(self):
        logger.debug('onClose')
        self.close()


    def onSave(self):
        logger.debug('onSave')
        path = gu.get_save_fname_through_dialog_box(self, self.fname, 'Select file to save', filter='*.txt')
        if path is None or path == '' : return
        text = str(self.box_txt.toPlainText())
        logger.info('Save in file:\n'+text)
        f=open(path,'w')
        f.write( text )
        f.close() 


    def onBrow(self):
        logger.debug('onBrow - select file')

        path0 ='./'
        if len(self.list_of_files) > 1 : path0 = self.list_of_files[1]

        path = gu.get_open_fname_through_dialog_box(self, path0, 'Select text file for browser', filter='Text files (*.txt *.dat *.data *.cfg *.npy)\nAll files (*)')
        if path is None or path == '' or path == path0 :
            #logger.debug('Loading is cancelled...')
            return

        #logger.info('File selected for browser: %s' % path)        


        if not path in self.list_of_files :
            self.list_of_files.append(path)
            self.box_txt.setText(load_textfile(path))

            self.setListOfFiles(self.list_of_files[1:])
            self.box_file.setCurrentIndex( len(self.list_of_files)-1 )
            self.setStatus(0, 'Status: browsing selected file')

 
    def onBox(self):
        self.fname = str( self.box_file.currentText() )
        #logger.debug('onBox - selected file: ' + self.fname)

        if self.fname == '' : return

        #self.list_of_supported = ['cfg', 'log', 'txt', 'txt-tmp', '', 'dat', 'data']
        self.list_of_supported = ['ALL']
        self.str_of_supported = ''
        for ext in self.list_of_supported : self.str_of_supported += ' ' + ext

        logger.debug('self.fname = %s' % self.fname)
        logger.debug('self.list_of_files: %s' % ', '.join(self.list_of_files))

        if self.list_of_files.index(self.fname) == 0 :
            self.setStatus(0, 'Waiting for file selection...')
            self.box_txt.setText('Click on file-box and select the file from pop-up list...')

        elif os.path.lexists(self.fname) :
            ext = os.path.splitext(self.fname)[1].lstrip('.')

            if ext in self.list_of_supported or self.list_of_supported[0] == 'ALL' :
                self.box_txt.setText(load_textfile(self.fname))
                self.setStatus(0, 'Status: enjoy browsing the selected file...')

            else :
                self.box_txt.setText('Sorry, but this browser supports text files with extensions:' +
                                     self.str_of_supported + '\nTry to select another file...')
                self.setStatus(1, 'Status: ' + ext + '-file is not supported...')

        else :
            self.box_txt.setText( 'Selected file is not avaliable...\nTry to select another file...')
            self.setStatus(2, 'Status: WARNING: FILE IS NOT AVAILABLE!')


    def startFileBrowser(self, selected_file=None) :
        logger.debug('Start the QWFileBrowser.')
        self.setStatus(0, 'Waiting for file selection...')

        if selected_file is not None and selected_file in self.list_of_files :
            index = self.list_of_files.index(selected_file)
            self.box_file.setCurrentIndex( index )

        elif len(self.list_of_files) == 2 :
            self.box_file.setCurrentIndex( 1 )
            #self.onBox()      
        else :
            self.box_file.setCurrentIndex( 0 )
        #self.box_txt.setText('Click on file-box and select the file from pop-up list...')


    def appendGUILog(self, msg='...'):
        self.box_txt.append(msg)
        scrol_bar_v = self.box_txt.verticalScrollBar() # QScrollBar
        scrol_bar_v.setValue(scrol_bar_v.maximum()) 

        
    def setStatus(self, status_index=0, msg=''):
        list_of_states = ['Good','Warning','Alarm']
        if status_index == 0 : self.tit_status.setStyleSheet(style.styleStatusGood)
        if status_index == 1 : self.tit_status.setStyleSheet(style.styleStatusWarning)
        if status_index == 2 : self.tit_status.setStyleSheet(style.styleStatusAlarm)

        #self.tit_status.setText('Status: ' + list_of_states[status_index] + msg)
        self.tit_status.setText(msg)

#------------------------------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    widget = QWFileBrowser ()
    widget.show()
    app.exec_()

#------------------------------
