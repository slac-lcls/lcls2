
#  QWStatus ...
#--------------------------------
__version__ = "v2018-02-20"
#--------------------------------

from time import time

from PyQt5.QtWidgets import QGroupBox, QTextEdit, QVBoxLayout
from PyQt5.QtCore import QTimer #QMargins
from psana.graphqt.Styles import style
from psana.graphqt.QWIcons import icon

#---------------------

#class QWStatus(QWidget) :
class QWStatus(QGroupBox) :
    """GUI State"""

    def __init__(self, parent=None, msg='No message in QWStatus...') :

        QGroupBox.__init__(self, 'State', parent)
        #QWidget.__init__(self, parent)

        icon.set_icons()
        try : self.setWindowIcon(icon.icon_logviewer)
        except : pass

        self.box_txt        = QTextEdit(self)
        #self.tit_status     = QLabel(' State ', self)

        #self.setTitle('My status')

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.box_txt)
        self.setLayout(self.vbox)

        #self.connect( self.but_close, QtCore.SIGNAL('clicked()'), self.onClose )
 
        self.setStatusMessage(msg)

        self.showToolTips()
        self.setStyle()

        #cp.guistatus = self
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timeout)
        self.timer.start(1000)


    def showToolTips(self):
        #self           .setToolTip('This GUI is intended for run control and monitoring.')
        #self.but_close .setToolTip('Close this window.')
        pass


    def setStyle(self):
        self.           setStyleSheet (style.styleBkgd)
        #self.tit_status.setStyleSheet (style.styleTitle)
        #self.tit_status.setStyleSheet (style.styleDefault)
        #self.tit_status.setStyleSheet (style.styleTitleInFrame)
        self.box_txt   .setReadOnly   (True)
        #self.box_txt   .setStyleSheet (style.styleBkgd)
        self.box_txt   .setStyleSheet (style.styleWhiteFixed)

        self.layout().setContentsMargins(0,0,0,0)
 
        #self.setMinimumHeight(60)
        self.setMinimumSize(300,60)


    def setParent(self,parent) :
        self.parent = parent


    #def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        #self.box_txt.setGeometry(self.contentsRect())


    #def moveEvent(self, e):
        #logger.debug('moveEvent', __name__) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())
        #pass


    def on_timeout(self) :
        self.timer.start(1000)
        self.setStatusMessage(msg='Time %.3f sec' % time())


    def closeEvent(self, event):
        #logger.debug('closeEvent', __name__)
        #self.saveLogTotalInFile() # It will be saved at closing of GUIMain

        #try    : cp.guimain.butLogger.setStyleSheet(style.styleButtonBad)
        #except : pass

        self.box_txt.close()

        #try    : del cp.guistatus # QWStatus
        #except : pass

        #cp.guistatus = None


    def onClose(self):
        #logger.debug('onClose', __name__)
        self.close()


    def setStatusMessage(self, msg='msg is empty...') :
        #logger.debug('Set status message',__name__)
        self.box_txt.setText(msg)
        #self.setStatus(0, 'Status: unknown')

#------------------------------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = QWStatus()
    w.setGeometry(100, 100, 300, 60)
    w.setWindowTitle('GUI Status')
    w.setStatusMessage('Test of QWStatus...')

    w.show()
    app.exec_()

    del w
    del app

#------------------------------
