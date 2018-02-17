
#------------------------------
"""GUI for tabs and switching window.
   Created: 2017-02-18
   Author : Mikhail Dubrovin
"""
#------------------------------
import sys
import os

from PyQt4 import QtGui, QtCore

from expmon.EMConfigParameters import cp
from expmon.Logger             import log
from expmon.EMQConfMonV1       import EMQConfMonV1
from expmon.EMQConfMonI        import EMQConfMonI
#from expmon.EMQTabBar          import EMQTabBar
#from graphqt.QIcons            import icon
#from expmon.EMQFrame           import Frame
#import time   # for sleep(sec)

#------------------------------

#class EMQTabs(Frame) :
class EMQTabs(QtGui.QWidget) :
    """GUI for tabs
    """
    orientation = 'H'
    #orientation = 'V'

    def __init__ (self, parent=None, app=None) :

        #Frame.__init__(self, parent, mlw=1)
        QtGui.QWidget.__init__(self, parent)
        self._name = self.__class__.__name__

        self.MON1        = cp.MON1
        self.tab_types   = cp.tab_types
        self.tab_names   = cp.tab_names
        self.current_tab = cp.current_tab
        cp.guitabs       = self

        #icon.set_icons()
        #self.setWindowIcon(icon.icon_monitor)
        #self.palette = QtGui.QPalette()
        #self.resetColorIsSet = False

        #self.butELog    .setIcon(icon.icon_mail_forward)
        #self.butFile    .setIcon(icon.icon_save)
        #self.butExit    .setIcon(icon.icon_exit)
        #self.butLogger  .setIcon(icon.icon_logger)
        #self.butFBrowser.setIcon(icon.icon_browser)
        #self.butSave    .setIcon(icon.icon_save_cfg)
        #self.butStop    .setIcon(icon.icon_stop)

        self.gui_win = None

        self.hboxW = QtGui.QHBoxLayout()

        self.make_monitors()
        self.make_tab_bar()
        self.gui_selector()

        if self.orientation == 'H' : self.box = QtGui.QVBoxLayout(self) 
        else :                       self.box = QtGui.QHBoxLayout(self) 

        self.box.addWidget(self.tab_bar)
        self.box.addLayout(self.hboxW)
        #self.box.addStretch(1)

        self.setLayout(self.box)

        self.show_tool_tips()
        self.set_style()
        #gu.printStyleInfo(self)

        self.move(10,25)
        
        #print 'End of init'
        
    #--------------------------

    def show_tool_tips(self):
        pass
        #self.butExit.setToolTip('Close all windows and \nexit this program') 

    def set_style(self):
        self.setMinimumHeight(250)
        self.setContentsMargins(QtCore.QMargins(-5,-5,-5,-5))

        #self.adjustSize()
        #self.        setStyleSheet(cp.styleBkgd)
        #self.butSave.setStyleSheet(cp.styleButton)
        #self.butFBrowser.setVisible(False)
        #self.butExit.setText('')
        #self.butExit.setFlat(True)
        #self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
 

    def make_tab_bar(self,mode=None) :
        #if mode != None : self.tab_bar.close()
        self.tab_bar = QtGui.QTabBar()
        #self.tab_bar = EMQTabBar(width=100)

        #len(self.tab_names)
        for tab_name in self.tab_names :
            tab_ind = self.tab_bar.addTab(tab_name)
            self.tab_bar.setTabTextColor(tab_ind, QtGui.QColor('blue')) #gray, red, grayblue

        #self.tab_bar.setTabsClosable(True)

        if self.orientation == 'H' :
            self.tab_bar.setShape(QtGui.QTabBar.RoundedNorth)
        else :
            self.tab_bar.setShape(QtGui.QTabBar.RoundedWest)

        self.set_tab_by_name(self.current_tab.value())
            
        self.connect(self.tab_bar, QtCore.SIGNAL('currentChanged(int)'), self.on_tab_bar)
        self.connect(self.tab_bar, QtCore.SIGNAL('tabCloseRequested(int)'), self.on_tab_close)


    def set_tab_by_name(self, tab_name) :
        try    :
            tab_index = self.tab_names.index(tab_name)
        except :
            tab_index = 0
            self.current_tab.setValue(self.tab_names[tab_index])
        log.debug('set_tab_by_name - set tab: %s' % tab_name, self._name)
        self.tab_bar.setCurrentIndex(tab_index)

    #--------------------------

    def reset_monitors(self):
        for mon in cp.monitors :
            mon.reset_monitor()

    #--------------------------

    def make_monitors(self):
        cp.monitors = []
        for itab in range(len(self.tab_names)) :
            cp.monitors.append(EMQConfMonV1(None,itab) if self.tab_types[itab] == self.MON1 else\
                               EMQConfMonI(None,itab))
                               #QtGui.QTextEdit('MON2 window: %d'%itab))


    def gui_selector(self):
        if self.gui_win is not None :
            self.hboxW.removeWidget(self.gui_win)
            self.gui_win.setVisible(False)

        for itab in range(len(self.tab_names)) :
            if self.current_tab.value() == self.tab_names[itab] :
               self.gui_win = cp.monitors[itab]
               self.gui_win.setVisible(True)

        self.hboxW.addWidget(self.gui_win)

    #--------------------------

    def on_tab_bar(self, ind):
        tab_name = str(self.tab_bar.tabText(ind))
        self.current_tab.setValue(tab_name)
        msg = 'Selected tab: %i - %s' % (ind, tab_name)
        log.info(msg, self._name)
        self.gui_selector()


    def on_tab_close(self, ind):
        log.debug('%s.on_tab_close ind:%d' % (self._name, ind))
        self.tab_bar.removeTab(ind)


    #def resizeEvent(self, e):
        #pass
        #self.frame.setGeometry(self.rect())
        #log.debug('resizeEvent', self._name) 
        #print 'EMQTabs resizeEvent: %s' % str(self.size())


    #def moveEvent(self, e):
        #log.debug('moveEvent', self._name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #log.debug('moveEvent - pos:' + str(self.position), __name__)       
        #pass


    def closeEvent(self, e):
        log.debug('closeEvent', self._name)
        #log.info('%s.closeEvent' % self._name)

        for mon in cp.monitors :
            mon.close()

        #try    : self.gui_win.close()
        #except : pass

        #try    : del self.gui_win
        #except : pass

        QtGui.QWidget.closeEvent(self, e)


    def onExit(self):
        log.debug('onExit', self._name)
        self.close()

#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    w = EMQTabs()
    w.setGeometry(10, 25, 400, 600)
    w.setWindowTitle(w._name)
    w.move(QtCore.QPoint(50,50))
    w.show()
    app.exec_()

#------------------------------
