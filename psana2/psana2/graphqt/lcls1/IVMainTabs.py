
#------------------------------
"""
Class :py:class:`IVMainTabs` - tabs in IVMain window
====================================================

Usage ::

    import sys
    from PyQt4 import QtGui
    from graphqt.IVMainTabs import IVMainTabs
    app = QtGui.QApplication(sys.argv)
    w = IVMainTabs()
    w.show()
    app.exec_()

See:
    - :class:`IVMain`
    - :class:`IVMainTabs`
    - :class:`IVMainButtons`
    - :class:`IVImageCursorInfo`
    - :class:`IVConfigParameters`
    - :class:`IVTabDataControl`
    - :class:`IVTabFileName`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2017-02-18 by Mikhail Dubrovin
"""
#------------------------------
import sys
import os

from PyQt4 import QtGui, QtCore

from graphqt.IVConfigParameters import cp
from graphqt.Logger             import log
from graphqt.IVTabDataControl   import IVTabDataControl
from graphqt.IVTabFileName      import IVTabFileName

#from graphqt.QIcons            import icon
#from graphqt.Frame             import Frame

#------------------------------

#class IVMainTabs(Frame) :
class IVMainTabs(QtGui.QWidget) :
    """GUI for tabs
    """
    orientation = 'H'
    #orientation = 'V'
    tab_names = ['File', 'Data', 'Mask Editor', 'Peaks', 'Etc.']
    number_of_tabs = len(tab_names)

    def __init__ (self, parent=None, app=None) :

        #Frame.__init__(self, parent, mlw=1)
        QtGui.QWidget.__init__(self, parent)
        self._name = self.__class__.__name__

        self.current_tab = cp.current_tab
        cp.guitabs = self

        self.gui_win = None

        self.hboxW = QtGui.QHBoxLayout()

        self.make_tab_bar()
        self.gui_selector()

        if self.orientation == 'H' : self.box = QtGui.QVBoxLayout(self) 
        else :                       self.box = QtGui.QHBoxLayout(self) 

        self.box.addWidget(self.tab_bar)
        self.box.addLayout(self.hboxW)
        self.setLayout(self.box)
        self.box.addStretch(1)

        self.show_tool_tips()
        self.set_style()

    #-------------------

    def show_tool_tips(self):
        self.tab_bar.setToolTip('Select tab') 


    def set_style(self):
        self.setGeometry(10, 25, 400, 600)
        self.setMinimumHeight(250)
        self.setContentsMargins(QtCore.QMargins(-9,-9,-9,-9))
        #self.setContentsMargins(QtCore.QMargins(-5,-5,-5,-5))

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
        log.info('%s.make_tab_bar - set tab: %s' % (self._name, tab_name))
        self.tab_bar.setCurrentIndex(tab_index)


    def gui_selector(self):

        try    : self.gui_win.close()
        except : pass

        try    : del self.gui_win
        except : pass

        self.gui_win = None

        for itab in range(len(self.tab_names)) :
            tab_name = self.current_tab.value()
            if tab_name == self.tab_names[itab] :
               self.gui_win = IVTabDataControl(cp, log, parent=None, show_mode=017, show_mode_evctl=017) if tab_name=='Data' else\
                              IVTabFileName(parent=None, show_mode=01)    if tab_name=='File' else\
                              QtGui.QTextEdit('Window for %s'%self.tab_names[itab])

        self.hboxW.addWidget(self.gui_win)


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
        #print 'IVMainTabs resizeEvent: %s' % str(self.size())


    #def moveEvent(self, e):
        #log.debug('moveEvent', self._name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #log.debug('moveEvent - pos:' + str(self.position), __name__)       
        #pass


    def closeEvent(self, e):
        log.debug('closeEvent', self._name)
        #log.info('%s.closeEvent' % self._name)

        try    : self.gui_win.close()
        except : pass

        #try    : del self.gui_win
        #except : pass

        QtGui.QWidget.closeEvent(self, e)
        cp.guitabs = None


    def on_exit(self):
        log.debug('on_exit', self._name)
        self.close()

#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    app = QtGui.QApplication(sys.argv)
    w = IVMainTabs()
    w.move(QtCore.QPoint(50,50))
    w.setWindowTitle(w._name)
    w.show()
    app.exec_()

#------------------------------
