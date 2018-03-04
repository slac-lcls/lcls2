#------------------------------
"""Class :py:class:`CMMainTabs` is a QWidget for tabs and switching window
========================================================================

Usage ::

Created on 2017-02-18 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-26 by Mikhail Dubrovin
"""
#------------------------------

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QTabBar, QTextEdit
from PyQt5.QtGui import QPalette, QColor# , QSizePolicy

#from psana.pyalgos.generic.PSConfigParameters import cp

from psana.pyalgos.generic.Logger import logger

from psana.graphqt.QWDateTimeSec import QWDateTimeSec
#from expmon.CMConfMonV1       import CMConfMonV1
#from expmon.CMConfMonI        import CMConfMonI

#------------------------------

class CMMainTabs(QWidget) :
    """GUI for tabs
    """
    #orientation = 'H'

    TAB1 = 1
    TAB2 = 2
    TCONV = 3

    def __init__ (self, parent=None, app=None, orientation='H') :

        #Frame.__init__(self, parent, mlw=1)
        QWidget.__init__(self, parent)
        self._name = self.__class__.__name__

        self.orientation = orientation

        self.monitors    = []
        self.tab_types   = [self.TAB1,    self.TAB1,    self.TAB1,    self.TAB2,   self.TCONV]
        self.tab_names   = ['Mon-A', 'Mon-B', 'Mon-C', 'Mon-D', 't-converter']
        self.current_tab = self.tab_names[1]

        self.gui_win = None

        self.hboxW = QHBoxLayout()

        self.make_monitors()
        self.make_tab_bar()
        self.gui_selector()

        if self.orientation == 'H' : self.box = QVBoxLayout(self) 
        else :                       self.box = QHBoxLayout(self) 

        self.box.addWidget(self.tab_bar)
        self.box.addLayout(self.hboxW)
        self.box.addStretch(1)

        self.setLayout(self.box)

        self.show_tool_tips()
        self.set_style()
        #gu.printStyleInfo(self)

        #self.move(10,25)
        
        #print('End of init')
        
    #--------------------------

    def show_tool_tips(self):
        pass
        #self.butExit.setToolTip('Close all windows and \nexit this program') 

    def set_style(self):

        #from psana.graphqt.Styles import style

        from psana.graphqt.QWIcons import icon
        icon.set_icons()
        self.setWindowIcon(icon.icon_monitor)
        #self.palette = QPalette()
        #self.resetColorIsSet = False

        #self.butELog    .setIcon(icon.icon_mail_forward)
        #self.butFile    .setIcon(icon.icon_save)
        #self.butExit    .setIcon(icon.icon_exit)
        #self.butLogger  .setIcon(icon.icon_logger)
        #self.butFBrowser.setIcon(icon.icon_browser)
        #self.butSave    .setIcon(icon.icon_save_cfg)
        #self.butStop    .setIcon(icon.icon_stop)

        self.setMinimumHeight(250)
        self.setMinimumWidth(550)
        self.setContentsMargins(-5,-5,-5,-5) # QMargins(-5,-5,-5,-5)

        #self.adjustSize()
        #self.        setStyleSheet(style.styleBkgd)
        #self.butSave.setStyleSheet(style.styleButton)
        #self.butFBrowser.setVisible(False)
        #self.butExit.setText('')
        #self.butExit.setFlat(True)
        #self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
 

    def make_tab_bar(self,mode=None) :
        #if mode != None : self.tab_bar.close()
        self.tab_bar = QTabBar()
        #self.tab_bar = CMTabBar(width=100)

        #len(self.tab_names)
        for tab_name in self.tab_names :
            tab_ind = self.tab_bar.addTab(tab_name)
            self.tab_bar.setTabTextColor(tab_ind, QColor('blue')) #gray, red, grayblue

        #self.tab_bar.setTabsClosable(True)

        if self.orientation == 'H' :
            self.tab_bar.setShape(QTabBar.RoundedNorth)
        else :
            self.tab_bar.setShape(QTabBar.RoundedWest)

        self.set_tab_by_name(self.current_tab)
            
        #self.connect(self.tab_bar, QtCore.SIGNAL('currentChanged(int)'), self.on_tab_bar)
        #self.connect(self.tab_bar, QtCore.SIGNAL('tabCloseRequested(int)'), self.on_tab_close)
        self.tab_bar.currentChanged['int'].connect(self.on_tab_bar)
        self.tab_bar.tabCloseRequested.connect(self.on_tab_close)

    def set_tab_by_name(self, tab_name) :
        try    :
            tab_index = self.tab_names.index(tab_name)
        except :
            tab_index = 0
            self.current_tab = self.tab_names[tab_index]
        logger.debug('set_tab_by_name - set tab: %s' % tab_name, self._name)
        self.tab_bar.setCurrentIndex(tab_index)

    #--------------------------

    def reset_monitors(self):
        for mon in self.monitors :
            mon.reset_monitor()

    #--------------------------

    def make_monitors(self):
        self.monitors = []
        for itab in range(len(self.tab_names)) :
            self.monitors.append(QTextEdit('tab1 window: %d'%itab) if self.tab_types[itab] == self.TAB1 else\
                                 QTextEdit('tab2 window: %d'%itab) if self.tab_types[itab] == self.TAB1 else\
                                 QWDateTimeSec(logger=logger) if self.tab_types[itab] == self.TCONV else\
                                 QTextEdit('Unknown tab type for tab %d'%itab)\
                                )

    def gui_selector(self):
        if self.gui_win is not None :
            self.hboxW.removeWidget(self.gui_win)
            self.gui_win.setVisible(False)

        for itab in range(len(self.tab_names)) :
            if self.current_tab  == self.tab_names[itab] :
               self.gui_win = self.monitors[itab]
               self.gui_win.setVisible(True)

        self.hboxW.addWidget(self.gui_win)

    #--------------------------

    def on_tab_bar(self, ind):
        tab_name = str(self.tab_bar.tabText(ind))
        self.current_tab = tab_name
        msg = 'Selected tab: %i - %s' % (ind, tab_name)
        logger.info(msg, self._name)
        self.gui_selector()


    def on_tab_close(self, ind):
        logger.debug('%s.on_tab_close ind:%d' % (self._name, ind))
        self.tab_bar.removeTab(ind)


    #def resizeEvent(self, e):
        #pass
        #self.frame.setGeometry(self.rect())
        #logger.debug('resizeEvent', self._name) 
        #print('CMMainTabs resizeEvent: %s' % str(self.size()))


    #def moveEvent(self, e):
        #logger.debug('moveEvent', self._name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        #pass


    def closeEvent(self, e):
        logger.debug('closeEvent', self._name)
        #logger.info('%s.closeEvent' % self._name)

        for mon in self.monitors :
            mon.close()

        #try    : self.gui_win.close()
        #except : pass

        #try    : del self.gui_win
        #except : pass

        QWidget.closeEvent(self, e)


    def onExit(self):
        logger.debug('onExit', self._name)
        self.close()

#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :

    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    w = CMMainTabs()
    w.setGeometry(10, 25, 400, 600)
    w.setWindowTitle(w._name)
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------
