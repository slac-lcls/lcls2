#------------------------------
"""Class :py:class:`CMWMainTabs` is a QWidget for tabs and switching window
==============================================================================

Usage ::
    # Test: python lcls2/psana/psana/graphqt/CMWMainTabs.py

Created on 2017-02-18 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-26 by Mikhail Dubrovin
"""
#------------------------------

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QTabBar, QTextEdit, QSplitter
from PyQt5.QtGui import QColor # QPalette, QSizePolicy
from PyQt5.QtCore import Qt #, QPoint

from psana.graphqt.CMConfigParameters import cp
from psana.pyalgos.generic.Logger import logger

from psana.graphqt.QWDateTimeSec import QWDateTimeSec
from psana.graphqt.CMWConfig     import CMWConfig
from psana.graphqt.CMWDBMain     import CMWDBMain

#------------------------------

class CMWMainTabs(QWidget) :
    """GUI for tabs and associated widgets
    """
    tab_names   = ['CDB', 't-converter', 'Configuration', 'Mon-A', 'Mon-B']

    def __init__ (self, parent=None, app=None) :

        QWidget.__init__(self, parent)
        self._name = self.__class__.__name__

        cp.cmwmaintabs = self

        self.box_layout = QHBoxLayout()

        self.gui_win = None
        self.make_tab_bar()
        self.gui_selector(cp.main_tab_name.value()) # USES self.box_layout

        #self.whbox = QWidget(self)
        #self.whbox.setLayout(self.box_layout)

        #self.vspl = QSplitter(Qt.Vertical)
        #self.vspl.addWidget(self.tab_bar)
        #self.vspl.addWidget(self.whbox)

        self.box = QVBoxLayout(self)
        self.box.addWidget(self.tab_bar)
        self.box.addLayout(self.box_layout)
        #self.box.addStretch(1)
        #self.box.addWidget(self.vspl)

        self.setLayout(self.box)

        self.show_tool_tips()
        self.set_style()
        #gu.printStyleInfo(self)

        #self.move(10,25)
        
        #print('End of init')
        #self.set_tabs_visible(False)

    #--------------------------

    def show_tool_tips(self):
        self.setToolTip('Main tab window')


    def set_style(self):

        from psana.graphqt.Styles import style
        from psana.graphqt.QWIcons import icon
        icon.set_icons()

        #self.tab_bar.setContentsMargins(-9,-9,-9,-9) # QMargins(-5,-5,-5,-5)

        self.setWindowIcon(icon.icon_monitor)
        self.setStyleSheet(style.styleBkgd)
        self.setContentsMargins(-9,-9,-9,-9) # QMargins(-5,-5,-5,-5)

        #self.palette = QPalette()
        #self.resetColorIsSet = False

        #self.butELog    .setIcon(icon.icon_mail_forward)
        #self.butFile    .setIcon(icon.icon_save)
        #self.butExit    .setIcon(icon.icon_exit)
        #self.butLogger  .setIcon(icon.icon_logger)
        #self.butFBrowser.setIcon(icon.icon_browser)
        #self.butSave    .setIcon(icon.icon_save_cfg)
        #self.butStop    .setIcon(icon.icon_stop)

        #self.setMinimumHeight(250)
        #self.setMinimumWidth(550)

        #self.adjustSize()
        #self.        setStyleSheet(style.styleBkgd)
        #self.butSave.setStyleSheet(style.styleButton)
        #self.butFBrowser.setVisible(False)
        #self.butExit.setText('')
        #self.butExit.setFlat(True)
        #self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
 
    #-------------------

    def make_tab_bar(self) :
        self.tab_bar = QTabBar()

        #len(self.tab_names)
        for tab_name in self.tab_names :
            tab_ind = self.tab_bar.addTab(tab_name)
            self.tab_bar.setTabTextColor(tab_ind, QColor('blue')) #gray, red, grayblue

        #self.tab_bar.setTabsClosable(True)
        #self.tab_bar.setMovable(True)
        self.tab_bar.setShape(QTabBar.RoundedNorth)

        tab_index = self.tab_names.index(cp.main_tab_name.value())            
        self.tab_bar.setCurrentIndex(tab_index)
        logger.debug(' make_tab_bar - set tab index: %d'%tab_index, self._name)

        self.tab_bar.currentChanged['int'].connect(self.on_tab_bar)
        self.tab_bar.tabCloseRequested.connect(self.on_tab_close_request)
        self.tab_bar.tabMoved[int,int].connect(self.on_tab_moved)

    #--------------------------

    def gui_selector(self, tab_name):

        if self.gui_win is not None :
            #self.box_layout.removeWidget(self.gui_win)
            #self.gui_win.setVisible(False)
            self.gui_win.close()
            del self.gui_win

        w_height = 200

        if tab_name == 'CDB' :
            self.gui_win = CMWDBMain()
            w_height = 500

        elif tab_name == 'Configuration' :
            self.gui_win = CMWConfig()

        elif tab_name == 't-converter' :
            self.gui_win = QWDateTimeSec(logger=logger)
            self.gui_win.setMaximumWidth(500)
            w_height = 80
            #self.gui_win.setMaximumHeight(w_height)

        else :
            self.gui_win = QTextEdit('Default window for tab %s' % tab_name)

        #self.gui_win.setFixedHeight(w_height)
        self.gui_win.setMinimumHeight(w_height)
        self.gui_win.setVisible(True)
        self.box_layout.addWidget(self.gui_win)

        #self.setStatus(0, s_msg)

    #-------------------

    def current_tab_index_and_name(self):
        tab_ind  = self.tab_bar.currentIndex()
        tab_name = str(self.tab_bar.tabText(tab_ind))
        return tab_ind, tab_name

    #--------------------------

    def on_tab_bar(self, ind):
        tab_ind, tab_name = self.current_tab_index_and_name()
        logger.debug('Selected tab name: %s index: %d' % (tab_name,tab_ind), self._name)
        cp.main_tab_name.setValue(tab_name)
        self.gui_selector(tab_name)

    #-------------------

    def on_tab_close_request(self, ind):
        logger.debug('%s.on_tab_close_request ind:%d' % (self._name, ind))
        #self.tab_bar.removeTab(ind)
        #print('%s.on_tab_close_request tab index:%d' % (self._name, itab))

    #-------------------

    def on_tab_moved(self, inew, iold) :
        print('%s.on_tab_close_request tab index begin:%d -> end:%d' % (self._name, iold, inew))

    #-------------------
 
    #def resizeEvent(self, e):
        #pass
        #self.frame.setGeometry(self.rect())
        #logger.debug('resizeEvent', self._name) 
        #print('CMWMainTabs resizeEvent: %s' % str(self.size()))


    #def moveEvent(self, e):
        #logger.debug('moveEvent', self._name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        #pass


    def closeEvent(self, e):
        logger.debug('closeEvent', self._name)
        #logger.info('%s.closeEvent' % self._name)

        #try    : self.gui_win.close()
        #except : pass

        #try    : del self.gui_win
        #except : pass

        if self.gui_win is not None :
            self.gui_win.close()

        QWidget.closeEvent(self, e)


    def onExit(self):
        logger.debug('onExit', self._name)
        self.close()


    def set_tabs_visible(self, is_visible):
        #print('XXX CMWMainTabs.set_tabs_visible: is_visible', is_visible)
        self.tab_bar.setVisible(is_visible)


    def tab_bar_is_visible(self) :
        return self.tab_bar.isVisible()


    def view_hide_tabs(self) :
        #self.set_tabs_visible(not self.tab_bar.isVisible())
        self.tab_bar.setVisible(not self.tab_bar.isVisible())


    def key_usage(self) :
        return 'Keys:'\
               '\n  V - view/hide tabs'\
               '\n'


    def keyPressEvent(self, e) :
        #print('keyPressEvent, key=', e.key())       
        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_V : 
            self.view_hide_tabs()

        else :
            print(self.key_usage())


#------------------------------

if __name__ == "__main__" :

    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    w = CMWMainTabs()
    w.setGeometry(10, 25, 400, 600)
    w.setWindowTitle(w._name)
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------
