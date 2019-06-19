#------------------------------
"""Class :py:class:`CGWMainTabs` is a QWidget for tabs and switching window
==============================================================================

Usage ::
    # Test: python lcls2/psdaq/psdaq/control_gui/CGWMainTabs.py

Created on 2017-05-07 by Mikhail Dubrovin
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

#------------------------------

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QTabBar, QTextEdit, QSplitter, QSizePolicy
from PyQt5.QtGui import QColor # QPalette
from PyQt5.QtCore import Qt, QSize #, QPoint

from psdaq.control_gui.CGWMainTabExpert import CGWMainTabExpert
from psdaq.control_gui.CGWMainTabUser   import CGWMainTabUser

#------------------------------

class CGWMainTabs(QWidget) :
    """GUI for tabs and associated widgets
    """
    tab_names = ['User', 'Expert']
    tab_ind_user   = 0
    tab_ind_expert = 1

    def __init__ (self, **kwargs) :

        self.kwargs = kwargs

        parent = kwargs.get('parent', None)
        QWidget.__init__(self, parent)

        self.box_layout = QHBoxLayout()

        self.gui_win = None
        self.make_tab_bar()
        self.gui_selector(self.tab_names[1])

        self.box = QVBoxLayout(self)
        self.box.addWidget(self.tab_bar)
        self.box.addLayout(self.box_layout)
        #self.box.addStretch(1)
        self.setLayout(self.box)

        self.show_tool_tips()
        self.set_style()

    #--------------------------

    def show_tool_tips(self):
        self.setToolTip('Main tab window')

    #-------------------

    def sizeHint(self):
        height = 400 if self.tab_bar.currentIndex()==self.tab_ind_expert else 50
        return QSize(300, height) 

    #-------------------

    def set_style(self):

        from psdaq.control_gui.Styles import style
        from psdaq.control_gui.QWIcons import icon
        icon.set_icons()

        self.setWindowIcon(icon.icon_monitor)
        self.setStyleSheet(style.styleBkgd)
        self.layout().setContentsMargins(0,0,0,0)

        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)

        #self.setMinimumSize(300,300)

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

        tab_index = self.tab_ind_expert # self.tab_names.index(self.tab_names[self.tab_ind_expert])            
        self.tab_bar.setCurrentIndex(tab_index)
        logger.debug('make_tab_bar - set tab index: %d'%tab_index)

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

        w_height = 100

        if tab_name == self.tab_names[1] :
            self.gui_win = CGWMainTabExpert(**self.kwargs)
            self.setMinimumHeight(400)
            w_height = 350

        elif tab_name == self.tab_names[0] :
            self.gui_win = CGWMainTabUser(**self.kwargs)
            #self.gui_win = QTextEdit(tab_name)
            self.setFixedHeight(110)
            #self.setMinimumHeight(100)
            #self.gui_win.setFixedHeight(100)
            w_height = 70

        else :
            self.gui_win = QTextEdit('Default window for tab %s' % tab_name)

        #self.gui_win.setMaximumHeight(w_height)
        #self.gui_win.setMaximumWidth(500)
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

    #-------------------

    def on_tab_bar(self, ind):
        tab_ind, tab_name = self.current_tab_index_and_name()
        logger.info('Selected tab "%s"' % tab_name)
        #cp.main_tab_name.setValue(tab_name)
        self.gui_selector(tab_name)

    #-------------------

    def on_tab_close_request(self, ind):
        logger.debug('on_tab_close_request ind:%d' % ind)
        #self.tab_bar.removeTab(ind)
        #logger.debug('on_tab_close_request tab index:%d' % (itab))

    #-------------------

    def on_tab_moved(self, inew, iold) :
        logger.debug('on_tab_close_request tab index begin:%d -> end:%d' % (iold, inew))

    #-------------------
 
    #def resizeEvent(self, e):
        #pass
        #self.frame.setGeometry(self.rect())
        #logger.debug('resizeEvent') 
        #logger.debug('CGWMainTabs resizeEvent: %s' % str(self.size()))


    #def moveEvent(self, e):
        #logger.debug('moveEvent') 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position))       
        #pass


    def closeEvent(self, e):
        logger.debug('CGWMainTabs.closeEvent')

        #try    : self.gui_win.close()
        #except : pass

        #try    : del self.gui_win
        #except : pass

        self.tab_bar.close()

        if self.gui_win is not None :
            self.gui_win.close()

        QWidget.closeEvent(self, e)


    def onExit(self):
        logger.debug('onExit')
        self.close()


    def set_tabs_visible(self, is_visible):
        logger.debug('set_tabs_visible: is_visible %s' % is_visible)
        self.tab_bar.setVisible(is_visible)


    def tab_bar_is_visible(self) :
        return self.tab_bar.isVisible()


    def view_hide_tabs(self) :
        self.tab_bar.setVisible(not self.tab_bar.isVisible())


    def key_usage(self) :
        return 'Keys:'\
               '\n  V - view/hide tabs'\
               '\n'

    if __name__ == "__main__" :
      def keyPressEvent(self, e) :
        #logger.debug('keyPressEvent, key=%s' % e.key())       
        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_V : 
            self.view_hide_tab()

        else :
            logger.debug(self.key_usage())

#------------------------------

if __name__ == "__main__" :

    from psdaq.control_gui.CGDaqControl import daq_control, DaqControlEmulator, Emulator
    daq_control.set_daq_control(DaqControlEmulator())

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    kwargs = {'parent':None, 'parent_ctrl':Emulator()}
    w = CGWMainTabs(**kwargs)
    w.setGeometry(10, 25, 400, 600)
    w.setWindowTitle('CGWMainTabs')
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------
