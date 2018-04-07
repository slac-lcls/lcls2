#------------------------------
"""
:py:class:`QWTabBar` - Re-implementation of QWTabBar
========================================

Usage::

    # Import
    from psana.graphqt.QWTabBar import QWTabBar

    # Methods - see test

See:
    - :py:class:`QWTabBar`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2017-02-08 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-16
"""
#------------------------------

from PyQt5.QtWidgets import QTabBar#, QWidget
from PyQt5.QtCore import QPoint, QSize # Qt, QEvent
from psana.graphqt.QWIcons import icon

#------------------------------

class QWTabBar(QTabBar) :
    """Re-implementation of QTabBar - add "+" tab
    """
    def __init__ (self, parent=None, width=80) :
        QTabBar.__init__(self, parent)
        self._name = self.__class__.__name__
        icon.set_icons()

        self.tab_width = width
        self.tabi_add = None
        self.setTabsClosable(True)
        self.setMovable(True)
        self.setExpanding(False) # need it to lock tab width

        self.make_tab_bar()

        self.set_style()
        self.set_tool_tips()

    #-------------------

    def make_tab_bar(self):

        w = self

        it0 = w.addTab('Tab0')
        it1 = w.addTab('Tab1')
        it2 = w.addTab('Tab2')
        it3 = w.addTab('Tab3')
        it4 = w.addTab('Tab4')
        it5 = w.addTab('Tab5')

        itab = 0
        w.setTabIcon(itab, icon.icon_folder_closed)
        w.tabButton(itab, QTabBar.RightSide).resize(0, 0) # hide closing button
        ##w.tabButton(itab, QTabBar.RightSide).hide() # hide closing button

        itab = 1
        but_close = QPushButton('x')
        but_close.setFixedWidth(30)
        but_close.clicked.connect(self.on_tab_close)
        but_close.setIcon(icon.icon_table)
        w.setTabButton(itab, QTabBar.RightSide, but_close)

        itab = 2
        w.tabButton(itab, QTabBar.RightSide).hide() # hide closing button
    
        #itab = 2
        #w.setTabIcon(itab, icon.icon_table)
        #but_tab2_close = w.tabButton(itab, QTabBar.RightSide)
        #but_tab2_close.clicked.connect(self.on_tab_close)

        #itab = 3
        #but_tab3_close = w.tabButton(itab, QTabBar.RightSide)
        #but_tab3_close.clicked.connect(self.on_tab_close)

        self.currentChanged[int].connect(self.on_current_changed)
        self.tabCloseRequested[int].connect(self.on_tab_close_request)
        self.tabMoved[int,int].connect(self.on_tab_moved)

      #but_close.setVisible(True)
        #print(str(but_close))
    
        #print(dir(but_close))

    #-------------------

    def on_current_changed(self, itab) :
        print('%s.on_current_changed tab index:%d' % (self._name, itab))

    #-------------------

    def current_tab_index_and_name(self):
        tab_ind  = self.currentIndex()
        tab_name = str(self.tabText(tab_ind))
        return tab_ind, tab_name

    #-------------------

    def on_tab_close(self) :
        tab_ind, tab_name = self.current_tab_index_and_name()
        print('%s.on_tab_close tab index:%d name:%s' % (self._name, tab_ind, tab_name))

    #-------------------

    def on_tab_close_request(self, itab) :
        print('%s.on_tab_close_request tab index:%d' % (self._name, itab))

    #-------------------

    def on_tab_moved(self, inew, iold) :
        print('%s.on_tab_close_request tab index begin:%d -> end:%d' % (self._name, iold, inew))

    #-------------------

    def set_tool_tips(self):
        self.setToolTip('This is a tabbar') 

    def set_style(self):
        self.setMinimumWidth(600)
        #self.setGeometry(10, 25, 500, 50)
        ss = "QTabBar::close-button { image: url(close.png) subcontrol-position: left; }"\
             "QTabBar::close-button:hover { image: url(close-hover.png) }"
        self.setStyleSheet(ss)


    def enterEvent(self, e) :
        print('%s.enterEvent' % self._name, e.type())
        #if e.type() == QEvent.Enter :
        #self.setTabEnabled(self.tabi_add, True)
        #self.setTabsClosable(True)
        self.tabi_add = self.addTab('+')
        self.tabButton(self.tabi_add, QTabBar.RightSide).resize(0, 0) # hide closing button

        #self.tabButton(self.tabi_add, QTabBar.RightSide).hide() # hide closing button


    def leaveEvent(self, e) :
        print('%s.leaveEvent' % self._name, e.type())
        #if e.type() == QtCore.QEvent.Leave :
        #self.setTabEnabled(self.tabi_add, False)
        #self.setTabsClosable(False)
        if self.tabi_add is not None : self.removeTab(self.tabi_add)
        self.tabi_add = None


    def tabSizeHint(self, index):
        w = self.tab_width
        if index==self.tabi_add or (self.tabText(index)[0] == '+') : w = 30
        h = QTabBar.tabSizeHint(self, index).height()
        return QSize(w, h)


    #def event(self, e) :
    #    QTabBar.event(self, e)
    #    print('%s.event' % self._name, e.type())


    #def setExpanding(self, enabled) :
    #    QtWidgets.QTabBar.setExpanding(self, False)


    #def mouseMoveEvent(self, e) :
    #    print '%s.mouseMoveEvent x,y=' % self._name, e.x(), e.y()


    def mouseHoverEvent(self, e) :
        print('%s.mouseHoverEvent' % self._name)


    #def closeEvent(self, event):
        #print '%s.closeEvent' % self._name
        #log.info('closeEvent', self._name)

        #try    : self.gui_win.close()
        #except : pass

#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication, QPushButton
    app = QApplication(sys.argv)
    w  = QWTabBar()
    w.setWindowTitle('Widget with tabs')
    w.move(QPoint(50,50))
    w.show()
    app.exec_()
    del w
    del app

#------------------------------
