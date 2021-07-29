
"""Class :py:class:`FMWTabs` is a QWidget for File Manager tabs and switching window
==========================================================================================

Usage ::
    # Test: python lcls2/psana/psana/graphqt/FMWTabs.py

Created on 2021-07-27 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QTabBar, QTextEdit
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
from psana.graphqt.CMConfigParameters import cp


class FMWTabs(QWidget):
    """GUI for tabs and associated widgets
    """
    tab_names = ['LCLS1', 'LCLS1->LCLS2', 'LCLS2']

    def __init__ (self, parent=None, app=None):

        QWidget.__init__(self, parent)
  
        cp.fmwtabs = self

        self.box_layout = QHBoxLayout()

        start_tab_name = cp.fmwtab_tab_name.value()
        self.gui_win = None
        self.make_tab_bar(start_tab_name)
        self.gui_selector(start_tab_name)

        self.box = QVBoxLayout(self)
        self.box.addWidget(self.tab_bar)
        self.box.addLayout(self.box_layout)

        self.setLayout(self.box)

        self.show_tool_tips()
        self.set_style()


    def show_tool_tips(self):
        self.setToolTip('Main tab window')


    def set_style(self):
        from psana.graphqt.Styles import style
        from psana.graphqt.QWIcons import icon
        icon.set_icons()
        self.setWindowIcon(icon.icon_monitor)
        self.setStyleSheet(style.styleBkgd)
        self.layout().setContentsMargins(0,0,0,0)
 

    def make_tab_bar(self, start_tab_name):
        self.tab_bar = QTabBar()

        for tab_name in self.tab_names:
            tab_ind = self.tab_bar.addTab(tab_name)
            self.tab_bar.setTabTextColor(tab_ind, QColor('blue')) #gray, red, grayblue

        #self.tab_bar.setTabsClosable(True)
        #self.tab_bar.setMovable(True)
        self.tab_bar.setShape(QTabBar.RoundedNorth)

        tab_index = self.tab_names.index(start_tab_name)
        self.tab_bar.setCurrentIndex(tab_index)
        logger.debug('make_tab_bar - set tab index: %d'%tab_index)

        self.tab_bar.currentChanged['int'].connect(self.on_tab_bar)
        self.tab_bar.tabCloseRequested.connect(self.on_tab_close_request)
        self.tab_bar.tabMoved[int,int].connect(self.on_tab_moved)


    def gui_selector(self, tab_name):

        if self.gui_win is not None:
            self.gui_win.close()
            del self.gui_win

        w_height = 200
        #if cp.cmwmain is not None: cp.cmwmain.wlog.setVisible(True)

        if tab_name == 'LCLS1->LCLS2':
#            from psana.graphqt.CMWDBMain import CMWDBMain
#            self.gui_win = CMWDBMain()
            self.gui_win = QTextEdit('Selected tab "%s"' % tab_name)
            w_height = 400

        elif tab_name == 'LCLS1':
            from psana.graphqt.FMW1Main import FMW1Main
            self.gui_win = FMW1Main()

        elif tab_name == 'LCLS2':
            self.gui_win = QTextEdit('Selected tab "%s"' % tab_name)

        else:
            self.gui_win = QTextEdit('Selected tab "%s"' % tab_name)

        #self.gui_win.setMinimumHeight(w_height)
        self.gui_win.setVisible(True)
        self.box_layout.addWidget(self.gui_win)


    def current_tab_index_and_name(self):
        tab_ind  = self.tab_bar.currentIndex()
        tab_name = str(self.tab_bar.tabText(tab_ind))
        return tab_ind, tab_name


    def on_tab_bar(self, ind):
        tab_ind, tab_name = self.current_tab_index_and_name()
        logger.info('Selected tab "%s"' % tab_name)
        cp.fmwtab_tab_name.setValue(tab_name)
        self.gui_selector(tab_name)


    def on_tab_close_request(self, ind):
        logger.debug('on_tab_close_request ind:%d' % ind)
        #self.tab_bar.removeTab(ind)
        #logger.debug('on_tab_close_request tab index:%d' % (itab))


    def on_tab_moved(self, inew, iold):
        logger.debug('on_tab_close_request tab index begin:%d -> end:%d' % (iold, inew))

 
#    def resizeEvent(self, e):
#        self.frame.setGeometry(self.rect())
#        logger.debug('resizeEvent: %s' % str(self.size()))


#    def moveEvent(self, e):
#        logger.debug('moveEvent - pos:' + str(self.position))       
#        self.position = self.mapToGlobal(self.pos())
#        self.position = self.pos()


    def closeEvent(self, e):
        logger.debug('closeEvent')

        #try   : self.gui_win.close()
        #except: pass

        #try   : del self.gui_win
        #except: pass

        if self.gui_win is not None:
            self.gui_win.close()

        QWidget.closeEvent(self, e)

        cp.fmwtabs = None


    def onExit(self):
        logger.debug('onExit')
        self.close()


    def set_tabs_visible(self, is_visible):
        logger.debug('set_tabs_visible: is_visible %s' % is_visible)
        self.tab_bar.setVisible(is_visible)


    def tab_bar_is_visible(self):
        return self.tab_bar.isVisible()


    def view_hide_tabs(self):
        self.tab_bar.setVisible(not self.tab_bar.isVisible())


    def key_usage(self):
        return 'Keys:'\
               '\n  V - view/hide tabs'\
               '\n'

    if __name__ == "__main__":
      def keyPressEvent(self, e):
        #logger.debug('keyPressEvent, key=%s' % e.key())       
        if   e.key() == Qt.Key_Escape:
            self.close()

        elif e.key() == Qt.Key_V: 
            self.view_hide_tabs()

        else:
            logger.debug(self.key_usage())


if __name__ == "__main__":

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='[%(levelname).1s] L:%(lineno)03d %(name)s: %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    w = FMWTabs()
    w.setGeometry(10, 25, 400, 600)
    w.setWindowTitle('FMWTabs')
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

# EOF
