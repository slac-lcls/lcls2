
"""Class :py:class:`CMWMainTabs` is a QWidget for tabs and switching window
==============================================================================

Usage ::
    # Test: python lcls2/psana/psana/graphqt/CMWMainTabs.py

Created on 2017-02-18 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-26 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QTabBar, QTextEdit
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
from psana.graphqt.CMConfigParameters import cp


class CMWMainTabs(QWidget):
    """GUI for tabs and associated widgets
    """
    tab_names = ['CDB', 'Image', 'Text', 'Calib', 'HDF5', 'XTC', 't-converter', 'Configuration', 'Mask']

    tool_tips = [\
      'Calibration Data Base\nviewer/manager',\
      'Image Viewer for ndarrays',\
      'Text Browser for text-like calibration constants',\
      'File Manager for easy access calibration constants\nunder experimental calib directory',\
      'HDF5 file browser',\
      'XTC data viewer',\
      'Time converter between Date&Time <-> POSIX <-> LCLS2',\
      'Configuration manager for this app',\
      'Mask Editor for images',\
    ]


    def __init__(self, parent=None, app=None):

        QWidget.__init__(self, parent)

        cp.cmwmaintabs = self

        self.box_layout = QHBoxLayout()

        start_tab_name = cp.main_tab_name.value()
        self.gui_win = None
        self.make_tab_bar(start_tab_name)
        self.gui_selector(start_tab_name)

        self.box = QVBoxLayout(self)
        self.box.addWidget(self.tab_bar)
        self.box.addLayout(self.box_layout)

        self.setLayout(self.box)

        self.set_tool_tips()
        self.set_style()


    def set_tool_tips(self):
        for t,s in zip(self.tab_names, self.tool_tips):
          self.tab_bar.setTabToolTip(self.tab_names.index(t), s)


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
        if cp.cmwmain is not None: cp.cmwmain.wlog.setVisible(True)

        if tab_name == 'CDB':
            from psana.graphqt.CMWDBMain import CMWDBMain
            self.gui_win = CMWDBMain()
            w_height = 500

        elif tab_name == 'Configuration':
            from psana.graphqt.CMWConfig import CMWConfig
            self.gui_win = CMWConfig()
            w_height = 500

        elif tab_name == 't-converter':
            from psana.graphqt.QWDateTimeSec import QWDateTimeSec
            self.gui_win = QWDateTimeSec()
            self.gui_win.setMaximumWidth(400)
            w_height = 80

        elif tab_name == 'HDF5':
            from psana.graphqt.H5VMain import H5VMain
            #if cp.cmwmain is not None: cp.cmwmain.wlog.setVisible(False)
            self.gui_win = H5VMain()

        elif tab_name == 'Calib':
            from psana.graphqt.FMW1Main import FMW1Main
            self.gui_win = FMW1Main()
            #from psana.graphqt.FMWTabs import FMWTabs
            #self.gui_win = FMWTabs()

        elif tab_name == 'Image':
            from psana.graphqt.IVMain import IVMain
            #if cp.cmwmain is not None: cp.cmwmain.wlog.setVisible(False)
            self.gui_win = IVMain()

        elif tab_name == 'Text':
            from psana.graphqt.QWTextBrowser import QWTextBrowser
            self.gui_win = QWTextBrowser()

        elif tab_name == 'XTC':
            from psana.graphqt.DMQWMain import DMQWMain
            self.gui_win = DMQWMain()

        elif tab_name == 'Mask':
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
        cp.main_tab_name.setValue(tab_name)
        self.gui_selector(tab_name)


    def on_tab_close_request(self, ind):
        logger.debug('on_tab_close_request ind:%d' % ind)
        #self.tab_bar.removeTab(ind)


    def on_tab_moved(self, inew, iold):
        logger.debug('on_tab_close_request tab index begin:%d -> end:%d' % (iold, inew))


    def closeEvent(self, e):
        logger.debug('closeEvent')

        if self.gui_win is not None:
            self.gui_win.close()

        QWidget.closeEvent(self, e)

        cp.cmwmaintabs = None


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


    def view_data(self, data=None, fname=None):
        from psana.pyalgos.generic.NDArrUtils import info_ndarr, np
        cp.last_selected_data = data
        cp.last_selected_fname.setValue(fname)
        if data is None and fname is None:
            logger.debug('data and fname are None - do not switch to viewer')
        elif data is None:
            tabname = 'Text' if 'geometry' in fname or 'common_mode' in fname else 'Image'
            self.set_tab(tabname)
        elif isinstance(data, np.ndarray):
            logger.info(info_ndarr(data, 'switch to Image Viewer to view data:'))
            #logger.warning('TBD: IV NEEDS TO BE SET TO ACCEPT DATA DIRECTLY FROM cp.last_selected_data')
            self.set_tab(tabname='Image')
        elif isinstance(data, str):
            logger.info(info_ndarr(data, 'switch to Text Viewer to view data:'))
            self.set_tab(tabname='Text')
        elif isinstance(data, dict):
            from psana.pscalib.calib.MDBConvertUtils import info_dict
            logger.info('data is dict switch to Text Viewer to view data as info_dict')
            cp.last_selected_data = info_dict(data, offset='  ', s='') #str(cp.last_selected_data)
            self.set_tab(tabname='Text')
        else:
            logger.debug('data of the selected document is not numpy array - do not switch to Image Viewer')
            cp.last_selected_data = None


    def set_tab(self, tabname='Image'):
        logger.debug('switch tab to %s' % str(tabname))
        self.tab_bar.setCurrentIndex(self.tab_names.index(tabname))


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

    logging.basicConfig(format='[%(levelname).1s] L:%(lineno)03d %(name)s %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    w = CMWMainTabs()
    w.setGeometry(10, 25, 400, 600)
    w.setWindowTitle('CMWMainTabs')
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

# EOF
