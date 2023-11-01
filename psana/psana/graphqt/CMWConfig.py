
"""Class :py:class:`CMWConfig` is a QWidget for configuration parameters
==============================================================================

Usage ::
    # Test: python lcls2/psana/psana/graphqt/CMWConfig.py

    # Import
    from psana.graphqt.CMConfigParameters import

    # See test at the EOF

See:
  - :class:`CMWMain`
  - :class:`CMWConfig`
  - `on github <https://github.com/slac-lcls/lcls2>`_.

Created on 2017-04-05 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

from psana.graphqt.CMConfigParameters import cp
from psana.graphqt.CMWConfigPars import CMWConfigPars
from psana.graphqt.CMWConfigFile import CMWConfigFile
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QTabBar
from PyQt5.QtGui import QColor#, QFont
from PyQt5.QtCore import Qt
from psana.graphqt.Styles import style


class CMWConfig(QWidget):
    """CMWConfig is a QWidget with tabs for configuration management"""

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.but_close  = QPushButton('&Close')
        self.but_save   = QPushButton('&Save')
        self.but_show   = QPushButton('Show &Image')

        self.hboxW = QHBoxLayout()
        self.hboxB = QHBoxLayout()
        self.hboxB.addStretch(1)
        self.hboxB.addWidget(self.but_close)
        self.hboxB.addWidget(self.but_save)
        self.hboxB.addWidget(self.but_show )

        self.tab_names = ['Parameters'
                         ,'Configuration File'
                         ]

        self.gui_win = None

        self.make_tab_bar()
        self.gui_selector(cp.current_config_tab.value())

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.tab_bar)
        self.vbox.addLayout(self.hboxW)
        self.vbox.addStretch(1)
        self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)

        self.but_close.clicked.connect(self.on_close)
        self.but_save .clicked.connect(self.on_save)
        self.but_show .clicked.connect(self.on_show)

        self.set_tool_tips()
        self.set_style()


    def set_tool_tips(self):
        self.but_close.setToolTip('Close this window.')
        self.but_save.setToolTip('Save all current configuration parameters.')
        self.but_show.setToolTip('Show ...')


    def set_style(self):
        self.          setStyleSheet(style.styleBkgd)
        self.but_close.setStyleSheet(style.styleButton)
        self.but_save .setStyleSheet(style.styleButton)
        self.but_show .setStyleSheet(style.styleButton)

        self.setMinimumSize(600,500)

        is_visible = False
        self.but_close.setVisible(is_visible)
        self.but_save .setVisible(is_visible)
        self.but_show .setVisible(is_visible)


    def make_tab_bar(self):
        self.tab_bar = QTabBar()

        self.ind_tab_0 = self.tab_bar.addTab(self.tab_names[0])
        self.ind_tab_1 = self.tab_bar.addTab(self.tab_names[1])

        self.tab_bar.setTabTextColor(self.ind_tab_0, QColor('magenta'))
        self.tab_bar.setTabTextColor(self.ind_tab_1, QColor('magenta'))
        self.tab_bar.setShape(QTabBar.RoundedNorth)

        tab_index = self.tab_names.index(cp.current_config_tab.value())

        self.tab_bar.setCurrentIndex(tab_index)

        logger.debug(' make_tab_bar - set tab: ' + cp.current_config_tab.value())

        #self.connect(self.tab_bar, QtCore.SIGNAL('currentChanged(int)'), self.on_tab_bar)
        self.tab_bar.currentChanged[int].connect(self.on_tab_bar)


    def gui_selector(self, tab_name):

        if self.gui_win is not None:
            self.gui_win.close()
            del self.gui_win

        if tab_name == self.tab_names[0]:
            self.gui_win = CMWConfigPars(self)

        elif tab_name == self.tab_names[1]:
            self.gui_win = CMWConfigFile(self)

        else:
            logger.warning('Unknown tab name "%s"' % tab_name)

        self.hboxW.addWidget(self.gui_win)
        #self.hboxW.addStretch(1)
        self.gui_win.setVisible(True)


    def current_tab_index_and_name(self):
        tab_ind  = self.tab_bar.currentIndex()
        tab_name = str(self.tab_bar.tabText(tab_ind))
        return tab_ind, tab_name


    def on_tab_bar(self):
        tab_ind, tab_name = self.current_tab_index_and_name()
        logger.info('Selected tab "%s"' % tab_name)
        cp.current_config_tab.setValue(tab_name)
        self.gui_selector(tab_name)


    def set_parent(self,parent):
        self.parent = parent


    def closeEvent(self, e):
        logger.debug('closeEvent')
        self.tab_bar.close()
        if self.gui_win is not None: self.gui_win.close()
        QWidget.close(self)


    def on_close(self):
        logger.debug('on_close')
        self.close()


    def on_save(self):
        logger.debug('on_save')
        cp.saveParametersInFile( cp.fname_cp.value() )


    def on_show(self):
        logger.debug('on_show - is not implemented yet...')


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CMWConfig()
    w.setGeometry(1, 1, 600, 200)
    w.setWindowTitle('Configuration manager')
    w.show()
    app.exec_()

    cp.printParameters()
    cp.saveParametersInFile()

    del w
    del app

# EOF
