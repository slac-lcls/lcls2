#------------------------------
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
#------------------------------

from psana.graphqt.CMConfigParameters import cp
from psana.graphqt.CMWConfigPars import CMWConfigPars
from psana.graphqt.CMWConfigFile import CMWConfigFile
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QTabBar
from PyQt5.QtGui import QColor#, QFont
from PyQt5.QtCore import Qt
from psana.pyalgos.generic.Logger import logger
from psana.graphqt.Styles import style

#------------------------------

class CMWConfig(QWidget) :
    """CMWConfig is a QWidget with tabs for configuration management"""

    def __init__(self, parent=None) :
        QWidget.__init__(self, parent)
        self._name = 'CMWConfig'

        #self.lab_title  = QLabel     ('Configuration settings')
        #self.lab_status = QLabel     ('Status: ')
        self.but_close  = QPushButton('&Close') 
        self.but_save   = QPushButton('&Save') 
        self.but_show   = QPushButton('Show &Image') 

        self.hboxW = QHBoxLayout()
        self.hboxB = QHBoxLayout()
        #self.hboxB.addWidget(self.lab_status)
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
        #msg = 'Edit field'
        self.but_close.setToolTip('Close this window.')
        self.but_save.setToolTip('Save all current configuration parameters.')
        self.but_show.setToolTip('Show ...')


    def set_style(self):
        self.          setStyleSheet(style.styleBkgd)
        self.but_close.setStyleSheet(style.styleButton)
        self.but_save .setStyleSheet(style.styleButton)
        self.but_show .setStyleSheet(style.styleButton)

        self.setMinimumSize(600,360)

        is_visible = False
        #self.lab_status.setVisible(False)
        self.but_close.setVisible(is_visible)
        self.but_save .setVisible(is_visible)
        self.but_show .setVisible(is_visible)


    def make_tab_bar(self) :
        self.tab_bar = QTabBar()

        #Uses self.tab_names
        self.ind_tab_0 = self.tab_bar.addTab(self.tab_names[0])
        self.ind_tab_1 = self.tab_bar.addTab(self.tab_names[1])

        self.tab_bar.setTabTextColor(self.ind_tab_0, QColor('magenta'))
        self.tab_bar.setTabTextColor(self.ind_tab_1, QColor('magenta'))
        self.tab_bar.setShape(QTabBar.RoundedNorth)

        #self.tab_bar.setTabsClosable(True)
        #self.tab_bar.setMovable(True)

        #self.tab_bar.setTabEnabled(1, False)
        #self.tab_bar.setTabEnabled(2, False)

        tab_index = self.tab_names.index(cp.current_config_tab.value())
        #try :
        #    tab_index = self.tab_names.index(cp.current_config_tab.value())
        #except :
        #    tab_index = 1
        #    cp.current_config_tab.setValue(self.tab_names[tab_index])

        self.tab_bar.setCurrentIndex(tab_index)

        logger.debug(' make_tab_bar - set tab: ' + cp.current_config_tab.value(), self._name)

        #self.connect(self.tab_bar, QtCore.SIGNAL('currentChanged(int)'), self.on_tab_bar)
        self.tab_bar.currentChanged[int].connect(self.on_tab_bar)


    def gui_selector(self, tab_name):

        #try    : self.gui_win.close()
        #except : pass

        #try    : del self.gui_win
        #except : pass

        if self.gui_win is not None : 
            self.gui_win.close()
            del self.gui_win

        w_height = 120

        if tab_name == self.tab_names[0] :
            self.gui_win = CMWConfigPars(self)

        elif tab_name == self.tab_names[1] :
            self.gui_win = CMWConfigFile(self)
            w_height = 170

        else :
            logger.warning('Unknown tab name "%s"' % tab_name, self._name)

        #self.set_status(0, 'Set configuration file')
        self.gui_win.setFixedHeight(w_height)
        self.hboxW.addWidget(self.gui_win)
        self.gui_win.setVisible(True)


    def current_tab_index_and_name(self):
        tab_ind  = self.tab_bar.currentIndex()
        tab_name = str(self.tab_bar.tabText(tab_ind))
        return tab_ind, tab_name


    def on_tab_bar(self):
        tab_ind, tab_name = self.current_tab_index_and_name()
        logger.debug('tab_name: %s' % tab_name, self._name)
        cp.current_config_tab.setValue(tab_name)
        logger.info('Selected tab index: %d name: %s' % (tab_ind, tab_name), self._name)
        self.gui_selector(tab_name)


    def set_parent(self,parent) :
        self.parent = parent


    #def resizeEvent(self, e):
        #logger.debug('resizeEvent', self._name) 
        #print self._name + ' config: self.size():', self.size()
        #self.setMinimumSize( self.size().width(), self.size().height()-40 )
        #pass


    #def moveEvent(self, e):
        #logger.debug('moveEvent', self._name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent: new pos:' + str(self.position), self._name)
        #pass


    def closeEvent(self, e):
        logger.debug('closeEvent', self._name)

        self.tab_bar.close()        
        if self.gui_win is not None : self.gui_win.close()

        QWidget.close(self)


    def on_close(self):
        logger.debug('on_close', self._name)
        self.close()


    def on_save(self):
        logger.debug('on_save', self._name)
        cp.saveParametersInFile( cp.fname_cp.value() )


    def on_show(self):
        logger.debug('on_show - is not implemented yet...', self._name)


#    def set_status(self, status_index=0, msg=''):
#        list_of_states = ['Good','Warning','Alarm']
#        if status_index == 0 : self.lab_status.setStyleSheet(style.styleStatusGood)
#        if status_index == 1 : self.lab_status.setStyleSheet(style.styleStatusWarning)
#        if status_index == 2 : self.lab_status.setStyleSheet(style.styleStatusAlarm)
#        #self.lab_status.setText('Status: ' + list_of_states[status_index] + msg)
#        self.lab_status.setText(msg)

#-----------------------------

if __name__ == "__main__" :
    from PyQt5.QtWidgets import QApplication
    import sys
    logger.setPrintBits(0o177777)
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

#-----------------------------
