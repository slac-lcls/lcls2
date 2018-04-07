#------------------------------
"""Class :py:class:`CMWDBButtons` is a QWidget for configuration parameters
==============================================================================

Usage ::
    # Test: python lcls2/psana/psana/graphqt/CMWDBButtons.py

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

#import os

from PyQt5.QtWidgets import QWidget, QLabel, QComboBox, QPushButton, QHBoxLayout, QLineEdit
from PyQt5.QtCore import QSize # Qt, QEvent, QPoint, 
# QLineEdit, QCheckBox, QFileDialog
# QTextEdit, QComboBox, QVBoxLayout, QGridLayout
#from PyQt5.QtGui import QIntValidator, QDoubleValidator# QRegExpValidator

from psana.graphqt.CMConfigParameters import cp
from psana.pyalgos.generic.Logger import logger
from psana.graphqt.Styles import style
from psana.graphqt.QWIcons import icon

#------------------------------

class CMWDBButtons(QWidget) :
    """QWidget for managements of configuration parameters"""

    def __init__(self, parent=None) :
        QWidget.__init__(self, parent)
        self._name = 'CMWDBButtons'

        self.lab_host = QLabel('Host:')
        self.lab_port = QLabel('Port:')
        self.lab_db_filter = QLabel('DB filter:')

        self.cmb_host = QComboBox(self)        
        self.cmb_host.addItems(cp.list_of_hosts)
        self.cmb_host.setCurrentIndex(cp.list_of_hosts.index(cp.cdb_host.value()))

        self.cmb_port = QComboBox(self)        
        self.cmb_port.addItems(cp.list_of_str_ports)
        self.cmb_port.setCurrentIndex(cp.list_of_str_ports.index(str(cp.cdb_port.value())))

        self.but_exp_col  = QPushButton('Expand')
        self.but_tabs     = QPushButton('Hide tabs')
        self.but_test     = QPushButton('Test')
        self.list_of_buts = (
             self.but_exp_col,
             self.but_tabs,
             self.but_test
        )

        self.edi_db_filter = QLineEdit(cp.cdb_filter.value())        
        #self.edi_db_filter.setReadOnly(True)  

#        #self.tit_dir_work = QtGui.QLabel('Parameters:')
#
#        self.edi_dir_work = QLineEdit(cp.dir_work.value())        
#        self.but_dir_work = QPushButton('Dir work:')
#        self.edi_dir_work.setReadOnly(True)  
#
#        self.edi_dir_results = QLineEdit(cp.dir_results.value())        
#        self.but_dir_results = PushButton('Dir results:')
#        self.edi_dir_results.setReadOnly( True )  
#
#        self.lab_fname_prefix = QLabel('File prefix:')
#        self.edi_fname_prefix = QLineEdit(cp.fname_prefix.value())        
#
#        self.lab_bat_queue  = QLabel('Queue:') 
#        self.box_bat_queue  = QComboBox(self) 
#        self.box_bat_queue.addItems(cp.list_of_queues)
#        self.box_bat_queue.setCurrentIndex(cp.list_of_queues.index(cp.bat_queue.value()))
#
#        self.lab_dark_start = QLabel('Event start:') 

#        self.but_show_vers  = QPushButton('Soft Vers')
#        self.edi_dark_start = QLineEdit(str(cp.bat_dark_start.value()))#

#        self.edi_dark_start .setValidator(QIntValidator(0,9999999,self))
#        self.edi_rms_thr_min.setValidator(QDoubleValidator(0,65000,3,self))
#        #self.edi_events.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[0-9]\\d{0,3}|end$"),self))
#
#        self.cbx_deploy_hotpix = QCheckBox('Deploy pixel_status')
#        self.cbx_deploy_hotpix.setChecked(cp.dark_deploy_hotpix.value())

#        self.grid = QGridLayout()
#        self.grid.addWidget(self.lab_host, 0, 0)
#        self.grid.addWidget(self.cmb_host, 0, 1, 1, 1)
#        self.grid.addWidget(self.lab_port, 2, 0)
#        self.grid.addWidget(self.cmb_port, 2, 1, 1, 1)
#        self.setLayout(self.grid)

        self.hbox = QHBoxLayout() 
        self.hbox.addWidget(self.lab_db_filter)
        self.hbox.addWidget(self.edi_db_filter)
        self.hbox.addWidget(self.but_exp_col)
        self.hbox.addWidget(self.lab_host)
        self.hbox.addWidget(self.cmb_host)
        self.hbox.addWidget(self.lab_port)
        self.hbox.addWidget(self.cmb_port)
        self.hbox.addStretch(1) 
        self.hbox.addSpacing(50)
        #self.hbox.addStrut(50)
        #self.hbox.addSpacerItem(QSpacerItem)
        self.hbox.addWidget(self.but_test)
        self.hbox.addWidget(self.but_tabs)
        #self.hbox.addLayout(self.grid)
        self.setLayout(self.hbox)

        #self.edi_.editingFinished.connect(self.on_edit_finished)
        #self.cbx_host.stateChanged[int].connect(self.on_cbx_host_changed)
        #self.cbx_host.stateChanged[int].connect(self.on_cbx_host_changed)
        self.cmb_host.currentIndexChanged[int].connect(self.on_cmb_host_changed)
        self.cmb_port.currentIndexChanged[int].connect(self.on_cmb_port_changed)

        self.edi_db_filter.editingFinished.connect(self.on_edi_db_filter_finished)

        self.but_exp_col .clicked.connect(self.on_but_clicked)
        self.but_tabs    .clicked.connect(self.on_but_clicked)

        self.but_test.clicked .connect(self.on_but_clicked)
        self.but_test.released.connect(self.on_but_released)
        self.but_test.pressed .connect(self.on_but_pressed)
        self.but_test.toggled .connect(self.on_but_toggled)


        self.set_tool_tips()
        self.set_style()


    def set_tool_tips(self):
        self.         setToolTip('Control buttons')
        self.cmb_host.setToolTip('Select DB host')
        self.cmb_port.setToolTip('Select DB port')
        self.but_tabs.setToolTip('Show/hide tabs')
        self.but_exp_col.setToolTip('Expand/collapse DB tree')


    def set_style(self):
        #self.         setStyleSheet(style.styleBkgd)
        self.lab_db_filter.setStyleSheet(style.styleLabel)
        self.lab_host.setStyleSheet(style.styleLabel)
        self.lab_port.setStyleSheet(style.styleLabel)

        icon.set_icons()

        self.but_exp_col .setIcon(icon.icon_folder_open)

        #self.but_exp_col.setStyleSheet(style.styleButton)

        self.but_tabs.setStyleSheet(style.styleButtonGood)
        #self.but_tabs.setFixedWidth(40)

        # TEST BUTTON
        self.but_test.setIcon(icon.icon_undo)
        self.but_test.setCheckable(True)
        #self.but_test.setChecked(True)
        #self.but_test.setFlat(True)
        #self.but_test.setVisible(False)
        #self.but_test.setFixedWidth(100) 
        #self.but_test.setFixedHeight(100) 
        #self.but_test.setIconSize(QSize(96,96)) 

        self.setContentsMargins(-9,-9,-9,-9)

        size_hint = self.minimumSizeHint()
        self.setMinimumWidth(size_hint.width())
        self.setFixedHeight(size_hint.height())
        #self.setMinimumSize(433,46)

#        self.tit_dir_work     .setStyleSheet(style.styleTitle)
#        self.edi_dir_work     .setStyleSheet(style.styleEditInfo)       
#        self.lab_fname_prefix .setStyleSheet(style.styleLabel)
#        self.edi_fname_prefix .setStyleSheet(style.styleEdit)
#        self.lab_pix_status   .setStyleSheet(style.styleTitleBold)
#
#        self.edi_dir_work    .setAlignment(QtCore.Qt.AlignRight)
#        self.lab_pix_status  .setAlignment(QtCore.Qt.AlignLeft)
#
#        self.edi_dir_work    .setMinimumWidth(300)
#        self.but_dir_work    .setFixedWidth(80)
#
#    def set_parent(self,parent) :
#        self.parent = parent

    def resizeEvent(self, e):
        logger.debug('resizeEvent size: %s' % str(e.size()), self._name) 


    def moveEvent(self, e):
        logger.debug('moveEvent pos: %s' % str(e.pos()), self._name) 
#        #cp.posGUIMain = (self.pos().x(),self.pos().y())

    def closeEvent(self, event):
        logger.debug('closeEvent', self._name)
        #try    : del cp.guiworkresdirs # CMWDBButtons
        #except : pass # silently ignore


#    def onButLsfStatus(self):
#        queue = cp.bat_queue.value()
#        farm = cp.dict_of_queue_farm[queue]
#        msg, status = gu.msg_and_status_of_lsf(farm)
#        msgi = '\nLSF status for queue %s on farm %s: \n%s\nLSF status for %s is %s' % \
#               (queue, farm, msg, queue, {False:'bad',True:'good'}[status])
#        logger.info(msgi, self._name)
#
#        cmd, msg = gu.text_status_of_queues(cp.list_of_queues)
#        msgq = '\nStatus of queues for command: %s \n%s' % (cmd, msg)       
#        logger.info(msgq, self._name)

#    def onEditPrefix(self):
#        logger.debug('onEditPrefix', self._name)
#        cp.fname_prefix.setValue(str(self.edi_fname_prefix.displayText()))
#        logger.info('Set file name common prefix: ' + str( cp.fname_prefix.value()), self._name)
#
#    def onEdiDarkStart(self):
#        str_value = str(self.edi_dark_start.displayText())
#        cp.bat_dark_start.setValue(int(str_value))      
#        logger.info('Set start event for dark run: %s' % str_value, self._name)

#    def on_cbx(self, par, cbx):
#        #if cbx.hasFocus() :
#        par.setValue(cbx.isChecked())
#        msg = 'check box %s is set to: %s' % (cbx.text(), str(par.value()))
#        logger.info(msg, self._name)
#
#    def on_cbx_host_changed(self, i):
#        self.on_cbx(cp.cdb_host, self.cbx_host)
#
#    def on_cbx_port_changed(self, i):
#        self.on_cbx(cp.cdb_port, self.cbx_port)
#-----------------------------

    def on_edi_db_filter_finished(self):
        wtree = cp.cmwdbtree
        if wtree is None : return
        edi = self.edi_db_filter
        txt = edi.displayText()
        logger.debug('on_edi_db_filter_finished txt: "%s"' % txt)
        wtree.fill_tree_model(pattern=txt)

#-----------------------------

#-----------------------------

    def on_cmb_host_changed(self):
        selected = self.cmb_host.currentText()
        cp.cdb_host.setValue(selected) 
        logger.info('on_cmb_host_changed - selected: %s' % selected, self._name)

    def on_cmb_port_changed(self):
        selected = self.cmb_port.currentText()
        cp.cdb_port.setValue(int(selected)) 
        logger.info('on_cmb_port_changed - selected: %s' % selected, self._name)

#-----------------------------

    def view_hide_tabs(self):
        wtabs = cp.cmwmaintabs
        but = self.but_tabs
        if wtabs is None : return
        is_visible = wtabs.tab_bar_is_visible()
        but.setText('Show tabs' if is_visible else 'Hide tabs')
        wtabs.set_tabs_visible(not is_visible)

#-----------------------------

    def expand_collapse_dbtree(self):
        if cp.cmwdbmain is None : return
        wtree = cp.cmwdbmain.wtree
        but = self.but_exp_col
        if but.text() == 'Expand' :
            wtree.process_expand()
            self.but_exp_col.setIcon(icon.icon_folder_closed)
            but.setText('Collapse')
        else :
            wtree.process_collapse()
            self.but_exp_col.setIcon(icon.icon_folder_open)
            but.setText('Expand')

#-----------------------------

    def on_but_clicked(self):
        for but in self.list_of_buts :
            if but.hasFocus() : break
        logger.debug('on_but_clicked "%s"' % but.text())
        if but == self.but_exp_col  : self.expand_collapse_dbtree()
        if but == self.but_tabs     : self.view_hide_tabs()


    def on_but_pressed(self):
        for but in self.list_of_buts :
            if but.hasFocus() : break
        logger.debug('on_but_pressed "%s"' % but.text())


    def on_but_released(self):
        for but in self.list_of_buts :
            if but.hasFocus() : break
        logger.debug('on_but_released "%s"' % but.text())


    def on_but_toggled(self):
        for but in self.list_of_buts :
            if but.hasFocus() : break
        logger.debug('on_but_pressed "%s"' % but.text())

#-----------------------------

if __name__ == "__main__" :
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    logger.setPrintBits(0o177777)
    w = CMWDBButtons()
    #w.setGeometry(200, 400, 500, 200)
    w.setWindowTitle('Config Parameters')
    w.show()
    app.exec_()
    del w
    del app

#-----------------------------
