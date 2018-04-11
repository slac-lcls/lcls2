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

from psana.graphqt.QWUtils import change_check_box_dict_in_popup_menu,\
     confirm_or_cancel_dialog_box, select_item_from_popup_menu

import psana.graphqt.CMDBUtils as dbu

#------------------------------

class CMWDBButtons(QWidget) :
    """QWidget for managements of configuration parameters"""

    def __init__(self, parent=None) :
        QWidget.__init__(self, parent)
        self._name = 'CMWDBButtons'

        self.lab_host = QLabel('Host:')
        self.lab_port = QLabel('Port:')
        self.lab_db_filter = QLabel('DB filter:')
        self.lab_docs = QLabel('Docs:')

        self.cmb_host = QComboBox(self)        
        self.cmb_host.addItems(cp.list_of_hosts)
        self.cmb_host.setCurrentIndex(cp.list_of_hosts.index(cp.cdb_host.value()))

        self.cmb_port = QComboBox(self)        
        self.cmb_port.addItems(cp.list_of_str_ports)
        self.cmb_port.setCurrentIndex(cp.list_of_str_ports.index(str(cp.cdb_port.value())))

        self.cmb_docw = QComboBox(self)        
        self.cmb_docw.addItems(cp.list_of_doc_widgets)
        self.cmb_docw.setCurrentIndex(cp.list_of_doc_widgets.index(str(cp.cdb_docw.value())))

        self.but_exp_col  = QPushButton('Expand')
        self.but_tabs     = QPushButton('Tabs %s' % cp.char_expand)
        self.but_test     = QPushButton('Test')
        self.but_buts     = QPushButton('Buts %s' % cp.char_expand)
        self.but_del      = QPushButton('Delete')
        self.but_add      = QPushButton('Add')
        self.but_docs     = QPushButton('Doc %s' % cp.char_expand)
        self.but_selm     = QPushButton('Mode %s' % cp.char_expand)

        self.list_of_buts = (
             self.but_exp_col,
             self.but_tabs,
             self.but_buts,
             self.but_del,
             self.but_add,
             self.but_docs,
             self.but_selm,
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
        self.hbox.addWidget(self.but_selm)
        self.hbox.addWidget(self.lab_host)
        self.hbox.addWidget(self.cmb_host)
        self.hbox.addWidget(self.lab_port)
        self.hbox.addWidget(self.cmb_port)
        self.hbox.addWidget(self.but_add)
        self.hbox.addWidget(self.but_del)
        self.hbox.addStretch(1) 
        self.hbox.addSpacing(50)
        #self.hbox.addStrut(50)
        #self.hbox.addSpacerItem(QSpacerItem)
        self.hbox.addWidget(self.but_test)
        self.hbox.addWidget(self.lab_docs)
        self.hbox.addWidget(self.cmb_docw)
        self.hbox.addWidget(self.but_docs)
        self.hbox.addWidget(self.but_buts)
        self.hbox.addWidget(self.but_tabs)
        #self.hbox.addLayout(self.grid)
        self.setLayout(self.hbox)

        #self.edi_.editingFinished.connect(self.on_edit_finished)
        #self.cbx_host.stateChanged[int].connect(self.on_cbx_host_changed)
        #self.cbx_host.stateChanged[int].connect(self.on_cbx_host_changed)
        self.cmb_host.currentIndexChanged[int].connect(self.on_cmb_host_changed)
        self.cmb_port.currentIndexChanged[int].connect(self.on_cmb_port_changed)
        self.cmb_docw.currentIndexChanged[int].connect(self.on_cmb_docw_changed)

        self.edi_db_filter.editingFinished.connect(self.on_edi_db_filter_finished)

        self.but_exp_col.clicked.connect(self.on_but_clicked)
        self.but_tabs   .clicked.connect(self.on_but_clicked)
        self.but_buts   .clicked.connect(self.on_but_clicked)
        self.but_add    .clicked.connect(self.on_but_clicked)
        self.but_del    .clicked.connect(self.on_but_clicked)
        self.but_docs   .clicked.connect(self.on_but_clicked)
        self.but_selm   .clicked.connect(self.on_but_clicked)
 
        self.but_test.clicked .connect(self.on_but_clicked)
        self.but_test.released.connect(self.on_but_released)
        self.but_test.pressed .connect(self.on_but_pressed)
        self.but_test.toggled .connect(self.on_but_toggled)

        self.set_tool_tips()
        self.set_style()
        self.set_buttons_visiable()


    def set_tool_tips(self):
        self.         setToolTip('Control buttons')
        self.cmb_host.setToolTip('Select DB host')
        self.cmb_port.setToolTip('Select DB port')
        self.but_tabs.setToolTip('Show/hide tabs')
        self.but_buts.setToolTip('Show/hide buttons')
        self.but_exp_col.setToolTip('Expand/collapse DB tree')
        self.edi_db_filter.setToolTip('Enter text in DB names\nto filter tree')
        self.but_docs.setToolTip('Select style of documents presentation')
        self.cmb_docw.setToolTip('Select style of documents presentation')


    def set_style(self):
        #self.         setStyleSheet(style.styleBkgd)
        self.lab_db_filter.setStyleSheet(style.styleLabel)
        self.lab_host.setStyleSheet(style.styleLabel)
        self.lab_port.setStyleSheet(style.styleLabel)
        self.lab_docs.setStyleSheet(style.styleLabel)

        icon.set_icons()

        self.but_exp_col .setIcon(icon.icon_folder_open)
        self.but_add     .setIcon(icon.icon_plus)
        self.but_del     .setIcon(icon.icon_minus)
        self.but_docs    .setIcon(icon.icon_table)

        #self.lab_docs    .setPixmap(icon.icon_table)

        #self.but_exp_col.setStyleSheet(style.styleButton)

        self.but_tabs.setStyleSheet(style.styleButtonGood)
        self.but_buts.setStyleSheet(style.styleButton)
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

        self.but_tabs.setFixedWidth(55)
        self.but_buts.setFixedWidth(55)
        self.but_add .setFixedWidth(60)
        self.but_del .setFixedWidth(60)
        self.but_docs.setFixedWidth(60)
        self.but_selm.setFixedWidth(80)
        self.cmb_docw.setFixedWidth(60)

        self.edi_db_filter.setFixedWidth(80)

#        self.tit_dir_work     .setStyleSheet(style.styleTitle)
#        self.edi_dir_work     .setStyleSheet(style.styleEditInfo)       
#        self.lab_fname_prefix .setStyleSheet(style.styleLabel)
#        self.edi_fname_prefix .setStyleSheet(style.styleEdit)
#        self.lab_pix_status   .setStyleSheet(style.styleTitleBold)
#
#        self.edi_dir_work    .setAlignment(QtCore.Qt.AlignRight)
#        self.lab_pix_status  .setAlignment(QtCore.Qt.AlignLeft)
#
#    def set_parent(self,parent) :
#        self.parent = parent

    #def resizeEvent(self, e):
    #    logger.debug('resizeEvent size: %s' % str(e.size()), self._name) 


    #def moveEvent(self, e):
    #    logger.debug('moveEvent pos: %s' % str(e.pos()), self._name) 
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

    def buttons_dict(self):
        r = cp.cdb_buttons.value()
        return {'DB filter'  : r & 1,\
                'Expand'     : r & 2,\
                'Host & port': r & 4,\
                'Add'        : r & 8,\
                'Delete'     : r & 16,\
                'Docs'       : r & 32,\
                'Docw'       : r & 64,\
                'Tabs'       : r & 128,\
                'Test'       : r & 256,\
                'Selection'  : r & 512,\
                }


    def set_buttons_config_bitword(self, d):
        w = 0
        if d['DB filter']  : w |= 1
        if d['Expand']     : w |= 2
        if d['Host & port']: w |= 4
        if d['Add']        : w |= 8
        if d['Delete']     : w |= 16
        if d['Docs']       : w |= 32
        if d['Docw']       : w |= 64
        if d['Tabs']       : w |= 128
        if d['Test']       : w |= 256
        if d['Selection']  : w |= 512
        #if  : w |= 0
        cp.cdb_buttons.setValue(w)


    def set_buttons_visiable(self, dic_buts=None):
        d = self.buttons_dict() if dic_buts is None else dic_buts
        self.set_db_filter_visible (d['DB filter'])
        self.set_host_port_visible (d['Host & port'])
        self.but_exp_col.setVisible(d['Expand'])
        self.but_add.setVisible    (d['Add'])
        self.but_del.setVisible    (d['Delete'])
        self.cmb_docw.setVisible   (d['Docw'])
        self.lab_docs.setVisible   (d['Docw'])
        self.but_docs.setVisible   (d['Docs'])
        self.but_tabs.setVisible   (d['Tabs'])
        self.but_test.setVisible   (d['Test'])
        self.but_selm.setVisible   (d['Selection'])
        #self.set_tabs_visible     (d['Tabs'])      


    def select_visible_buttons(self):
        logger.debug('select_visible_buttons', self._name)
        d = self.buttons_dict()
        resp = change_check_box_dict_in_popup_menu(d, 'Select buttons')
        print('XXX:select_visible_buttons resp:', resp)
        if resp==1 :
            self.set_buttons_visiable(d)
            self.set_buttons_config_bitword(d)


    def select_doc_widget(self):
        logger.debug('select_doc_widget', self._name)
        resp = select_item_from_popup_menu(cp.list_of_doc_widgets)
        print('XXX: select_doc_widget resp:', resp)
        if resp is None : return
        cp.cdb_docw.setValue(resp)
        cp.cmwdbmain.wdocs.gui_selector()

#-----------------------------
#-----------------------------

    def set_host_port_visible(self, is_visible=True):
        self.lab_host.setVisible(is_visible)
        self.lab_port.setVisible(is_visible)
        self.cmb_host.setVisible(is_visible)
        self.cmb_port.setVisible(is_visible)


    def set_db_filter_visible(self, is_visible=True):
        self.lab_db_filter.setVisible(is_visible)
        self.edi_db_filter.setVisible(is_visible)


    def set_tabs_visible(self, is_visible=True):
        wtabs = cp.cmwmaintabs
        wtabs.set_tabs_visible(is_visible)

#-----------------------------

    def view_hide_tabs(self):
        wtabs = cp.cmwmaintabs
        but = self.but_tabs
        if wtabs is None : return
        is_visible = wtabs.tab_bar_is_visible()
        but.setText('Tabs %s'%cp.char_shrink if is_visible else 'Tabs %s'%cp.char_expand)
        wtabs.set_tabs_visible(not is_visible)

#-----------------------------

    def on_cmb_host_changed(self):
        selected = self.cmb_host.currentText()
        cp.cdb_host.setValue(selected) 
        logger.info('on_cmb_host_changed - selected: %s' % selected, self._name)

    def on_cmb_port_changed(self):
        selected = self.cmb_port.currentText()
        cp.cdb_port.setValue(int(selected)) 
        logger.info('on_cmb_port_changed - selected: %s' % selected, self._name)

    def on_cmb_docw_changed(self):
        selected = self.cmb_docw.currentText()
        cp.cdb_docw.setValue(selected)
        logger.info('on_cmb_docw_changed - selected: %s' % selected, self._name)
        cp.cmwdbmain.wdocs.gui_selector()

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

    def delete_selected_items(self):
        wtree = cp.cmwdbtree
        if wtree is None :
            logger.warning('delete_selected_items - tree object does not exist?', self._name)
            return
        # dict of pairs {<item-name> : <item-parent-name>}
        # where <item-parent-name> is None for DB item or DB name for collection item.
        dic_name_parent = dict([(item.text(), None if item.parent() is None else item.parent().text())\
                          for item in wtree.selected_items()])

        if not len(dic_name_parent) :
            logger.warning('delete_selected_items - selected nothing - nothing to delete...', self._name)
            return

        for k,v in dic_name_parent.items() :
            print('  %s : %s' % (k.ljust(20),v))

        # Define del_mode = 'collections' or 'DBs' if at least one DB is selected
        del_mode = 'collections'
        for v in dic_name_parent.values() :
            if v is None : 
                del_mode = 'DBs'
                break

        list_db_names = [] # list of DB names
        dic_db_cols = {}   # dict {DBname : list of collection names} 
        msg = 'Delete %s:\n  ' % del_mode

        if del_mode == 'DBs' : 
            list_db_names = [k for k,v in dic_name_parent.items() if v is None]
            msg += '\n  '.join(list_db_names)
        else :
            for name,parent in dic_name_parent.items() :
                if parent not in dic_db_cols :
                    dic_db_cols[parent] = [name,]
                else :
                    dic_db_cols[parent].append(name)

            for dbname, lstcols in dic_db_cols.items() :
                msg += '\nfrom DB: %s\n    ' % dbname
                msg += '\n    '.join(lstcols)

        logger.info(msg, self._name)
        resp = confirm_or_cancel_dialog_box(parent=None, text=msg, title='Confirm or cancel')
        print('Response: ', resp)

        if resp :
           if del_mode == 'DBs' : dbu.delete_databases(list_db_names)
           else                 : dbu.delete_collections(dic_db_cols)

           # Regenerate tree model
           self.on_edi_db_filter_finished()

 #-----------------------------

    def set_selection_mode(self):
        #logger.debug('set_selection_model', self._name)
        wtree = cp.cmwdbtree
        if wtree is None : return
        selected = select_item_from_popup_menu(wtree.dic_smodes.keys(), title='Select mode',\
                                               default=cp.cdb_selection_mode.value())
        logger.debug('set_selection_model: %s' % selected, self._name)
        if selected is None : return
        cp.cdb_selection_mode.setValue(selected)
        wtree.set_selection_mode(selected)

 #-----------------------------

    def on_but_clicked(self):
        for but in self.list_of_buts :
            if but.hasFocus() : break
        logger.debug('on_but_clicked "%s"' % but.text())
        if   but == self.but_exp_col  : self.expand_collapse_dbtree()
        elif but == self.but_tabs     : self.view_hide_tabs()
        elif but == self.but_buts     : self.select_visible_buttons()
        elif but == self.but_del      : self.delete_selected_items()
        elif but == self.but_docs     : self.select_doc_widget()
        elif but == self.but_selm     : self.set_selection_mode()
        elif but == self.but_add      : print('TBD')


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
