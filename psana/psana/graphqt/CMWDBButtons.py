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
import os
import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QWidget, QLabel, QComboBox, QPushButton, QHBoxLayout, QLineEdit
from PyQt5.QtCore import QSize # Qt, QEvent, QPoint, 

from psana.graphqt.CMConfigParameters import cp
from psana.graphqt.Styles import style
from psana.graphqt.QWIcons import icon

from psana.graphqt.QWUtils import change_check_box_dict_in_popup_menu,\
     confirm_or_cancel_dialog_box, select_item_from_popup_menu,\
     get_existing_directory_through_dialog_box,\
     get_open_fname_through_dialog_box

import psana.graphqt.CMDBUtils as dbu

#------------------------------
#------------------------------

class CMWDBButtons(QWidget) :
    """QWidget for managements of configuration parameters"""

    def __init__(self, parent=None) :
        QWidget.__init__(self, parent)
        self._name = 'CMWDBButtons'

        self.log_level_names = list(logging._levelToName.values())

        self.lab_host = QLabel('Host:')
        self.lab_port = QLabel('Port:')
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

        self.cmb_level = QComboBox(self)        
        self.cmb_level.addItems(self.log_level_names)
        self.cmb_level.setCurrentIndex(self.log_level_names.index(cp.log_level.value()))

        self.but_exp_col  = QPushButton('Expand')
        self.but_tabs     = QPushButton('Tabs %s' % cp.char_expand)
        self.but_test     = QPushButton('Test')
        self.but_buts     = QPushButton('Buts %s' % cp.char_expand)
        self.but_del      = QPushButton('Delete')
        self.but_add      = QPushButton('Add')
        self.but_save     = QPushButton('Save')
        self.but_docs     = QPushButton('%s %s' % (cp.cdb_docw.value(), cp.char_expand))
        self.but_selm     = QPushButton('Selection %s' % cp.char_expand)

        self.list_of_buts = (
             self.but_exp_col,
             self.but_tabs,
             self.but_buts,
             self.but_del,
             self.but_add,
             self.but_save,
             self.but_docs,
             self.but_selm,
             self.but_test
        )

        self.edi_db_filter = QLineEdit(cp.cdb_filter.value()) 
        self.edi_db_filter.setPlaceholderText('DB filter')       

        self.hbox = QHBoxLayout() 
        self.hbox.addWidget(self.edi_db_filter)
        self.hbox.addWidget(self.but_exp_col)
        self.hbox.addWidget(self.but_selm)
        self.hbox.addWidget(self.lab_host)
        self.hbox.addWidget(self.cmb_host)
        self.hbox.addWidget(self.lab_port)
        self.hbox.addWidget(self.cmb_port)
        self.hbox.addWidget(self.cmb_level)
        self.hbox.addWidget(self.but_add)
        self.hbox.addWidget(self.but_del)
        self.hbox.addWidget(self.but_save)
        self.hbox.addStretch(1) 
        #self.hbox.addSpacing(20)
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
        self.cmb_level.currentIndexChanged[int].connect(self.on_cmb_level_changed)

        self.edi_db_filter.editingFinished.connect(self.on_edi_db_filter_finished)

        self.but_exp_col.clicked.connect(self.on_but_clicked)
        self.but_tabs   .clicked.connect(self.on_but_clicked)
        self.but_buts   .clicked.connect(self.on_but_clicked)
        self.but_add    .clicked.connect(self.on_but_clicked)
        self.but_del    .clicked.connect(self.on_but_clicked)
        self.but_save   .clicked.connect(self.on_but_clicked)
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
        #self.         setToolTip('Control buttons')
        self.cmb_host.setToolTip('Select DB host')
        self.cmb_port.setToolTip('Select DB port')
        self.but_tabs.setToolTip('Show/hide tabs')
        self.but_buts.setToolTip('Show/hide buttons')
        self.but_exp_col.setToolTip('Expand/collapse DB tree')
        self.edi_db_filter.setToolTip('Enter pattern for\nDB tree item filter')
        self.but_docs.setToolTip('Select style of\ndocuments presentation')
        self.cmb_docw.setToolTip('Select style of\ndocuments presentation')
        self.but_selm.setToolTip('Set items\nselection mode')
        self.but_add.setToolTip('Add current\ndocument to DB')
        self.but_del.setToolTip('Delete selected\nDBs, collections, documents')
        self.but_save.setToolTip('Save current document\nand data in file')
        self.cmb_level.setToolTip('Select logger level')


    def set_style(self):
        #self.         setStyleSheet(style.styleBkgd)
        #self.lab_db_filter.setStyleSheet(style.styleLabel)
        self.lab_host.setStyleSheet(style.styleLabel)
        self.lab_port.setStyleSheet(style.styleLabel)
        self.lab_docs.setStyleSheet(style.styleLabel)

        icon.set_icons()

        self.but_exp_col .setIcon(icon.icon_folder_open)
        self.but_add     .setIcon(icon.icon_plus)
        self.but_save    .setIcon(icon.icon_save)
        self.but_del     .setIcon(icon.icon_minus)
        self.but_docs    .setIcon(icon.icon_table)
        #self.but_level   .setIcon(icon.icon_logger)

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

        self.layout().setContentsMargins(0,0,0,0) # L,U,R,D
        self.setMinimumWidth(500)
        self.setFixedHeight(30)

        #buthi = 30
        #self.but_tabs .setFixedHeight(buthi)
        #self.but_buts .setFixedHeight(buthi)
        #self.but_add  .setFixedHeight(buthi)
        #self.but_del  .setFixedHeight(buthi)
        #self.but_save .setFixedHeight(buthi)
        #self.but_docs .setFixedHeight(buthi)
        #self.but_selm .setFixedHeight(buthi)
        #self.cmb_docw .setFixedHeight(buthi)
        #self.cmb_level.setFixedHeight(buthi)

        self.but_tabs .setFixedWidth(55)
        self.but_buts .setFixedWidth(55)
        self.but_add  .setFixedWidth(60)
        self.but_del  .setFixedWidth(60)
        self.but_save .setFixedWidth(60)
        self.but_docs .setFixedWidth(80)
        self.but_selm .setFixedWidth(80)
        self.cmb_docw .setFixedWidth(60)
        self.cmb_level.setFixedWidth(80)

        self.edi_db_filter.setFixedWidth(80)

    #def resizeEvent(self, e):
    #    logger.debug('resizeEvent size: %s' % str(e.size())) 


    #def moveEvent(self, e):
    #    logger.debug('moveEvent pos: %s' % str(e.pos())) 
#        #cp.posGUIMain = (self.pos().x(),self.pos().y())

    def closeEvent(self, event):
        logger.debug('closeEvent')
        #try    : del cp.guiworkresdirs # CMWDBButtons
        #except : pass # silently ignore


#    def onButLsfStatus(self):
#        queue = cp.bat_queue.value()
#        farm = cp.dict_of_queue_farm[queue]
#        msg, status = gu.msg_and_status_of_lsf(farm)
#        msgi = '\nLSF status for queue %s on farm %s: \n%s\nLSF status for %s is %s' % \
#               (queue, farm, msg, queue, {False:'bad',True:'good'}[status])
#        logger.info(msgi)
#
#        cmd, msg = gu.text_status_of_queues(cp.list_of_queues)
#        msgq = '\nStatus of queues for command: %s \n%s' % (cmd, msg)       
#        logger.info(msgq)

#    def onEditPrefix(self):
#        logger.debug('onEditPrefix')
#        cp.fname_prefix.setValue(str(self.edi_fname_prefix.displayText()))
#        logger.info('Set file name common prefix: ' + str( cp.fname_prefix.value()))
#
#    def onEdiDarkStart(self):
#        str_value = str(self.edi_dark_start.displayText())
#        cp.bat_dark_start.setValue(int(str_value))      
#        logger.info('Set start event for dark run: %s' % str_value)

#    def on_cbx(self, par, cbx):
#        #if cbx.hasFocus() :
#        par.setValue(cbx.isChecked())
#        msg = 'check box %s is set to: %s' % (cbx.text(), str(par.value()))
#        logger.info(msg)
#
#    def on_cbx_host_changed(self, i):
#        self.on_cbx(cp.cdb_host, self.cbx_host)
#
#    def on_cbx_port_changed(self, i):
#        self.on_cbx(cp.cdb_port, self.cbx_port)
#-----------------------------

    def on_edi_db_filter_finished(self):
        """Regenerate tree model tking into account filter field.
        """
        wtree = cp.cmwdbtree
        if wtree is None : return
        edi = self.edi_db_filter
        txt = edi.displayText()
        logger.info('Set filter pattern: "%s"' % txt)
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
                'Save'       : r & 1<<10,\
                'Level'      : r & 1<<11,\
                'Labels'     : r & 1<<12,\
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
        if d['Save']       : w |= 1<<10
        if d['Level']      : w |= 1<<11
        if d['Labels']     : w |= 1<<12
        #if  : w |= 0
        cp.cdb_buttons.setValue(w)


    def set_buttons_visiable(self, dic_buts=None):
        d = self.buttons_dict() if dic_buts is None else dic_buts
        self.set_db_filter_visible (d['DB filter'])
        self.set_host_port_visible (d['Host & port'])
        self.but_exp_col.setVisible(d['Expand'])
        self.but_add.setVisible    (d['Add'])
        self.but_del.setVisible    (d['Delete'])
        self.but_save.setVisible   (d['Save'])
        self.cmb_docw.setVisible   (d['Docw'])
        self.lab_docs.setVisible   (d['Docw'])
        self.but_docs.setVisible   (d['Docs'])
        self.but_tabs.setVisible   (d['Tabs'])
        self.but_test.setVisible   (d['Test'])
        self.but_selm.setVisible   (d['Selection'])
        self.cmb_level.setVisible  (d['Level'])

        self.lab_docs.setVisible   (d['Labels'])
        self.lab_host.setVisible   (d['Labels'])
        self.lab_port.setVisible   (d['Labels'])


    def select_visible_buttons(self):
        logger.debug('select_visible_buttons')
        d = self.buttons_dict()
        resp = change_check_box_dict_in_popup_menu(d, 'Select buttons', msg='Check visible buttons then click Apply or Cancel', parent=self.but_buts)
        logger.debug('select_visible_buttons resp: %s' % resp)
        if resp is None :
            logger.info('Visible buttons selection is cancelled')
            return
        if resp==1 :
            logger.info('Visible buttons: %s' % str(d))
            self.set_buttons_visiable(d)
            self.set_buttons_config_bitword(d)


    def select_doc_widget(self):
        resp = select_item_from_popup_menu(cp.list_of_doc_widgets, parent=self)
        logger.debug('select_doc_widget resp: %s' % resp)
        if resp is None : return
        cp.cdb_docw.setValue(resp)
        cp.cmwdbmain.wdocs.set_docs_widget()

        self.but_docs.setText('%s %s' % (cp.cdb_docw.value(), cp.char_expand))

#-----------------------------
#-----------------------------

    def set_host_port_visible(self, is_visible=True):
        self.lab_host.setVisible(is_visible)
        self.lab_port.setVisible(is_visible)
        self.cmb_host.setVisible(is_visible)
        self.cmb_port.setVisible(is_visible)


    def set_db_filter_visible(self, is_visible=True):
        #self.lab_db_filter.setVisible(is_visible)
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
        logger.info('on_cmb_host_changed - selected: %s' % selected)
        self.on_edi_db_filter_finished() # regenerate tree model

    def on_cmb_port_changed(self):
        selected = self.cmb_port.currentText()
        cp.cdb_port.setValue(int(selected)) 
        logger.info('on_cmb_port_changed - selected: %s' % selected)
        self.on_edi_db_filter_finished() # regenerate tree model

    def on_cmb_docw_changed(self):
        selected = self.cmb_docw.currentText()
        cp.cdb_docw.setValue(selected)
        logger.info('on_cmb_docw_changed - selected: %s' % selected)
        cp.cmwdbmain.wdocs.set_docs_widget()

    def on_cmb_level_changed(self):
        selected = self.cmb_level.currentText()
        cp.log_level.setValue(selected)
        logger.info('Set logger level %s' % selected)
        #cp.cmwdbmain.wdocs.set_docs_widget()
        if cp.qwloggerstd is not None : cp.qwloggerstd.set_level(selected)

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
        """On press of Delete button deside what to delete 
           dbs and collections from the tree or documents from the list
        """
        if   cp.last_selection == cp.DB_COLS: self.delete_selected_items_db_cols()
        elif cp.last_selection == cp.DOCS   : self.delete_selected_items_docs()
        else :
            logger.warning('Nothing selected. Select DBs, collections, '\
                           'or documents then click on Add/Delete/Save button again.')
            return

#-----------------------------

    def delete_selected_items_docs(self):
        logger.debug('delete_selected_items_docs')
        wdocs = cp.cmwdbdocswidg
        if wdocs is None :
            logger.warning('Window with a List of documents does not exist?')
            return

        msg = 'From db: %s\n  col: %s\n  delete docs:\n    ' % (wdocs.dbname, wdocs.colname)
        doc_ids = [item.accessibleText() for item in wdocs.selected_items()]
        msg_recs = ['%s : %s'%(id, dbu.timestamp_id(id)) for id in doc_ids]
        msg += '\n    '.join(msg_recs)

        logger.info(msg)
        resp = confirm_or_cancel_dialog_box(parent=self.but_del, text=msg, title='Confirm or cancel')
        logger.debug('delete_selected_items_docs response: %s' % resp)

        if resp :
            dbu.delete_documents(wdocs.dbname, wdocs.colname, doc_ids)
            cp.cmwdbdocs.show_documents(wdocs.dbname, wdocs.colname, force_update=True)
        else :
            logger.warning('Command "Delete" is cancelled')

#-----------------------------

    def delete_selected_items_db_cols(self):
        wtree = cp.cmwdbtree
        if wtree is None :
            logger.warning('delete_selected_items_db_cols - CMWDBTree object does not exist?')
            return
        # dict of pairs {<item-name> : <item-parent-name>}
        # where <item-parent-name> is None for DB item or DB name for collection item.

        list_name_parent = [(item.text(), None if item.parent() is None else item.parent().text())\
                            for item in wtree.selected_items()]

        if not len(list_name_parent) :
            logger.warning('delete_selected_items: selected nothing - nothing to delete...')
            return

        #for k,v in list_name_parent :
        #    logger.debug('  %s : %s' % (k.ljust(20),v))

        # Define del_mode = COLS or DBS if at least one DB is selected
        del_mode = cp.COLS if all([p is not None for n,p in list_name_parent]) else cp.DBS

        list_db_names = [] # list of DB names
        dic_db_cols = {}   # dict {DBname : list of collection names} 
        msg = 'Delete %s:\n  ' % del_mode

        if del_mode == cp.DBS : 
            list_db_names = [n for n,p in list_name_parent if p is None]
            msg += '\n  '.join(list_db_names)
        else :
            for n,p in list_name_parent :
                if p not in dic_db_cols :
                    dic_db_cols[p] = [n,]
                else :
                    dic_db_cols[p].append(n)

            for dbname, lstcols in dic_db_cols.items() :
                msg += '\nfrom DB: %s\n    ' % dbname
                msg += '\n    '.join(lstcols)

        logger.info(msg)
        resp = confirm_or_cancel_dialog_box(parent=self.but_del, text=msg, title='Confirm or cancel')
        logger.debug('delete_selected_items_db_cols response: %s' % resp)

        if resp :
           if del_mode == cp.DBS : dbu.delete_databases(list_db_names)
           else                  : dbu.delete_collections(dic_db_cols)

           # Regenerate tree model
           self.on_edi_db_filter_finished()

        else :
            logger.warning('Command "Delete" is cancelled')

#-----------------------------

    def set_selection_mode(self):
        #logger.debug('set_selection_model')
        wtree = cp.cmwdbtree
        if wtree is None : return
        selected = select_item_from_popup_menu(wtree.dic_smodes.keys(), title='Select mode',\
                                               default=cp.cdb_selection_mode.value(), parent=self)
        logger.info('Set selection mode: %s' % selected)
        if selected is None : return
        cp.cdb_selection_mode.setValue(selected)
        wtree.set_selection_mode(selected)

#-----------------------------

    def add_selected_item(self):
        """On press of Add button deside what to add
           db from file or document from editor window
        """
        if cp.last_selection == cp.DOCS : self.add_doc()
        else : self.add_db()
        #if cp.last_selection == cp.DB_COLS: self.add_db()
        #else :
        #    logger.warning('Nothing selected to delete. Select DBs, collections, '\
        #                   'or documents then click on Delete button again.')

#-----------------------------

    def add_db(self):
        """Adds DB from file
        """
        logger.debug('TBD: In add_db - Adds DB from file')

        path0 = '.'
        path = get_open_fname_through_dialog_box(self, path0, 'Select file with DB to add', filter='*')
        if path is None : 
            logger.warning('DB file selection is cancelled')
            return

        host = cp.cdb_host.value()
        port = cp.cdb_port.value()

        dbname = os.path.basename(path) # ????????????????????? split file extension?

        logger.info('Add DB "%s" from file %s' % (dbname, path))
        dbu.importdb(host, port, dbname, path) #, **kwa)

#-----------------------------

    def add_doc(self):
        """Adds document from editor to DB
        """
        logger.debug('In add_doc')
        wdoce = cp.cmwdbdoceditor
        if wdoce is None or wdoce.data_nda is None :
            logger.warning('Document is not selected. Select collection in DB then document in List mode.')
            return

        dicdoc = wdoce.get_model_dicdoc()
        nda = wdoce.get_data_nda()

        msg = '\n  '.join(['%12s : %s' % (k,v) for k,v in dicdoc.items()])

#------------------------------
#------------------------------

class CMWDBButtons(QWidget) :
    """QWidget for managements of configuration parameters"""

    def __init__(self, parent=None) :
        QWidget.__init__(self, parent)
        self._name = 'CMWDBButtons'

        self.log_level_names = list(logging._levelToName.values())

        self.lab_host = QLabel('Host:')
        self.lab_port = QLabel('Port:')
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

        self.cmb_level = QComboBox(self)        
        self.cmb_level.addItems(self.log_level_names)
        self.cmb_level.setCurrentIndex(self.log_level_names.index(cp.log_level.value()))

        self.but_exp_col  = QPushButton('Expand')
        self.but_tabs     = QPushButton('Tabs %s' % cp.char_expand)
        self.but_test     = QPushButton('Test')
        self.but_buts     = QPushButton('Buts %s' % cp.char_expand)
        self.but_del      = QPushButton('Delete')
        self.but_add      = QPushButton('Add')
        self.but_save     = QPushButton('Save')
        self.but_docs     = QPushButton('%s %s' % (cp.cdb_docw.value(), cp.char_expand))
        self.but_selm     = QPushButton('Selection %s' % cp.char_expand)

        self.list_of_buts = (
             self.but_exp_col,
             self.but_tabs,
             self.but_buts,
             self.but_del,
             self.but_add,
             self.but_save,
             self.but_docs,
             self.but_selm,
             self.but_test
        )

        self.edi_db_filter = QLineEdit(cp.cdb_filter.value()) 
        self.edi_db_filter.setPlaceholderText('DB filter')       

        self.hbox = QHBoxLayout() 
        self.hbox.addWidget(self.edi_db_filter)
        self.hbox.addWidget(self.but_exp_col)
        self.hbox.addWidget(self.but_selm)
        self.hbox.addWidget(self.lab_host)
        self.hbox.addWidget(self.cmb_host)
        self.hbox.addWidget(self.lab_port)
        self.hbox.addWidget(self.cmb_port)
        self.hbox.addWidget(self.cmb_level)
        self.hbox.addWidget(self.but_add)
        self.hbox.addWidget(self.but_del)
        self.hbox.addWidget(self.but_save)
        self.hbox.addStretch(1) 
        #self.hbox.addSpacing(20)
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
        self.cmb_level.currentIndexChanged[int].connect(self.on_cmb_level_changed)

        self.edi_db_filter.editingFinished.connect(self.on_edi_db_filter_finished)

        self.but_exp_col.clicked.connect(self.on_but_clicked)
        self.but_tabs   .clicked.connect(self.on_but_clicked)
        self.but_buts   .clicked.connect(self.on_but_clicked)
        self.but_add    .clicked.connect(self.on_but_clicked)
        self.but_del    .clicked.connect(self.on_but_clicked)
        self.but_save   .clicked.connect(self.on_but_clicked)
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
        #self.         setToolTip('Control buttons')
        self.cmb_host.setToolTip('Select DB host')
        self.cmb_port.setToolTip('Select DB port')
        self.but_tabs.setToolTip('Show/hide tabs')
        self.but_buts.setToolTip('Show/hide buttons')
        self.but_exp_col.setToolTip('Expand/collapse DB tree')
        self.edi_db_filter.setToolTip('Enter pattern for\nDB tree item filter')
        self.but_docs.setToolTip('Select style of\ndocuments presentation')
        self.cmb_docw.setToolTip('Select style of\ndocuments presentation')
        self.but_selm.setToolTip('Set items\nselection mode')
        self.but_add.setToolTip('Add current\ndocument to DB')
        self.but_del.setToolTip('Delete selected\nDBs, collections, documents')
        self.but_save.setToolTip('Save current document\nand data in file')
        self.cmb_level.setToolTip('Select logger level')


    def set_style(self):
        #self.         setStyleSheet(style.styleBkgd)
        #self.lab_db_filter.setStyleSheet(style.styleLabel)
        self.lab_host.setStyleSheet(style.styleLabel)
        self.lab_port.setStyleSheet(style.styleLabel)
        self.lab_docs.setStyleSheet(style.styleLabel)

        icon.set_icons()

        self.but_exp_col .setIcon(icon.icon_folder_open)
        self.but_add     .setIcon(icon.icon_plus)
        self.but_save    .setIcon(icon.icon_save)
        self.but_del     .setIcon(icon.icon_minus)
        self.but_docs    .setIcon(icon.icon_table)
        #self.but_level   .setIcon(icon.icon_logger)

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

        self.layout().setContentsMargins(0,0,0,0) # L,U,R,D
        self.setMinimumWidth(500)
        self.setFixedHeight(30)

        #buthi = 30
        #self.but_tabs .setFixedHeight(buthi)
        #self.but_buts .setFixedHeight(buthi)
        #self.but_add  .setFixedHeight(buthi)
        #self.but_del  .setFixedHeight(buthi)
        #self.but_save .setFixedHeight(buthi)
        #self.but_docs .setFixedHeight(buthi)
        #self.but_selm .setFixedHeight(buthi)
        #self.cmb_docw .setFixedHeight(buthi)
        #self.cmb_level.setFixedHeight(buthi)

        self.but_tabs .setFixedWidth(55)
        self.but_buts .setFixedWidth(55)
        self.but_add  .setFixedWidth(60)
        self.but_del  .setFixedWidth(60)
        self.but_save .setFixedWidth(60)
        self.but_docs .setFixedWidth(80)
        self.but_selm .setFixedWidth(80)
        self.cmb_docw .setFixedWidth(60)
        self.cmb_level.setFixedWidth(80)

        self.edi_db_filter.setFixedWidth(80)

    #def resizeEvent(self, e):
    #    logger.debug('resizeEvent size: %s' % str(e.size())) 


    #def moveEvent(self, e):
    #    logger.debug('moveEvent pos: %s' % str(e.pos())) 
#        #cp.posGUIMain = (self.pos().x(),self.pos().y())

    def closeEvent(self, event):
        logger.debug('closeEvent')
        #try    : del cp.guiworkresdirs # CMWDBButtons
        #except : pass # silently ignore


#    def onButLsfStatus(self):
#        queue = cp.bat_queue.value()
#        farm = cp.dict_of_queue_farm[queue]
#        msg, status = gu.msg_and_status_of_lsf(farm)
#        msgi = '\nLSF status for queue %s on farm %s: \n%s\nLSF status for %s is %s' % \
#               (queue, farm, msg, queue, {False:'bad',True:'good'}[status])
#        logger.info(msgi)
#
#        cmd, msg = gu.text_status_of_queues(cp.list_of_queues)
#        msgq = '\nStatus of queues for command: %s \n%s' % (cmd, msg)       
#        logger.info(msgq)

#    def onEditPrefix(self):
#        logger.debug('onEditPrefix')
#        cp.fname_prefix.setValue(str(self.edi_fname_prefix.displayText()))
#        logger.info('Set file name common prefix: ' + str( cp.fname_prefix.value()))
#
#    def onEdiDarkStart(self):
#        str_value = str(self.edi_dark_start.displayText())
#        cp.bat_dark_start.setValue(int(str_value))      
#        logger.info('Set start event for dark run: %s' % str_value)

#    def on_cbx(self, par, cbx):
#        #if cbx.hasFocus() :
#        par.setValue(cbx.isChecked())
#        msg = 'check box %s is set to: %s' % (cbx.text(), str(par.value()))
#        logger.info(msg)
#
#    def on_cbx_host_changed(self, i):
#        self.on_cbx(cp.cdb_host, self.cbx_host)
#
#    def on_cbx_port_changed(self, i):
#        self.on_cbx(cp.cdb_port, self.cbx_port)
#-----------------------------

    def on_edi_db_filter_finished(self):
        """Regenerate tree model tking into account filter field.
        """
        wtree = cp.cmwdbtree
        if wtree is None : return
        edi = self.edi_db_filter
        txt = edi.displayText()
        logger.info('Set filter pattern: "%s"' % txt)
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
                'Save'       : r & 1<<10,\
                'Level'      : r & 1<<11,\
                'Labels'     : r & 1<<12,\
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
        if d['Save']       : w |= 1<<10
        if d['Level']      : w |= 1<<11
        if d['Labels']     : w |= 1<<12
        #if  : w |= 0
        cp.cdb_buttons.setValue(w)


    def set_buttons_visiable(self, dic_buts=None):
        d = self.buttons_dict() if dic_buts is None else dic_buts
        self.set_db_filter_visible (d['DB filter'])
        self.set_host_port_visible (d['Host & port'])
        self.but_exp_col.setVisible(d['Expand'])
        self.but_add.setVisible    (d['Add'])
        self.but_del.setVisible    (d['Delete'])
        self.but_save.setVisible   (d['Save'])
        self.cmb_docw.setVisible   (d['Docw'])
        self.lab_docs.setVisible   (d['Docw'])
        self.but_docs.setVisible   (d['Docs'])
        self.but_tabs.setVisible   (d['Tabs'])
        self.but_test.setVisible   (d['Test'])
        self.but_selm.setVisible   (d['Selection'])
        self.cmb_level.setVisible  (d['Level'])

        self.lab_docs.setVisible   (d['Labels'])
        self.lab_host.setVisible   (d['Labels'])
        self.lab_port.setVisible   (d['Labels'])


    def select_visible_buttons(self):
        logger.debug('select_visible_buttons')
        d = self.buttons_dict()
        resp = change_check_box_dict_in_popup_menu(d, 'Select buttons', msg='Check visible buttons then click Apply or Cancel', parent=self.but_buts)
        logger.debug('select_visible_buttons resp: %s' % resp)
        if resp is None :
            logger.info('Visible buttons selection is cancelled')
            return
        if resp==1 :
            logger.info('Visible buttons: %s' % str(d))
            self.set_buttons_visiable(d)
            self.set_buttons_config_bitword(d)


    def select_doc_widget(self):
        resp = select_item_from_popup_menu(cp.list_of_doc_widgets, parent=self)
        logger.debug('select_doc_widget resp: %s' % resp)
        if resp is None : return
        cp.cdb_docw.setValue(resp)
        cp.cmwdbmain.wdocs.set_docs_widget()

        self.but_docs.setText('%s %s' % (cp.cdb_docw.value(), cp.char_expand))

#-----------------------------
#-----------------------------

    def set_host_port_visible(self, is_visible=True):
        self.lab_host.setVisible(is_visible)
        self.lab_port.setVisible(is_visible)
        self.cmb_host.setVisible(is_visible)
        self.cmb_port.setVisible(is_visible)


    def set_db_filter_visible(self, is_visible=True):
        #self.lab_db_filter.setVisible(is_visible)
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
        logger.info('on_cmb_host_changed - selected: %s' % selected)
        self.on_edi_db_filter_finished() # regenerate tree model

    def on_cmb_port_changed(self):
        selected = self.cmb_port.currentText()
        cp.cdb_port.setValue(int(selected)) 
        logger.info('on_cmb_port_changed - selected: %s' % selected)
        self.on_edi_db_filter_finished() # regenerate tree model

    def on_cmb_docw_changed(self):
        selected = self.cmb_docw.currentText()
        cp.cdb_docw.setValue(selected)
        logger.info('on_cmb_docw_changed - selected: %s' % selected)
        cp.cmwdbmain.wdocs.set_docs_widget()

    def on_cmb_level_changed(self):
        selected = self.cmb_level.currentText()
        cp.log_level.setValue(selected)
        logger.info('Set logger level %s' % selected)
        #cp.cmwdbmain.wdocs.set_docs_widget()
        if cp.qwloggerstd is not None : cp.qwloggerstd.set_level(selected)

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
        """On press of Delete button deside what to delete 
           dbs and collections from the tree or documents from the list
        """
        if   cp.last_selection == cp.DB_COLS: self.delete_selected_items_db_cols()
        elif cp.last_selection == cp.DOCS   : self.delete_selected_items_docs()
        else :
            logger.warning('Nothing selected. Select DBs, collections, '\
                           'or documents then click on Add/Delete/Save button again.')
            return

#-----------------------------

    def delete_selected_items_docs(self):
        logger.debug('delete_selected_items_docs')
        wdocs = cp.cmwdbdocswidg
        if wdocs is None :
            logger.warning('Window with a List of documents does not exist?')
            return

        msg = 'From db: %s\n  col: %s\n  delete docs:\n    ' % (wdocs.dbname, wdocs.colname)
        doc_ids = [item.accessibleText() for item in wdocs.selected_items()]
        msg_recs = ['%s : %s'%(id, dbu.timestamp_id(id)) for id in doc_ids]
        msg += '\n    '.join(msg_recs)

        logger.info(msg)
        resp = confirm_or_cancel_dialog_box(parent=self.but_del, text=msg, title='Confirm or cancel')
        logger.debug('delete_selected_items_docs response: %s' % resp)

        if resp :
            dbu.delete_documents(wdocs.dbname, wdocs.colname, doc_ids)
            cp.cmwdbdocs.show_documents(wdocs.dbname, wdocs.colname, force_update=True)
        else :
            logger.warning('Command "Delete" is cancelled')

#-----------------------------

    def delete_selected_items_db_cols(self):
        wtree = cp.cmwdbtree
        if wtree is None :
            logger.warning('delete_selected_items_db_cols - CMWDBTree object does not exist?')
            return
        # dict of pairs {<item-name> : <item-parent-name>}
        # where <item-parent-name> is None for DB item or DB name for collection item.

        list_name_parent = [(item.text(), None if item.parent() is None else item.parent().text())\
                            for item in wtree.selected_items()]

        if not len(list_name_parent) :
            logger.warning('delete_selected_items: selected nothing - nothing to delete...')
            return

        #for k,v in list_name_parent :
        #    logger.debug('  %s : %s' % (k.ljust(20),v))

        # Define del_mode = COLS or DBS if at least one DB is selected
        del_mode = cp.COLS if all([p is not None for n,p in list_name_parent]) else cp.DBS

        list_db_names = [] # list of DB names
        dic_db_cols = {}   # dict {DBname : list of collection names} 
        msg = 'Delete %s:\n  ' % del_mode

        if del_mode == cp.DBS : 
            list_db_names = [n for n,p in list_name_parent if p is None]
            msg += '\n  '.join(list_db_names)
        else :
            for n,p in list_name_parent :
                if p not in dic_db_cols :
                    dic_db_cols[p] = [n,]
                else :
                    dic_db_cols[p].append(n)

            for dbname, lstcols in dic_db_cols.items() :
                msg += '\nfrom DB: %s\n    ' % dbname
                msg += '\n    '.join(lstcols)

        logger.info(msg)
        resp = confirm_or_cancel_dialog_box(parent=self.but_del, text=msg, title='Confirm or cancel')
        logger.debug('delete_selected_items_db_cols response: %s' % resp)

        if resp :
           if del_mode == cp.DBS : dbu.delete_databases(list_db_names)
           else                  : dbu.delete_collections(dic_db_cols)

           # Regenerate tree model
           self.on_edi_db_filter_finished()

        else :
            logger.warning('Command "Delete" is cancelled')

#-----------------------------

    def set_selection_mode(self):
        #logger.debug('set_selection_model')
        wtree = cp.cmwdbtree
        if wtree is None : return
        selected = select_item_from_popup_menu(wtree.dic_smodes.keys(), title='Select mode',\
                                               default=cp.cdb_selection_mode.value(), parent=self)
        logger.info('Set selection mode: %s' % selected)
        if selected is None : return
        cp.cdb_selection_mode.setValue(selected)
        wtree.set_selection_mode(selected)

#-----------------------------

    def add_selected_item(self):
        """On press of Add button deside what to add
           db from file or document from editor window
        """
        if cp.last_selection == cp.DOCS : self.add_doc()
        else : self.add_db()
        #if cp.last_selection == cp.DB_COLS: self.add_db()
        #else :
        #    logger.warning('Nothing selected to delete. Select DBs, collections, '\
        #                   'or documents then click on Delete button again.')

#-----------------------------

    def add_db(self):
        """Adds DB from file
        """
        logger.debug('TBD: In add_db - Adds DB from file')

        path0 = '.'
        path = get_open_fname_through_dialog_box(self, path0, 'Select file with DB to add', filter='*')
        if path is None : 
            logger.warning('DB file selection is cancelled')
            return

        host = cp.cdb_host.value()
        port = cp.cdb_port.value()

        dbname = os.path.basename(path) # ????????????????????? split file extension?

        logger.info('Add DB "%s" from file %s' % (dbname, path))
        dbu.importdb(host, port, dbname, path) #, **kwa)

#-----------------------------

    def add_doc(self):
        """Adds document from editor to DB
        """
        logger.debug('In add_doc')
        wdoce = cp.cmwdbdoceditor
        if wdoce is None or wdoce.data_nda is None :
            logger.warning('Document is not selected. Select collection in DB then document in List mode.')
            return

        dicdoc = wdoce.get_model_dicdoc()
        nda = wdoce.get_data_nda()

        msg = '\n  '.join(['%12s : %s' % (k,v) for k,v in dicdoc.items()])

        from psana.pyalgos.generic.NDArrUtils import info_ndarr
        logger.debug('add_doc \n%s  \n%s' % (msg, info_ndarr(nda, 'data n-d array ')))

        dbnexp = dbu.db_prefixed_name(dicdoc.get('experiment', 'exp_def'))
        dbndet = dbu.db_prefixed_name(dicdoc.get('detector', 'det_def'))
        colname = dicdoc.get('detector', None)

        d = {dbnexp : True, dbndet : True}
        resp = change_check_box_dict_in_popup_menu(d, msg='Add constants\nand metadata to DB', parent=self.but_add)
        logger.debug('add_doc resp: %s' % resp)

        if resp==1 :
            if d[dbnexp] : _,_=dbu.insert_document_and_data(dbnexp, colname, dicdoc, nda)
            if d[dbndet] : _,_=dbu.insert_document_and_data(dbndet, colname, dicdoc, nda)

            wdocs = cp.cmwdbdocswidg
            if wdocs is None : return
            cp.cmwdbdocs.show_documents(wdocs.dbname, wdocs.colname, force_update=True)

        else : 
            logger.warning('Uploading of calibration constants in DB is cancelled')

#-----------------------------

    def save_selected_item(self):
        """On press of Delete button deside what to delete 
           dbs and collections from the tree or documents from the list
        """
        if   cp.last_selection == cp.DB_COLS: self.save_db()
        elif cp.last_selection == cp.DOCS   : self.save_doc()
        else :
            logger.warning('Nothing selected to delete. Select DBs, collections, '\
                           'or documents then click on Delete button again.')
            return

#-----------------------------

    def selected_db_names(self):
        wtree = cp.cmwdbtree
        if wtree is None :
            logger.warning('selected_db_names - CMWDBTree object does not exist?')
            return []
        return [item.text() for item in wtree.selected_items() if item.parent() is None]

#-----------------------------

    def save_db(self):
        """Saves selected DBs in files
        """
        bdnames = self.selected_db_names()
        if len(bdnames) == 0 :
            logger.warning('DB is not selected. Click on DB(s) before "Save" button.')
            return

        logger.debug('In save_db bdnames:\n    %s' % '\n    '.join(bdnames))
        host = cp.cdb_host.value()
        port = cp.cdb_port.value()

        path0 = '.'
        #resp = get_open_fname_through_dialog_box(self, path0, title='Select directory for DB files', filter='*')
        resp = get_existing_directory_through_dialog_box(self, path0, title='Select directory for DB files')

        if resp is None : 
            logger.warning('Saving of DBs is cancelled')
            return

        for dbname in bdnames :
            fname = '%s/%s' % (resp, dbname)
            dbu.exportdb(host, port, dbname, fname) #, **kwa)

#-----------------------------

    def save_doc(self):
        """Saves document metadata and data in files
        """
        logger.debug('In save_doc')

        wdoce = cp.cmwdbdoceditor
        if wdoce is None  or wdoce.data_nda is None :
            logger.warning('Document editor is not available. Select collection in DB then document in List mode.')
            return

        doc  = wdoce.get_model_dicdoc(discard_id_ts=False)
        data = wdoce.get_data_nda()

        prefix = dbu.out_fname_prefix(**doc)

        control = {'data' : True, 'meta' : True}
        resp = change_check_box_dict_in_popup_menu(control, 'Select and confirm',\
               msg='Save current document in file\n%s\nfor types:'%prefix, parent=self.but_add)

        if resp==1 :
            logger.info('Save document data and metadata in file(s) with prefix: %s' % prefix)
            dbu.save_doc_and_data_in_file(doc, data, prefix, control)
        else : 
            logger.warning('Command "Save" is cancelled')

#-----------------------------

    def on_but_clicked(self):
        for but in self.list_of_buts :
            if but.hasFocus() : break
        logger.info('Click on "%s"' % but.text())
        if   but == self.but_exp_col  : self.expand_collapse_dbtree()
        elif but == self.but_tabs     : self.view_hide_tabs()
        elif but == self.but_buts     : self.select_visible_buttons()
        elif but == self.but_del      : self.delete_selected_items()
        elif but == self.but_docs     : self.select_doc_widget()
        elif but == self.but_selm     : self.set_selection_mode()
        elif but == self.but_add      : self.add_selected_item()
        elif but == self.but_save     : self.save_selected_item()
        #elif but == self.but_level    : self.set_logger_level()


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
        logger.debug('on_but_toggled "%s"' % but.text())


    def set_logger_level(self): 
        logger.debug('In set_logger_level')

#-----------------------------

if __name__ == "__main__" :
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    #logger.setPrintBits(0o177777)
    w = CMWDBButtons()
    #w.setGeometry(200, 400, 500, 200)
    w.setWindowTitle('Config Parameters')
    w.show()
    app.exec_()
    del w
    del app

#-----------------------------
