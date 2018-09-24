#------------------------------
"""Class :py:class:`CMWDBDocEditor` implementation for CMWDBDocsBase
====================================================================

Usage ::
    #### Test: python lcls2/psana/psana/graphqt/CMWDBDocEditor.py

    # Import
    from psana.graphqt.CMWDBDocEditor import *

See:
  - :class:`CMWDBDocs`
  - `on github <https://github.com/slac-lcls/lcls2>`_.

Created on 2017-04-20 by Mikhail Dubrovin
"""
#------------------------------
import os 
import numpy as np 

import logging
logger = logging.getLogger(__name__)

from psana.graphqt.CMConfigParameters import cp

from psana.graphqt.QWTable import QWTable, QStandardItem, icon
import psana.graphqt.CMDBUtils as dbu
from psana.graphqt.QWUtils import get_open_fname_through_dialog_box

from psana.pyalgos.generic.NDArrUtils import info_ndarr
import psana.pyalgos.generic.Utils as gu
from psana.pscalib.calib.NDArrIO import load_txt#, save_txt

from psana.pscalib.calib.MDBUtils import ObjectId

from PyQt5.QtWidgets import QHeaderView
from PyQt5.QtGui import QBrush
from PyQt5.QtCore import Qt

#------------------------------
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout

class CMWDBDocEditorItem(QWidget) :
    def __init__(self, txt) :
        QWidget.__init__(self, parent=None)
        self.lab = QLabel(txt)
        self.but = QPushButton('Select')
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.lab)
        self.hbox.addWidget(self.but)
        self.setLayout(self.hbox)
        self.set_style()
        self.set_tool_tips()


    def set_style(self):
        self.but.setFixedWidth(60)
        self.but.setFixedHeight(26)
        self.layout().setContentsMargins(0,0,0,0)

        #self.           setStyleSheet(style.styleBkgd)
        #self.lab.setStyleSheet(style.styleTitle)
        #self.but.setStyleSheet(style.styleButton)
 
    def set_tool_tips(self) :
        pass

#------------------------------

class CMWDBDocEditor(QWTable) :
    data_fname = 'data_fname'
    data_fname_value = '<Click and select calibration data file>'

    def __init__(self) :
        QWTable.__init__(self, parent=None)
        logger.debug('c-tor CMWDBDocEditor')
        cp.cmwdbdoceditor = self
        self.data_nda = None

        self.setToolTip('Document editor')


    def __del__(self) :
        #QWTable.__del__(self)
        cp.cmwdbdoceditor = None

#------------------------------

    def show_document(self, dbname, colname, doc) :        
        """Implementation of the abstract method in CMWDBDocsBase
        """
        #CMWDBDocsBase.show_documents(self, dbname, colname, docs)
        msg = 'Show document for db: %s col: %s'%(dbname, colname)
        logger.debug(msg)

        if doc.get('id_data', None) is not None : doc[self.data_fname] = ''
        #for doc in docs : print(doc)
        self.fill_table_model(doc)

        self.data_nda = dbu.get_data_for_doc(dbname, doc)
        logger.debug(info_ndarr(self.data_nda, 'array from DB linked to the document'))

#------------------------------

    def item_is_editable_for_key(self, k):
        forbid = ('id_exp', 'host', 'extpars', 'time_sec', 'data_fname', 'data_size', 'data_dtype',\
                  'data_type', 'uid', 'cwd', 'id_data', 'id_data_ts', '_id', '_id_ts', 'md5')
        return not (k in forbid)

#------------------------------

    def fill_table_model(self, doc=None):
        """Re-implementation of the method in QWList.fill_table_model
        """
        self.disconnect_item_changed_from(self.on_item_changed)

        self.clear_model()

        if doc is None :
            self.model.setVerticalHeaderLabels(['Select document']) 
            #self.model.setHorizontalHeaderLabels(['col0', 'col1', 'col2']) 
            #for row in range(0, 3):
            #  for col in range(0, 6):
            #    item = QStandardItem("itemA %d %d"%(row,col))
            #    item.setIcon(icon.icon_table)
            #    item.setCheckable(True) 
            #    self.model.setItem(row,col,item)
            #    if col==2 : item.setIcon(icon.icon_folder_closed)
            #    if col==3 : item.setText('Some text')
            #    #self.model.appendRow(item)
        else :

            self.model.setHorizontalHeaderLabels(('key', 'value')) 
            self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

            # add items with timestamp for id-s as '_id_ts', 'id_data_ts', 'id_exp_ts'
            for k in ('_id', 'id_data', 'id_exp') :
                v = doc.get(k, None)
                if v is not None : doc['%s_ts'%k] = dbu.timestamp_id(v)

            for r,k in enumerate(sorted(doc.keys())):
                v = doc[k]

                # set key item
                item = QStandardItem(k)
                item.setEnabled(True)
                item.setEditable(False)
                self.model.setItem(r,0,item)

                # set value item
                cond = any([isinstance(v,o) for o in (str,dict,ObjectId)])
                s = str(v) if (cond and len(str(v))<512) else 'str longer 512 chars'
                #if k in ('_id', 'id_data') : 
                #    s = dbu.timestamp_id(v)
                #    msg = '%s = %s converted to %s' % (('doc["%s"]'%k).ljust(16), v, s)
                #    logger.debug(msg)
                item = QStandardItem(s)

                editable = self.item_is_editable_for_key(k) # and k!=self.data_fname
                #item.setCheckable(editable)
                #if editable : item.setCheckState(1)
                item.setEditable(editable)
                item.setEnabled(editable)
                item.setToolTip('Double-click on item\nor click on checkbox\nto change value' if editable else\
                                'This field is auto-generated')

                self.model.setItem(r,1,item)

                if k==self.data_fname :
                    item.setText(self.data_fname_value)
                    item.setEnabled(False)
                    #item.setCheckable(True)
                    #item.setCheckState(1)
                    item.setToolTip('Data file name - click to change')
                    item.setBackground(QBrush(Qt.yellow))
                    #self.widg = QPushButton('Select file')
                    #self.widg = CMWDBDocEditorItem('file-name')
                    #index = self.model.indexFromItem(item)
                    #self.setIndexWidget(index, self.widg)

        self.setColumnWidth(1, 300) # QTableView
        #self.horizontalHeader().setResizeMode(1, 1) # (index, mode)

        self.connect_item_changed_to(self.on_item_changed)

#------------------------------
# Overloaded methods
#------------------------------

    def on_item_changed(self, item):
        """Override method in QWTable"""
        value = self.getFullNameFromItem(item)
        cbxst = item.checkState()
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        #logger.debug('on_item_changed: "%s" state: %s' % (item.text(), state))
        logger.info('Field value changed to "%s"' % item.text())

        #if cbxst == Qt.Checked : 
        #    logger.debug('on_item_changed: "%s" checked' % (value))
        #    self.select_file_name(item)
        #else :
        #    logger.debug('on_item_changed "%s" state: %d' % (value, cbxst))


    def select_file_name(self, item):
        index = self.model.indexFromItem(item)
        value = self.getFullNameFromItem(item)
        row = index.row()
        key = self.model.item(row, 0).text()
        logger.info('Select calibration array file name: %s' % value)
        #if key == self.data_fname : logger.debug("XXX that's it =====================")
        path0 = './'
        path = get_open_fname_through_dialog_box(self, path0, 'Select data file', filter='Text files (*.txt *.dat *.data *.npy)\nAll files (*)')
        logger.debug('select_file_name: %s' % (path))
        if path is None : 
            logger.info('Selection cancelled')
            return

        #item.setCheckState(0)
        self.change_value(item, key, path)


    def change_value(self, item, key, path):
        logger.debug('change_value for key: %s' % (key))
        if key == self.data_fname :
            item.setText(str(path))
            self.data_nda = self.load_nda_from_file(path)
            logger.info(info_ndarr(self.data_nda, 'From file %s loaded array' % path))
            self.set_metadata_values()
            item.setBackground(QBrush(Qt.cyan))
        else :
            txt = gu.load_textfile(path)
            logger.info('From file %s fill field: %s' % (path,txt))
            item.setText(txt)


    def set_metadata_values(self):
        """Sets metadata values associated with self.data_nda
        """
        logger.debug('in set_metadata_values')
        model = self.model
        nda = self.data_nda
        colk, colv = 0, 1
        for row in range(model.rowCount()) :
            key = model.item(row, colk).text()
            if   key == 'data_size'  : model.item(row, colv).setText(str(nda.size))
            elif key == 'data_dtype' : model.item(row, colv).setText(str(nda.dtype))
            elif key == 'data_ndim'  : model.item(row, colv).setText(str(nda.ndim))
            elif key == 'data_shape' : model.item(row, colv).setText(str(nda.shape))
            elif key == 'host'       : model.item(row, colv).setText(gu.get_hostname())
            elif key == 'uid'        : model.item(row, colv).setText(gu.get_login())
            elif key == 'cwd'        : model.item(row, colv).setText(gu.get_cwd())

        logger.info('Model document content:\n  %s\n%s' % (self.info_model_dicdoc(), info_ndarr(self.data_nda, 'data n-d array ')))


    def info_model_dicdoc(self):
        return '\n  '.join(['%12s : %s' % (k,v) for k,v in self.get_model_dicdoc().items()])


    def get_data_nda(self):
        return self.data_nda


    def get_model_dicdoc(self, discard_id_ts=True):
        """Returns dictionary of key-values of current model
        """
        m = self.model
        d = dict([(m.item(r, 0).text(), m.item(r, 1).text()) for r in range(m.rowCount())])
        if d[self.data_fname] == self.data_fname_value : d[self.data_fname] = None
        d['time_sec'] = dbu.time_and_timestamp(**d)[0] # 'time_stamp' is used to fill 'time_sec'

        # remove info items added for display purpose
        if discard_id_ts :
          for k in ('_id_ts', 'id_data_ts', 'id_exp_ts') :
            if d.get(k, None) is not None : del d[k]

        return d


    def load_nda_from_file(self, path):        
        ext = os.path.splitext(path)[1]
        nda = np.load(path) if ext in ('.npy', ) else load_txt(path)
        return nda


    def on_click(self, index):
        """Override method in QWTable"""
        item = self.model.itemFromIndex(index)
        value = self.getFullNameFromItem(item)
        row = index.row()
        key = self.model.item(row, 0).text()
        msg = 'on_click item in row:%02d text: %s' % (index.row(), item.text())
        logger.debug(msg)

        if key == self.data_fname :
            #logger.debug('on_clic: "%s"' % value)
            self.select_file_name(item)
        else :
            if item.isEditable() : logger.info('To edit "%s" use double-click' % value)
            else :                 logger.info('Value for key "%s" is auto-filled' % key)


    def on_double_click(self, index):
        """Override method in QWTable"""
        item = self.model.itemFromIndex(index)
        msg = 'on_double_click: begin edit "%s"' % (item.text())
        logger.debug('on_double_click: begin edit "%s"' % (item.text()))

        
    def on_item_selected(self, ind_sel, ind_desel):
        item = self.model.itemFromIndex(ind_sel)
        logger.debug('on_item_selected "%s"' % (item.text() if item is not None else None))


    def keyPressEvent(self, e) :
        """Override method in QWTable"""
        pass

#------------------------------
#------------------------------

if __name__ == "__main__" :
  def test_CMWDBDocEditor() :
    import sys
    from PyQt5.QtWidgets import QApplication

    doc = {'key0':'val0', 'key1':'val1', 'key2':'val2', 'key3':'val3'}

    logging.basicConfig(format='%(levelname)s %(name)s : %(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CMWDBDocEditor()
    #w.setMinimumSize(600, 300)
    w.fill_table_model(doc)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------

if __name__ == "__main__" :
    test_CMWDBDocEditor()

#------------------------------
