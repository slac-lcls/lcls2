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
import logging
logger = logging.getLogger(__name__)

from psana.graphqt.CMConfigParameters import cp

from psana.graphqt.QWTable import QWTable, QStandardItem, icon
import psana.graphqt.CMDBUtils as dbu
from psana.graphqt.QWUtils import get_open_fname_through_dialog_box

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
        self.setContentsMargins(-9,-9,-9,-9)
        #self.           setStyleSheet(style.styleBkgd)
        #self.lab.setStyleSheet(style.styleTitle)
        #self.but.setStyleSheet(style.styleButton)
 
    def set_tool_tips(self) :
        pass

#------------------------------

class CMWDBDocEditor(QWTable) :
    data_fname = 'data_fname'

    def __init__(self) :
        QWTable.__init__(self, parent=None)
        logger.debug('c-tor CMWDBDocEditor')
        cp.cmwdbdoceditor = self

        self.setToolTip('Document editor')

#------------------------------

    def show_document(self, dbname, colname, doc) :        
        """Implementation of the abstract method in CMWDBDocsBase
        """
        #CMWDBDocsBase.show_documents(self, dbname, colname, docs)
        msg = 'Show document for db: %s col: %s'%(dbname, colname)
        logger.info(msg)

        doc[self.data_fname] = ''
        #for doc in docs : print(doc)
        self.fill_table_model(doc)

#------------------------------

    def item_is_editable_for_key(self, k):
        forbid = ('id_exp', 'data_type', 'host', 'extpars', 'data_size', 'data_type', 'cwd', 'id_data', '_id')
        return not (k in forbid)

#------------------------------

    def fill_table_model(self, doc=None):
        """Re-implementation of the method in QWList.fill_table_model
        """
        self.disconnect_item_changed_from(self.on_item_changed)

        self.clear_model()

        if doc is None :
            self.model.setVerticalHeaderLabels(['Select document or collection']) 
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

            for r,k in enumerate(sorted(doc.keys())):
                v = doc[k]

                # set key item
                item = QStandardItem(k)
                item.setEnabled(True)
                item.setEditable(False)
                self.model.setItem(r,0,item)

                # set value item
                s = v if (isinstance(v,str) and len(v)<128) else 'str longer 128 chars'
                if k in ('_id', 'id_data') : 
                    s = dbu.timestamp_id(v)
                    msg = '%s = %s converted to %s' % (('doc["%s"]'%k).ljust(16), v, s)
                    logger.debug(msg)
                item = QStandardItem(s)

                editable = self.item_is_editable_for_key(k) # and k!=self.data_fname
                item.setCheckable(editable)
                if editable : item.setCheckState(1)
                #item.setCheckState(2 if editable else 0)
                item.setEditable(editable)
                item.setEnabled(editable)
                item.setToolTip('Double-click on item\nor click on checkbox\nto change value' if editable else\
                                'This field is auto-generated')

                self.model.setItem(r,1,item)

                if k==self.data_fname :
                    #item.setCheckable(False)
                    #item.setCheckState(1)
                    item.setToolTip('Field for data file name')
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

    def change_field(self, item):
        index = self.model.indexFromItem(item)
        value = self.getFullNameFromItem(item)
        row = index.row()
        key = self.model.item(row, 0).text()
        logger.debug('change_field "%s" in row:%d key: %s' % (value, row, key))
        if key == self.data_fname : logger.debug("XXX that's it =====================")
        path0 = './'
        path = get_open_fname_through_dialog_box(self, path0, 'Select data file', filter='Text files (*.txt *.dat *.data *.npy)\nAll files (*)')
        logger.debug('change_field: selected file: %s' % (path))
        if path is None : return

        item.setCheckState(0)
        item.setText(str(path))


    def on_item_changed(self, item):
        """Override method in QWTable"""
        value = self.getFullNameFromItem(item)
        cbxst = item.checkState()
        #state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        #logger.debug('Item "%s" at state %s' % (item_name, state))
            
        if cbxst == Qt.Checked : 
            logger.debug('on_item_changed: "%s" checked' % (value))
            self.change_field(item)

        else :
            logger.debug('on_item_changed "%s" state: %d' % (value, cbxst))


    def on_click(self, index):
        """Override method in QWTable"""
        #item = self.model.itemFromIndex(index)
        #msg = 'overriden on_click item in row:%02d text: %s' % (index.row(), item.text())
        #logger.debug(msg)
        pass


    def on_double_click(self, index):
        """Override method in QWTable"""
        item = self.model.itemFromIndex(index)
        #msg = 'overriden on_double_click item in row:%02d text: %s' % (index.row(), item.text())
        msg = 'overriden on_double_click begin edit: %s' % (item.text())
        logger.debug(msg)

        
    def on_item_selected(self, ind_sel, ind_desel):
        #item = self.model.itemFromIndex(ind_sel)
        #logger.info('overriden on_item_selected "%s" is selected' % (item.text() if item is not None else None))
        pass

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
