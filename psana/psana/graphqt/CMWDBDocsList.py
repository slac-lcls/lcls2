#------------------------------
"""Class :py:class:`CMWDBDocsList` implementation for CMWDBDocsBase
===================================================================

Usage ::
    #### Test: python lcls2/psana/psana/graphqt/CMWDBDocsList.py

    # Import
    from psana.graphqt.CMWDBDocsList import *

See:
  - :class:`CMWDBDocs`
  - `on github <https://github.com/slac-lcls/lcls2>`_.

Created on 2017-04-20 by Mikhail Dubrovin
"""
#------------------------------

from psana.graphqt.CMWDBDocsBase import * # CMWDBDocsBase, dbu, logger
from psana.graphqt.QWList import QWList, QStandardItem, icon
from PyQt5.QtCore import QSize, pyqtSignal

logger = logging.getLogger(__name__)

#------------------------------

class CMWDBDocsList(CMWDBDocsBase, QWList) :

    document_clicked = pyqtSignal('QString','QString', dict)

    def __init__(self) :
        QWList.__init__(self, parent=None)
        CMWDBDocsBase.__init__(self)
        logger.debug('c-tor CMWDBDocsList')
        self.document_clicked.connect(self.on_document_clicked)
        self.setToolTip('Click on any document to select it\nor [Ctl]-click and drag to select group')

#------------------------------

    def show_documents(self, dbname, colname, docs) :
        """Implementation of the abstract method in CMWDBDocsBase.show_documents
        """
        CMWDBDocsBase.show_documents(self, dbname, colname, docs)
        msg = 'Show documents for db: %s col: %s' % (dbname, colname)
        logger.info(msg)

        self.dbname = dbname
        self.colname = colname
        self.docs = docs

        recs = [dbu.document_info(doc)[0] for doc in docs]

        #for doc in docs : print(doc)
        self.fill_list_model(docs, recs)

#------------------------------

    def fill_list_model(self, docs, recs=None):
        """Re-implementation of the method in QWList.fill_list_model
        """
        self.clear_model()

        if recs is not None :
          rec = 'List of documents'
          rec += ' for DB: %s cols: %s' % (self.dbname, self.colname)\
                 if not(None in (self.dbname, self.colname)) else ''
          item = QStandardItem(rec)
          #item.setCheckable(False) 
          item.setSizeHint(QSize(-1,30))
          item.setEnabled(False)
          item.setSelectable(False)
          self.model.appendRow(item)
          #item.setIcon(icon.icon_contents)

          for doc,rec in zip(docs, recs) :
            item = QStandardItem(rec)
            item.setAccessibleText(str(doc.get('_id','None')))
            item.setIcon(icon.icon_contents)
            #item.setCheckable(True) 
            item.setSizeHint(QSize(-1,22))
            item.setEditable(False)
            self.model.appendRow(item)

        else :
          #for i in range(10):
            #item = QStandardItem('%02d item text'%(i))
            item = QStandardItem('Select collection in the DB pannel')
            item.setIcon(icon.icon_table)
            item.setSelectable(False)
            item.setCheckable(False) 
            item.setEditable(False)
            self.model.appendRow(item)

#------------------------------

    def on_item_selected(self, selected, deselected):
        itemsel = self.model.itemFromIndex(selected)
        if itemsel is not None :
            txt = itemsel.text()
            txtshow = txt if len(txt) < 50 else '%s...'%txt[:50]
            msg = 'Selected document: %s' % txtshow # selected.row()
            logger.info(msg)
        cp.last_selection = cp.DOCS

#------------------------------

    def document_for_id(self, id) :
        """Return document object for (str) id from list of local documents self.docs.
        """
        for doc in self.docs :
            if str(doc['_id']) == id : return doc
        return None

#------------------------------

    def on_click(self, index) :
        """Override method in QWList"""
        item = self.model.itemFromIndex(index)
        itemname = item.text()
        doc_id = item.accessibleText()
        doc = self.document_for_id(doc_id)
        if doc is None : return

        #msg = 'clicked row:%02d ts:%s doc: %s' % (index.row(), doc['time_sec'], itemname)
        msg = 'on_click row:%02d doc: %s' % (index.row(), itemname)
        logger.debug(msg)
        self.document_clicked.emit(self.dbname, self.colname, doc)

#------------------------------

    def on_document_clicked(self, dbname, colname, doc) :
        msg = 'on_document_clicked: DB: %s coll: %s doc: %s' % (dbname, colname, str(doc))
        logger.debug(msg)
        wdoce = cp.cmwdbdoceditor
        if wdoce is None : return
        wdoce.show_document(dbname, colname, doc)
        wdoce.setMinimumWidth(300)
        #wdoce.setFixedWidth(300)

        s0, s1, s2 = cp.cmwdbmain.hsplitter_sizes()
        if s2 < 10 : cp.cmwdbmain.set_hsplitter_size2(300)

#------------------------------

#    def __del__(self) :
#        #QWList.__del__(self)
#        #CMWDBDocsBase.__del__(self)
#       logger.debug('d-tor CMWDBDocsList')
#       h0, h1, h2 = cp.cmwdbmain.hspl.sizes()
#       cp.cmwdbmain.hspl.setSizes((h0, h1+h2, 0)) # shrink 3-d panel

#------------------------------
