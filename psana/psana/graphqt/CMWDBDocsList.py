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
from PyQt5.QtCore import QSize

logger = logging.getLogger(__name__)

#------------------------------

class CMWDBDocsList(CMWDBDocsBase, QWList) :
    def __init__(self) :
        QWList.__init__(self, parent=None)
        CMWDBDocsBase.__init__(self)
        logger.debug('c-tor CMWDBDocsList')

#------------------------------

    def show_documents(self, dbname, colname, docs) :
        """Implementation of the abstract method in CMWDBDocsBase
        """
        CMWDBDocsBase.show_documents(self, dbname, colname, docs)
        msg = 'Show documents for db: %s col: %s' % (dbname, colname)
        logger.info(msg)

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
          #item.setSelectable(False)
          #item.setCheckable(False) 
          item.setSizeHint(QSize(-1,30))
          item.setEnabled(False)
          self.model.appendRow(item)
          #item.setIcon(icon.icon_contents)
          for doc,rec in zip(docs, recs) :
            item = QStandardItem(rec)
            item.setAccessibleText(str(doc.get('_id','None')))
            item.setIcon(icon.icon_contents)
            #item.setCheckable(True) 
            item.setSizeHint(QSize(-1,22))
            self.model.appendRow(item)

        else :
          #for i in range(10):
            #item = QStandardItem('%02d item text'%(i))
            item = QStandardItem('Select collection in the DB pannel')
            item.setIcon(icon.icon_table)
            item.setSelectable(False)
            item.setCheckable(False) 
            self.model.appendRow(item)

#------------------------------

    def on_item_selected(self, selected, deselected):
        itemsel = self.model.itemFromIndex(selected)
        if itemsel is not None :
            txt = itemsel.text()
            txtshow = txt if len(txt)<50 else '%s...'%txt[:50]
            msg = 'doc selected in row%02d: %s' % (selected.row(), txtshow) 
            logger.info(msg)

        cp.last_selection = cp.DOCS

#------------------------------
