#------------------------------
"""Class :py:class:`CMWDBDocsTable` implementation for CMWDBDocsBase
====================================================================

Usage ::
    #### Test: python lcls2/psana/psana/graphqt/CMWDBDocsTable.py

    # Import
    from psana.graphqt.CMWDBDocsTable import *

See:
  - :class:`CMWDBDocs`
  - `on github <https://github.com/slac-lcls/lcls2>`_.

Created on 2017-04-20 by Mikhail Dubrovin
"""
#------------------------------

from psana.graphqt.CMWDBDocsBase import *
from psana.graphqt.QWTable import QWTable, QStandardItem, icon

#from psana.pscalib.calib.MDBUtils import ObjectId
from psana.graphqt.CMDBUtils import ObjectId, doc_add_id_ts

#from psana.graphqt.CMDBUtils import timestamp_id # use dbu
logger = logging.getLogger(__name__)

#------------------------------

class CMWDBDocsTable(CMWDBDocsBase, QWTable) :
    def __init__(self) :
        QWTable.__init__(self, parent=None)
        CMWDBDocsBase.__init__(self)
        logger.debug('c-tor CMWDBDocsTable')

        #self.connect_item_changed_to(self.on_item_changed)
        self.disconnect_item_changed_from(self.on_item_changed)

#------------------------------

    def show_documents(self, dbname, colname, docs) :        
        """Implementation of the abstract method in CMWDBDocsBase.show_documents
        """
        CMWDBDocsBase.show_documents(self, dbname, colname, docs)
        msg = 'Show documents for db: %s col: %s'%(dbname, colname)
        logger.info(msg)
        #for doc in docs : print(doc)
        self.fill_table_model(docs)

        cp.cmwdbmain.set_hsplitter_size2(0)

#------------------------------

    def fill_table_model(self, docs=None):
        """Re-implementation of the method in QWTable.fill_table_model
        """
        self.disconnect_item_changed_from(self.on_item_changed)

        self.clear_model()

        if docs is None :
            self.model.setVerticalHeaderLabels(['Select collection in the DB pannel']) 
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
            for doc in docs : doc_add_id_ts(doc) # add timestamps for all ids

            keys = sorted(docs[0].keys())
            #self.model.setVerticalHeaderLabels(keys) 
            self.model.setHorizontalHeaderLabels(keys) 

            for r,doc in enumerate(docs):
                for c,key in enumerate(keys):
                     v = str(doc.get(key,'N/A'))
                     #s = v if (isinstance(v,str) and len(v)<128) else 'str longer 128 chars'
                     cond = any([isinstance(v,o) for o in (str,dict,ObjectId)])
                     s = str(v) if (cond and len(str(v))<2048) else 'str longer 2048 chars'
                     item = QStandardItem(s)
                     item.setEditable(False)
                     self.model.setItem(r,c,item)

        self.connect_item_changed_to(self.on_item_changed)

#------------------------------

#    def on_item_changed(self, item):
#        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
#        logger.info("Item with full name %s, is at state %s\n" % (self.getFullNameFromItem(item), state))

#------------------------------
