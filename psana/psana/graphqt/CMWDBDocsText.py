#------------------------------
"""Class :py:class:`CMWDBDocsText` implementation for CMWDBDocsBase
===================================================================

Usage ::
    #### Test: python lcls2/psana/psana/graphqt/CMWDBDocsText.py

    # Import
    from psana.graphqt.CMWDBDocsText import *

See:
  - :class:`CMWDBDocs`
  - `on github <https://github.com/slac-lcls/lcls2>`_.

Created on 2017-04-20 by Mikhail Dubrovin
"""
#------------------------------

from psana.graphqt.CMWDBDocsBase import *
logger = logging.getLogger(__name__)

#------------------------------

class CMWDBDocsText(CMWDBDocsBase, QTextEdit) :
    def __init__(self, text='Select collection in the DB pannel') :
        QTextEdit.__init__(self, text)
        CMWDBDocsBase.__init__(self)
        logger.debug('c-tor CMWDBDocsText')

#------------------------------

    def show_documents(self, dbname, colname, docs) :
        """Re-implementation of the method in QWList.fill_list_model
        """
        CMWDBDocsBase.show_documents(self, dbname, colname, docs)
        msg = 'Show documents for db: %s col: %s' % (dbname, colname)
        logger.info(msg)
        #docs = self.current_docs
        txt = dbu.collection_info(dbname, colname)
        self.setText(txt)

#------------------------------
