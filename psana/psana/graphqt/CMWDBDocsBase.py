
"""Class :py:class:`CMWDBDocsBase` abstract interface
=====================================================

Usage ::
    #### Test: python lcls2/psana/psana/graphqt/CMWDBDocsBase.py

    # Import
    from psana.graphqt.CMWDBDocsBase import *

See:
  - :class:`CMWDBDocs`
  - `on github <https://github.com/slac-lcls/lcls2>`_.

Created on 2017-04-20 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import Qt

from psana.graphqt.CMConfigParameters import cp
from psana.graphqt.CMDBUtils import dbu


class CMWDBDocsBase():
    def __init__(self):
        logger.debug('in c-tor')
        self.dbname  = None
        self.colname = None
        self.current_docs = []
        cp.cmwdbdocswidg = self


    def __del__(self):
        cp.cmwdbdocswidg = None


    def show_documents(self, dbname, colname, docs):
        self.dbname, self.colname, self.current_docs = dbname, colname, docs

# EOF

