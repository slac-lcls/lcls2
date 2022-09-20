
"""Class :py:class:`CMWDBTree` is a QWTree for database-collection tree presentation
======================================================================================

Usage ::
    # Test: python lcls2/psana/psana/graphqt/CMWDBTree.py

    # Import
    from psana.graphqt.CMConfigParameters import

    # See test at the EOF

See:
  - :class:`CMWMain`
  - :class:`CMWDBTree`
  - `on github <https://github.com/slac-lcls/lcls2>`_.

Created on 2017-03-23 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

from psana.graphqt.CMConfigParameters import cp
from psana.graphqt.QWTree import *
from psana.graphqt.CMDBUtils import dbu
from psana.graphqt.CMQThreadClient import CMQThreadClient
from PyQt5.QtCore import pyqtSignal


class CMWDBTree(QWTree):
    """GUI for database-collection tree
    """
    db_and_collection_selected = pyqtSignal('QString','QString')

    def __init__(self, parent=None):
        logger.debug('CMWDBTree.__init__')
        self.thread = None
        QWTree.__init__(self, parent)
        self._name = self.__class__.__name__
        cp.cmwdbtree = self
        self.set_selection_mode(cp.cdb_selection_mode.value())
        self.db_and_collection_selected.connect(self.on_db_and_collection_selected)


    def fill_tree_model(self, pattern=''):
        logger.debug('CMWDBTree.fill_tree_model')
        self._pattern = pattern
        self.clear_model()
        #self.fill_tree_model_for_client()

        # connect in thread
        if self.thread is not None: self.thread.quit()
        self.thread = CMQThreadClient()
        self.thread.connect_client_is_ready(self.fill_tree_model_for_client)
        self.thread.start()


    def fill_tree_model_for_client(self):
        #client = dbu.connect_client()
        client = self.thread.client()
        stat = self.thread.quit()

        if client is None:
            host = cp.cdb_host.value()
            port = cp.cdb_port.value()
            logger.warning("Can't connect to host: %s port: %d" % (host, port))
            return

        #pattern = 'cdb_xcs'
        #pattern = 'cspad'
        pattern = self._pattern
        dbnames = dbu.database_names(client)

        logger.debug('CMWDBTree.fill_tree_model_for_client dbnames: %s' % str(dbnames))

        if pattern:
            dbnames = [name for name in dbnames if pattern in name]

        for dbname in dbnames:
            parentItem = self.model.invisibleRootItem()
            #parentItem.setIcon(icon.icon_folder_open)

            itdb = QStandardItem(dbname)
            itdb.setIcon(icon.icon_folder_closed)
            itdb.setEditable(False)
            #itdb.setCheckable(True)
            parentItem.appendRow(itdb)

            db = dbu.database(client, dbname)

            for col in dbu.collection_names(db):
                if not col: continue
                itcol = QStandardItem(col)
                itcol.setIcon(icon.icon_folder_closed)
                itcol.setEditable(False)
                itdb.appendRow(itcol)


    def on_click(self, index):
        """Override method in QWTree"""
        item = self.model.itemFromIndex(index)
        itemname = item.text()
        parent = item.parent()
        parname = parent.text() if parent is not None else None
        msg = 'clicked item: %s parent: %s' % (itemname, parname) # index.row()
        logger.debug(msg)
        if parent is not None: self.db_and_collection_selected.emit(parname, itemname)


    def on_db_and_collection_selected(self, dbname, colname):
        msg = 'on_db_and_collection_selected DB: %s collection: %s' % (dbname, colname)
        logger.debug(msg)
        wdocs = cp.cmwdbdocs
        if wdocs is None: return
        wdocs.show_documents(dbname, colname)


    def on_item_selected(self, selected, deselected):
        QWTree.on_item_selected(self, selected, deselected)

        itemsel = self.model.itemFromIndex(selected)
        if itemsel is not None:
            cp.last_selection = cp.DB_COLS


    def closeEvent(self, e):
        logger.debug('closeEvent')
        if self.thread is not None: self.thread.stop()
        del self.thread
        QWTree.closeEvent(self, e)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = CMWDBTree()
    w.setGeometry(10, 25, 400, 600)
    w.setWindowTitle(w._name)
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

# EOF
