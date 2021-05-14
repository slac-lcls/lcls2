
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

#from psana.graphqt.CMQThreadClient import CMQThreadClient
from psana.graphqt.CMDBUtils import dbu


from PyQt5.QtCore import pyqtSignal # Qt 


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
        from time import time
        t0_sec = time()

        self._pattern = pattern
        self.clear_model()

        self.fill_tree_model_dbs()

        logger.info('tree-model filling time %.3f sec' % (time()-t0_sec))

        # connect in thread
        #if self.thread is not None: self.thread.quit()
        #self.thread = CMQThreadClient()
        #self.thread.connect_client_is_ready_to(self.fill_tree_model_web)
        #self.thread.start()


    def fill_tree_model_dbs(self):

        #pattern = 'cdb_xcs'
        #pattern = 'cspad'
        pattern = self._pattern
        dbnames = dbu.database_names()

        s = 'CMWDBTree.fill_tree_model_web dbnames: %s\nnumber of dbs: %d' % (str(dbnames), len(dbnames))
        logger.debug(s)
        print(s)

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

            #db = dbu.database(client, dbname)

            if False: # DO NOT FILL COLLECTIONS

              for col in dbu.collection_names(dbname):
                if not col: continue
                itcol = QStandardItem(col)  
                itcol.setIcon(icon.icon_folder_closed)
                itcol.setEditable(False)
                itdb.appendRow(itcol)

                #item.setIcon(icon.icon_table)
                #item.setCheckable(True) 
                #print('append item %s' % (item.text()))


    def fill_tree_model_collections(self, index):
        m = self.model
        item = m.itemFromIndex(index)
        itemname = item.text()
        if m.hasChildren(index):
            logger.info('item %s already has children - update' % itemname)
            m.removeRows(0, m.rowCount(index), index)

        parent = item.parent()
        if parent is not None:
            logger.debug('clicked item %s is not at DB level - do not add collections' % itemname)
            return

        #parname = parent.text() if parent is not None else None
        dbname = itemname
        itdb = item
        for col in dbu.collection_names(dbname):
            if not col: continue
            itcol = QStandardItem(col)
            itcol.setIcon(icon.icon_folder_closed)
            itcol.setEditable(False)
            itdb.appendRow(itcol)
        self.expand(index)


    def on_click(self, index):
        """Override method in QWTree"""
        item = self.model.itemFromIndex(index)
        itemname = item.text()
        parent = item.parent()
        parname = parent.text() if parent is not None else None
        msg = 'clicked item: %s parent: %s' % (itemname, parname) # index.row()
        logger.debug(msg)
        self.fill_tree_model_collections(index)
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


if __name__ == "__main__":

    logging.getLogger('matplotlib').setLevel(logging.WARNING) # supress messages from other logs
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    logging.basicConfig(format='[%(levelname).1s] L:%(lineno)03d %(name)s %(message)s', level=logging.DEBUG)
    logger.info('set logger for module %s' % __name__)

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
