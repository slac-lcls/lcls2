#------------------------------
"""Class :py:class:`CMDBTree` is a QWTree for database-collection tree presentation
======================================================================================

Usage ::

Created on 2017-03-23 by Mikhail Dubrovin
"""
#------------------------------

from psana.graphqt.QWTree import *
import psana.pscalib.calib.MDBUtils as dbu

#------------------------------

class CMDBTree(QWTree) :
    """GUI for database-collection tree 
    """
    def __init__ (self, parent=None) :

        QWTree.__init__(self, parent)
        self._name = self.__class__.__name__


    def fill_tree_model(self):

        client = dbu.connect_to_server()# host, port)

        #pattern = 'cdb_xcs'
        #pattern = 'cspad'
        #dbnames = [name for name in dbu.database_names(client) if pattern in name]

        dbnames = dbu.database_names(client)

        for dbname in dbnames :
            parentItem = self.model.invisibleRootItem()
            #parentItem.setIcon(icon.icon_folder_open)

            itdb = QStandardItem(dbname)
            itdb.setIcon(icon.icon_folder_closed)
            #itdb.setCheckable(True) 
            parentItem.appendRow(itdb)

            db = dbu.database(client, dbname) 

            for col in dbu.collection_names(db) :
                if not col : continue
                itcol = QStandardItem(col)  
                itcol.setIcon(icon.icon_folder_closed)
                itdb.appendRow(itcol)

                #item.setIcon(icon.icon_table)
                #item.setCheckable(True) 
                #print('append item %s' % (item.text()))

#------------------------------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = CMDBTree()
    w.setGeometry(10, 25, 400, 600)
    w.setWindowTitle(w._name)
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------
