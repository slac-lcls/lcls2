#------------------------------
"""Class :py:class:`QWTable` is a QTableView->QWidget for tree model
======================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/QWTable.py

    from psana.graphqt.QWTable import QWTable
    w = QWTable()

Created on 2017-03-26 by Mikhail Dubrovin
"""
#------------------------------
import logging
logger = logging.getLogger(__name__)
#from psana.pyalgos.generic.Logger import logger

from psana.graphqt.QWIcons import icon
from PyQt5.QtWidgets import QTableView, QVBoxLayout, QAbstractItemView #QWidget
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, QModelIndex

from psana.graphqt.CMConfigParameters import cp

#------------------------------ 

class QWTable(QTableView):
     
    def __init__(self, parent=None):
        QTableView.__init__(self, parent)
        self._name = self.__class__.__name__

        icon.set_icons()

        self.model = QStandardItemModel()
        self.set_selection_mode()
        self.fill_table_model() # defines self.model
        self.setModel(self.model)

        self.connect_item_selected_to(self.on_cell_selected)
        #self.clicked.connect(self.on_click)
        #self.doubleClicked.connect(self.on_double_click)
        self.model.itemChanged.connect(self.on_item_changed)

        self.set_style()


    def set_selection_mode(self, smode=QAbstractItemView.ExtendedSelection) :
        logger.debug('Set selection mode: %s'%smode)
        self.setSelectionMode(smode)


    def connect_item_selected_to(self, recipient) :
        #self.selectionModel().selectionChanged[QModelIndex, QModelIndex].connect(recipient)
        self.selectionModel().currentChanged[QModelIndex, QModelIndex].connect(recipient)


    def disconnect_item_selected_from(self, recipient) :
        #self.selectionModel().selectionChanged[QModelIndex, QModelIndex].disconnect(recipient)
        self.selectionModel().currentChanged[QModelIndex, QModelIndex].disconnect(recipient)


    def set_style(self): 
        self.setStyleSheet("QTableView::item:hover{background-color:#00FFAA;}")


    def fill_table_model(self):
        self.clear_model()
        self.model.setHorizontalHeaderLabels(['col0', 'col1', 'col2', 'col3', 'col4']) 
        self.model.setVerticalHeaderLabels(['row0', 'row1', 'row2', 'row3']) 
        for row in range(0, 4):
            for col in range(0, 6):
                item = QStandardItem("itemA %d %d"%(row,col))
                item.setIcon(icon.icon_table)
                item.setCheckable(True) 
                self.model.setItem(row,col,item)
                if col==2 : item.setIcon(icon.icon_folder_closed)
                if col==3 : item.setText('Some text')
                #self.model.appendRow(item)


    def clear_model(self):
        rows,cols = self.model.rowCount(), self.model.columnCount()
        self.model.removeRows(0, rows)
        self.model.removeColumns(0, cols)


    def selected_indexes(self):
        return self.selectedIndexes()


    def selected_items(self):
        indexes =  self.selectedIndexes()
        return [self.model.itemFromIndex(i) for i in self.selectedIndexes()]

 
    def getFullNameFromItem(self, item): 
        #item = self.model.itemFromIndex(ind)        
        ind   = self.model.indexFromItem(item)        
        return self.getFullNameFromIndex(ind)


    def getFullNameFromIndex(self, ind): 
        item = self.model.itemFromIndex(ind)
        if item is None : return None
        self._full_name = item.text()
        self._getFullName(ind)
        return self._full_name


    def _getFullName(self, ind): 
        ind_par  = self.model.parent(ind)
        if(ind_par.column() == -1) :
            item = self.model.itemFromIndex(ind)
            self.full_name = '/' + self._full_name
            #logger.debug('Item full name :' + self._full_name)
            return self._full_name
        else:
            item_par = self.model.itemFromIndex(ind_par)
            self._full_name = item_par.text() + '/' + self._full_name
            self._getFullName(ind_par)


    def closeEvent(self, event): # if the x is clicked
        logger.debug('closeEvent')


    def on_click(self, index):
        item = self.model.itemFromIndex(index)
        msg = 'on_click item in row:%02d text: %s' % (index.row(), item.text())
        logger.info(msg)


    def on_double_click(self, index):
        item = self.model.itemFromIndex(index)
        msg = 'on_double_click item in row:%02d text: %s' % (index.row(), item.text())
        logger.debug(msg)


    def on_item_selected(self, selected, deselected):
        logger.info("%d items selected" % len(selected))
        logger.debug("%d items deselected" % len(deselected))

        
    def on_cell_selected(self, ind_sel, ind_desel):
        #logger.debug("ind   selected : ", ind_sel.row(),  ind_sel.column())
        #logger.debug("ind deselected : ", ind_desel.row(),ind_desel.column()) 
        item = self.model.itemFromIndex(ind_sel)
        logger.info("Item with text '%s' is selected" % (item.text() if item is not None else None),)
        logger.debug("full name: %s" % self.getFullNameFromItem(item))
        #logger.debug(' isEnabled=',item.isEnabled())
        #logger.debug(' isCheckable=',item.isCheckable()) 
        #logger.debug(' checkState=',item.checkState())
        #logger.debug(' isSelectable=',item.isSelectable()) 
        #logger.debug(' isTristate=',item.isTristate())
        #logger.debug(' isEditable=',item.isEditable())
        #logger.debug(' isExpanded=',self.view.isExpanded(ind_sel))


    def on_item_changed(self, item):
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        logger.info("Item with full name %s, is at state %s\n" % (self.getFullNameFromItem(item), state))


    def process_selected_items(self) :
        selitems = self.selected_items()
        msg = '%d Selected items:' % len(selitems)
        for i in selitems :
            msg += '\n  %s' % i.text()
        logger.info(msg)


    def key_usage(self) :
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  S - show selected items'\
               '\n'


    def keyPressEvent(self, e) :
        #logger.info('keyPressEvent, key=', e.key())       
        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_S : 
            self.process_selected_items()

        else :
            logger.info(self.key_usage())

#------------------------------ 

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
    #logger.setPrintBits(0o177777)
    app = QApplication(sys.argv)
    w = QWTable()
    w.setGeometry(100, 100, 700, 300)
    w.setWindowTitle('QWTable')
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------

