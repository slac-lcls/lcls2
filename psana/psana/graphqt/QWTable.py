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

from psana.graphqt.QWIcons import icon
from PyQt5.QtWidgets import QTableView, QVBoxLayout #QWidget
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, QModelIndex

from psana.graphqt.CMConfigParameters import cp
from psana.pyalgos.generic.Logger import logger

#------------------------------ 

class QWTable(QTableView):
     
    def __init__(self, parent=None):
        QTableView.__init__(self, parent)
        self._name = self.__class__.__name__

        icon.set_icons()

        self.model = QStandardItemModel()
        self.fill_table_model() # defines self.model

        self.setModel(self.model)

        self.connect_item_selected_to(self.on_cell_selected)
 
        #self.clicked.connect(self.on_click)       # This works
        #self.doubleClicked.connect(self.on_double_click) # This works
        self.model.itemChanged.connect(self.on_item_changed)

        self.set_style()


    def connect_item_selected_to(self, recipient) :
        #self.selectionModel().selectionChanged[QModelIndex, QModelIndex].connect(recipient)
        self.selectionModel().currentChanged[QModelIndex, QModelIndex].connect(recipient)


    def disconnect_item_selected_from(self, recipient) :
        #self.selectionModel().selectionChanged[QModelIndex, QModelIndex].disconnect(recipient)
        self.selectionModel().currentChanged[QModelIndex, QModelIndex].disconnect(recipient)


    def set_style(self): 
        self.setStyleSheet("QTableView::item:hover{background-color:#00FFAA;}")


    def fill_table_model(self):
        self.model.setHorizontalHeaderLabels(['col1', 'col2', 'col3', 'col4', 'col5']) 
        self.model.setVerticalHeaderLabels(['row1', 'row2', 'row3']) 

        for row in range(0, 4):
            parentItem = self.model.invisibleRootItem()
            #parentItem.setIcon(icon.icon_folder_open)
            for col in range(0, 6):
                item = QStandardItem("itemA %d %d"%(row,col))
                item.setIcon(icon.icon_table)
                item.setCheckable(True) 
                self.model.setItem(row,col,item)
                if col==2 : item.setIcon(icon.icon_folder_closed)
                if col==3 : item.setText('Some text')

                self.model.appendRow(item)
                #parentItem.appendRow(item)

 
    def getFullNameFromItem(self, item): 
        #item = self.model.itemFromIndex(ind)        
        ind   = self.model.indexFromItem(item)        
        return self.getFullNameFromIndex(ind)


    def getFullNameFromIndex(self, ind): 
        item = self.model.itemFromIndex(ind)
        self._full_name = item.text()
        self._getFullName(ind)
        return self._full_name


    def _getFullName(self, ind): 
        ind_par  = self.model.parent(ind)
        if(ind_par.column() == -1) :
            item = self.model.itemFromIndex(ind)
            self.full_name = '/' + self._full_name
            #print('Item full name :' + self._full_name)
            return self._full_name
        else:
            item_par = self.model.itemFromIndex(ind_par)
            self._full_name = item_par.text() + '/' + self._full_name
            self._getFullName(ind_par)


    def closeEvent(self, event): # if the x is clicked
        print('closeEvent')


    def on_click(self):
        print('1-clicked')


    def on_double_click(self):
        print('2-clicked')


    def on_item_selected(self, selected, deselected):
        print(len(selected),   "items selected")
        print(len(deselected), "items deselected")

        
    def on_cell_selected(self, ind_sel, ind_desel):
        #print("ind   selected : ", ind_sel.row(),  ind_sel.column())
        #print("ind deselected : ", ind_desel.row(),ind_desel.column()) 
        item     = self.model.itemFromIndex(ind_sel)
        print("Item with text '%s' is selected" % ( item.text() ),)
        print("full name:", self.getFullNameFromItem(item))
        #print(' isEnabled=',item.isEnabled())
        #print(' isCheckable=',item.isCheckable()) 
        #print(' checkState=',item.checkState())
        #print(' isSelectable=',item.isSelectable()) 
        #print(' isTristate=',item.isTristate())
        #print(' isEditable=',item.isEditable())
        #print(' isExpanded=',self.view.isExpanded(ind_sel))


    def on_item_changed(self, item):
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        print("Item with full name %s, is at state %s\n" % (self.getFullNameFromItem(item), state))


    def key_usage(self) :
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  E - expand'\
               '\n  C - collapse'\
               '\n'


    def keyPressEvent(self, e) :
        #print('keyPressEvent, key=', e.key())       
        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_E : 
            pass #self.process_expand()

        elif e.key() == Qt.Key_C : 
            pass #self.process_collapse()

        else :
            print(self.key_usage())

#------------------------------ 

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = QWTable()
    w.setGeometry(100, 100, 700, 300)
    w.setWindowTitle('Item selection table')
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------

