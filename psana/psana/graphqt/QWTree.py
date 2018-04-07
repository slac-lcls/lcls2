#------------------------------
"""Class :py:class:`QWTree` is a QTreeView->QWidget for tree model
===================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/QWTree.py

    from psana.graphqt.QWTree import QWTree
    w = QWTree()

Created on 2017-03-23 by Mikhail Dubrovin
"""
#------------------------------

from PyQt5.QtWidgets import QTreeView, QVBoxLayout # QWidget
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, QModelIndex

from psana.graphqt.CMConfigParameters import cp
from psana.pyalgos.generic.Logger import logger

from psana.graphqt.QWIcons import icon

#------------------------------
            #QListView()
            #QTableView()
class QWTree(QTreeView) :
    """Widget for tree
    """
    def __init__ (self, parent=None) :

        QTreeView.__init__(self, parent)
        self._name = self.__class__.__name__

        icon.set_icons()

        self.model = QStandardItemModel()

        self.fill_tree_model() # defines self.model

        self.setModel(self.model)
        self.setAnimated(True)

        self.set_style()
        self.show_tool_tips()

        self.expanded.connect(self.on_item_expanded)
        self.collapsed.connect(self.on_item_collapsed)
        self.model.itemChanged.connect(self.on_item_changed)
        self.connect_item_selected_to(self.on_item_selected)
        #self.clicked.connect(self.on_click)       # This works
        #self.doubleClicked.connect(self.on_double_click) # This works
 

    def connect_item_selected_to(self, recipient) :
        self.selectionModel().currentChanged[QModelIndex, QModelIndex].connect(recipient)


    def disconnect_item_selected_from(self, recipient) :
        self.selectionModel().currentChanged[QModelIndex, QModelIndex].disconnect(recipient)


    def clear_model(self):
        rows = self.model.rowCount()
        self.model.removeRows(0, rows)


    def fill_tree_model(self):
        for k in range(0, 4):
            parentItem = self.model.invisibleRootItem()
            parentItem.setIcon(icon.icon_folder_open)
            for i in range(0, k):
                item = QStandardItem('itemA %s %s'%(k,i))
                item.setIcon(icon.icon_table)
                item.setCheckable(True) 
                parentItem.appendRow(item)
                item = QStandardItem('itemB %s %s'%(k,i))
                item.setIcon(icon.icon_folder_closed)
                parentItem.appendRow(item)
                parentItem = item
                print('append item %s' % (item.text()))


    def on_item_expanded(self, ind):
        item = self.model.itemFromIndex(ind) # get QStandardItem
        item.setIcon(icon.icon_folder_open)
        print('Item expanded : ', item.text())


    def on_item_collapsed(self, ind):
        item = self.model.itemFromIndex(ind)
        item.setIcon(icon.icon_folder_closed)
        print('Item collapsed : ', item.text())


    def on_item_selected(self, selected, deselected):
        itemsel = self.model.itemFromIndex(selected)
        if itemsel is not None :
            parent = itemsel.parent()
            spar = parent.text() if parent is not None else 'None'
            msg = 'row: %d selected: %s parent: %s' % (selected.row(), itemsel.text(), spar) 
            print(msg)

        itemdes = self.model.itemFromIndex(deselected)
        if itemdes is not None :
            print('row:', deselected.row(), " deselected", itemdes.text())


    def on_item_changed(self,  item):
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        print("Item with text '%s', is at state %s\n" % ( item.text(),  state))
        #print("Item with text '%s' is changed\n" % ( item.text() ))
    

    def on_click(self):
        print('1-clicked!')

    def on_double_click(self):
        print('2-clicked!')

    #--------------------------
    #--------------------------

    def process_expand(self):
        logger.info('Expand button is clicked', self._name)
        #self.model.set_all_group_icons(self.icon_expand)
        self.expandAll()
        #self.tree_view_is_expanded = True


    def process_collapse(self):
        logger.info('Collapse button is clicked', self._name)
        #self.model.set_all_group_icons(self.icon_collapse)
        self.collapseAll()
        #self.tree_view_is_expanded = False


    #def expand_collapse(self):
    #    if self.isExpanded() : self.collapseAll()
    #    else                 : self.expandAll()

    #--------------------------
    #--------------------------
    #--------------------------

    def show_tool_tips(self):
        self.setToolTip('Tree model') 


    def set_style(self):
        #from psana.graphqt.Styles import style
        self.setWindowIcon(icon.icon_monitor)
        self.setContentsMargins(-9,-9,-9,-9) # QMargins(-5,-5,-5,-5)

        self.setStyleSheet("QTreeView::item:hover{background-color:#00FFAA;}")

        #self.palette = QPalette()
        #self.resetColorIsSet = False
        #self.butELog    .setIcon(icon.icon_mail_forward)
        #self.butFile    .setIcon(icon.icon_save)  
        #self.setMinimumHeight(250)
        #self.setMinimumWidth(550)
        #self.setContentsMargins(-5,-5,-5,-5) # QMargins(-5,-5,-5,-5)
        #self.adjustSize()
        #self.        setStyleSheet(style.styleBkgd)
        #self.butSave.setStyleSheet(style.styleButton)
        #self.butFBrowser.setVisible(False)
        #self.butExit.setText('')
        #self.butExit.setFlat(True)
        #self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
 

    #def resizeEvent(self, e):
        #pass
        #self.frame.setGeometry(self.rect())
        #logger.debug('resizeEvent', self._name) 
        #print('QWTree resizeEvent: %s' % str(self.size()))


    #def moveEvent(self, e):
        #logger.debug('moveEvent', self._name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        #pass


    def closeEvent(self, e):
        logger.debug('closeEvent', self._name)
        QTreeView.closeEvent(self, e)

        #try    : self.gui_win.close()
        #except : pass

        #try    : del self.gui_win
        #except : pass



    def on_exit(self):
        logger.debug('on_exit', self._name)
        self.close()


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
            self.process_expand()

        elif e.key() == Qt.Key_C : 
            self.process_collapse()

        else :
            print(self.key_usage())


#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = QWTree()
    w.setGeometry(10, 25, 400, 600)
    w.setWindowTitle(w._name)
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------
