#------------------------------
"""Class :py:class:`QWTree` is a QTreeView->QWidget for tree model
===================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/QWTree.py
    # Run test: python lcls2/psana/psana/graphqt/QWTree.py

    #from psana.graphqt.QWTree import QWTree
    psdaq.control_gui.QWTree import QWTree
    w = QWTree()

Copied from psana/psana/graphqt/QWTree.py on 2019-03-21 by Mikhail Dubrovin
"""
#------------------------------
import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QTreeView, QVBoxLayout, QAbstractItemView
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, QModelIndex

#from psana.graphqt.CMConfigParameters import cp
#from psana.pyalgos.generic.Logger import logger
from psdaq.control_gui.QWIcons import icon

#------------------------------

class QWTree(QTreeView) :
    """Widget for tree
    """
    dic_smodes = {'single'      : QAbstractItemView.SingleSelection,
                  'contiguous'  : QAbstractItemView.ContiguousSelection,
                  'extended'    : QAbstractItemView.ExtendedSelection,
                  'multi'       : QAbstractItemView.MultiSelection,
                  'no selection': QAbstractItemView.NoSelection}

    def __init__(self, parent=None) :

        QTreeView.__init__(self, parent)
        #self._name = self.__class__.__name__

        icon.set_icons()

        self.model = QStandardItemModel()
        self.set_selection_mode()

        self.fill_tree_model() # defines self.model

        self.setModel(self.model)
        self.setAnimated(True)

        self.set_style()
        self.show_tool_tips()

        self.expanded.connect(self.on_item_expanded)
        self.collapsed.connect(self.on_item_collapsed)
        #self.model.itemChanged.connect(self.on_item_changed)
        self.connect_item_selected_to(self.on_item_selected)
        self.clicked[QModelIndex].connect(self.on_click)
        #self.doubleClicked[QModelIndex].connect(self.on_double_click)
 

    def set_selection_mode(self, smode='extended') :
        logger.debug('Set selection mode: %s'%smode)
        mode = self.dic_smodes[smode]
        self.setSelectionMode(mode)


    def connect_item_selected_to(self, recipient) :
        self.selectionModel().currentChanged[QModelIndex, QModelIndex].connect(recipient)


    def disconnect_item_selected_from(self, recipient) :
        self.selectionModel().currentChanged[QModelIndex, QModelIndex].disconnect(recipient)


    def selected_indexes(self):
        return self.selectedIndexes()


    def selected_items(self):
        indexes =  self.selectedIndexes()
        return [self.model.itemFromIndex(i) for i in self.selectedIndexes()]


    def clear_model(self):
        rows = self.model.rowCount()
        self.model.removeRows(0, rows)


    def fill_tree_model(self):
        self.clear_model()
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
                logger.debug('append item %s' % (item.text()))


    def on_item_expanded(self, ind):
        item = self.model.itemFromIndex(ind) # get QStandardItem
        if item.hasChildren() :
           item.setIcon(icon.icon_folder_open)
        #msg = 'on_item_expanded: %s' % item.text()
        #logger.debug(msg)


    def on_item_collapsed(self, ind):
        item = self.model.itemFromIndex(ind)
        if item.hasChildren() :
           item.setIcon(icon.icon_folder_closed)
        #msg = 'on_item_collapsed: %s' % item.text()
        #logger.debug(msg)


    def on_item_selected(self, selected, deselected):
        itemsel = self.model.itemFromIndex(selected)
        if itemsel is not None :
            parent = itemsel.parent()
            parname = parent.text() if parent is not None else None
            msg = 'selected item: %s row: %d parent: %s' % (itemsel.text(), selected.row(), parname) 
            logger.debug(msg)

        #itemdes = self.model.itemFromIndex(deselected)
        #if itemdes is not None :
        #    msg = 'on_item_selected row: %d deselected %s' % (deselected.row(), itemdes.text())
        #    logger.debug(msg)


    def on_item_changed(self, item):
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        msg = 'on_item_changed: item "%s", is at state %s' % (item.text(), state)
        logger.debug(msg)


    def on_click(self, index):
        item = self.model.itemFromIndex(index)
        parent = item.parent()
        spar = parent.text() if parent is not None else None
        msg = 'clicked item: %s parent: %s' % (item.text(), spar) # index.row()
        logger.debug(msg)


    def on_double_click(self, index):
        item = self.model.itemFromIndex(index)
        msg = 'on_double_click item in row:%02d text: %s' % (index.row(), item.text())
        logger.debug(msg)

    #--------------------------
    #--------------------------

    def process_expand(self):
        logger.debug('process_expand')
        #self.model.set_all_group_icons(self.icon_expand)
        self.expandAll()
        #self.tree_view_is_expanded = True


    def process_collapse(self):
        logger.debug('process_collapse')
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
        self.setContentsMargins(0,0,0,0)
        self.setStyleSheet("QTreeView::item:hover{background-color:#00FFAA;}")


    #def resizeEvent(self, e):
        #pass
        #self.frame.setGeometry(self.rect())
        #logger.debug('resizeEvent') 


    #def moveEvent(self, e):
        #logger.debug('moveEvent') 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        #pass


    def closeEvent(self, e):
        logger.debug('closeEvent')
        QTreeView.closeEvent(self, e)

        #try    : self.gui_win.close()
        #except : pass

        #try    : del self.gui_win
        #except : pass



    def on_exit(self):
        logger.debug('on_exit')
        self.close()


    if __name__ == "__main__" :

      def key_usage(self) :
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  E - expand'\
               '\n  C - collapse'\
               '\n'


      def keyPressEvent(self, e) :
        #logger.debug('keyPressEvent, key = %s'%e.key())       
        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_E : 
            self.process_expand()

        elif e.key() == Qt.Key_C : 
            self.process_collapse()

        else :
            logger.debug(self.key_usage())


#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
    #logger.setPrintBits(0o177777)
    app = QApplication(sys.argv)
    w = QWTree()
    w.setGeometry(10, 25, 400, 600)
    w.setWindowTitle('QWTree')
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------
