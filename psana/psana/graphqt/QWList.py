
"""Class :py:class:`QWList` is a QListView->QWidget for list model
===================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/QWList.py

    from psana.graphqt.QWList import QWList
    w = QWList()

Created on 2017-04-20 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QApplication, QListView, QVBoxLayout, QAbstractItemView
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QBrush
from PyQt5.QtCore import Qt, QModelIndex, QSize
from psana.graphqt.QWIcons import icon


class QWList(QListView):
    """Widget for the list of QListView
    """
    def __init__(self, **kwa):
        kwa.setdefault('parent', None)
        QListView.__init__(self, **kwa)

        icon.set_icons()

        self.model = QStandardItemModel()
        self.set_selection_mode()
        self.fill_list_model() # defines self.model
        self.setModel(self.model)

        self.set_style()
        self.set_tool_tips()

        self.model.itemChanged.connect(self.on_item_changed)
        self.connect_item_selected(self.on_item_selected)
        self.clicked[QModelIndex].connect(self.on_click)
        self.doubleClicked[QModelIndex].connect(self.on_double_click)


    def set_selection_mode(self, smode='extended'):
        logger.debug('Set selection mode: %s'%smode)
        mode = {'single'      : QAbstractItemView.SingleSelection,
                'contiguous'  : QAbstractItemView.ContiguousSelection,
                'extended'    : QAbstractItemView.ExtendedSelection,
                'multi'       : QAbstractItemView.MultiSelection,
                'no selection': QAbstractItemView.NoSelection}[smode]
        self.setSelectionMode(mode)


    def connect_item_selected(self, recipient):
        self.selectionModel().currentChanged[QModelIndex, QModelIndex].connect(recipient)


    def disconnect_item_selected(self, recipient):
        self.selectionModel().currentChanged[QModelIndex, QModelIndex].disconnect(recipient)


    def selected_indexes(self):
        return self.selectedIndexes()


    def selected_items(self):
        indexes =  self.selectedIndexes()
        return [self.model.itemFromIndex(i) for i in self.selectedIndexes()]


    def clear_model(self):
        rows = self.model.rowCount()
        self.model.removeRows(0, rows)


    def fill_list_model(self, **kwa):
        self.clear_model()
        for i in range(20):
            item = QStandardItem('%02d item text'%(i))
            item.setIcon(icon.icon_table)
            item.setCheckable(True)
            item.setSizeHint(QSize(-1,22))
            self.model.appendRow(item)


    def clear_model(self):
        rows = self.model.rowCount()
        self.model.removeRows(0, rows)


    def on_item_selected(self, selected, deselected):
        itemsel = self.model.itemFromIndex(selected)
        if itemsel is not None:
            msg = 'on_item_selected row:%02d selected: %s' % (selected.row(), itemsel.text())
            logger.info(msg)


    def on_item_changed(self, item):
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        msg = 'on_item_changed: item "%s", is at state %s' % (item.text(), state)
        logger.info(msg)


    def on_click(self, index):
        item = self.model.itemFromIndex(index)
        txt = item.text()
        txtshow = txt if len(txt)<50 else '%s...'%txt[:50]
        msg = 'doc clicked in row%02d: %s' % (index.row(), txtshow)
        logger.info(msg)


    def on_double_click(self, index):
        item = self.model.itemFromIndex(index)
        msg = 'on_double_click item in row:%02d text: %s' % (index.row(), item.text())
        logger.debug(msg)


    def set_tool_tips(self):
        self.setToolTip('List model')


    def set_style(self):
        #from psana.graphqt.Styles import style
        self.setWindowIcon(icon.icon_monitor)
        #self.layout().setContentsMargins(0,0,0,0)
        self.setStyleSheet("QListView::item:hover{background-color:#00FFAA;}")


    def closeEvent(self, e):
        logger.debug('closeEvent')
        QListView.closeEvent(self, e)


    def on_exit(self):
        logger.debug('on_exit')
        self.close()


    def process_selected_items(self):
        selitems = self.selected_items()
        msg = '%d Selected items:' % len(selitems)
        for i in selitems:
            msg += '\n  %s' % i.text()
        logger.info(msg)


    if __name__ == "__main__":

      def keyPressEvent(self, e):
        logger.info('keyPressEvent, key=%s' % e.key())
        if   e.key() == Qt.Key_Escape:
            self.close()

        elif e.key() == Qt.Key_S:
            self.process_selected_items()

        else:
            logger.info('Keys:'\
               '\n  ESC - exit'\
               '\n  S - show selected items'\
               '\n')


if __name__ == "__main__":
    import sys
    logging.basicConfig(format='[%(levelname).1s] L:%(lineno)03d %(name)s %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    w = QWList()
    w.setGeometry(10, 25, 400, 600)
    w.setWindowTitle('QWList')
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

# EOF
