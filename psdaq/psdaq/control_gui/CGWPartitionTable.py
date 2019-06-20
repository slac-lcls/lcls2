#------------------------------
"""Class :py:class:`CGWPartitionTable` is derived as QWTableOfCheckBoxes->QWTable->QTableView
===============================================================================================

Usage ::

    # Run test: python lcls2/psdaq/psdaq/control_gui/CGWPartitionTable.py

    from psdaq.control_gui.CGWPartitionTable import CGWPartitionTable
    w = CGWPartitionTable(tableio=tableio, title_h=title_h, do_ctrl=True, is_visv=True)

Created on 2019-03-11 by Mikhail Dubrovin
"""
#----------

import logging
logger = logging.getLogger(__name__)

from psdaq.control_gui.QWTableOfCheckBoxes import QWTableOfCheckBoxes, QStandardItem#, Qt #icon
from psdaq.control_gui.QWPopupSelectItem import popup_select_item_from_list
#from re import search as re_search

#----------

class CGWPartitionTable(QWTableOfCheckBoxes) :
    """Re-implemented connect_control and field editor using pull-down menu
    """
    LIST_OF_VALUES = [str(v) for v in range(8)]

    def __init__(self, **kwargs) :
        QWTableOfCheckBoxes.__init__(self, **kwargs)
        self.sort_items()
        #self.insert_group_titles()

#----------

    def connect_control(self) :
        self.connect_item_selected_to(self.on_item_selected)
        #self.clicked.connect(self.on_click)
        #self.doubleClicked.connect(self.on_double_click)
        #self.connect_item_changed_to(self.on_item_changed)


    def on_item_selected(self, ind_sel, ind_desel):
        #logger.debug("ind   selected : ", ind_sel.row(),  ind_sel.column())
        #logger.debug("ind deselected : ", ind_desel.row(),ind_desel.column()) 
        item = self._si_model.itemFromIndex(ind_sel)
        logger.debug('on_item_selected: "%s" is selected' % (item.text() if item is not None else None))
        #logger.debug('on_item_selected: %s' % self.getFullNameFromItem(item))

        if not item.isEditable() : return

        selected = popup_select_item_from_list(self, self.LIST_OF_VALUES, min_height=80, dx=-46, dy=-33)
        msg = 'selected %s of the list %s' % (selected, str(self.LIST_OF_VALUES))
        logger.debug(msg)

        if selected is None : return

        item.setText(selected)


    def sort_items(self):
        self.sortByColumn(3,0) # Qt.AscendingOrder:0,  Qt.DescendingOrder:1

 

    def insert_group_titles(self):
        model = self._si_model
        print('XXX rowCount   : %d' % model.rowCount())
        print('XXX columnCount: %d' % model.columnCount())
        
        for r in range(model.rowCount()) :
            print('r:%d t:%s' % (r, model.item(r, 3).text()))

        item_add = QStandardItem()
        idx_add = model.indexFromItem(item_add)
        model.insertRow(3, idx_add)

        #self.clear_model()
        #self.setModel(model)

        #self.hideRow(1)

#----------

def test00_CGWPartitionTable() :
    title_h = ['sel', 'grp', 'level/pid/host', 'ID']
    tableio = [\
               [[True,  ''], '1', 'drp/123456/drp-tst-dev008', 'cookie_9'],\
               [[True,  ''], '1', 'drp/123457/drp-tst-dev009', 'cookie_1'],\
               [[True,  ''], '1', 'drp/123456/drp-tst-dev008', 'cookie_8'],\
               [[True,  ''], '1', 'drp/123457/drp-tst-dev009', 'cookie_0'],\
               [[False, ''],  '', 'teb/123458/drp-tst-dev001', 'teb1'],\
               [[False, ''],  '', 'ctr/123459/drp-tst-acc06',  'control'],\
    ]

    print('%s\nInput table:' % (50*'_'))
    for rec in tableio : print(rec)

    w = CGWPartitionTable(tableio=tableio, title_h=title_h,\
                          do_ctrl=True, do_live=False, do_edit=True, do_sele=True, is_visv=False)
    return w

#----------

def test01_CGWPartitionTable() :
    title_h = ['str', 'cbx', 'flags']
    tableio = [\
               [[False, '11', 6], [True,  'name 12', 3], [False, 'name 13aa', 0]],\
               [[False, '21', 6], [False, 'name 22', 3], [False, 'name 23bb', 1]],\
               [[False, '31', 6], [True,  'name 32', 3], [False, 'name 33cc', 2]],\
               [[False, '41', 6], [False, 'name 42', 3], [False, 'name 43dd', 6]],\
               [[False, '51', 6], [True,  'name 52', 3], [False, 'name 53ee',14]],\
               [[False, '61', 6], [False, 'name 62', 3], [True,  'name 63ff',15]],\
    ]

    print('%s\nInput table:' % (50*'_'))
    for rec in tableio : print(rec)

    w = CGWPartitionTable(tableio=tableio, title_h=title_h,\
                          do_ctrl=True, do_live=False, do_edit=True, do_sele=True)
    return w

#----------

if __name__ == "__main__" :
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.debug('%s\nTest %s' % (50*'_', tname))

    w = None
    if   tname == '0': w = test00_CGWPartitionTable()
    elif tname == '1': w = test01_CGWPartitionTable()
    else             : logger.warning('Not-implemented test "%s"' % tname)

    w.setWindowTitle('CGWPartitionTable')
    w.move(100,50)
    w.show()
    app.exec_()
    print('%s\nOutput table:' % (50*'_'))
    for rec in w.fill_output_object() : print(rec)
    del w
    del app

#----------
