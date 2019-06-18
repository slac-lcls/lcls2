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

from psdaq.control_gui.QWTableOfCheckBoxes import QWTableOfCheckBoxes#, QStandardItem, Qt #icon
from psdaq.control_gui.QWPopupSelectItem import popup_select_item_from_list
#from re import search as re_search

#----------

class CGWPartitionTable(QWTableOfCheckBoxes) :
    """Re-implemented connect_control and field editor using pull-down menu
    """
    LIST_OF_VALUES = [str(v) for v in range(8)]

    def __init__(self, **kwargs) :
        QWTableOfCheckBoxes.__init__(self, **kwargs)

#----------

    def connect_control(self) :
        self.connect_item_selected_to(self.on_item_selected)
        #self.clicked.connect(self.on_click)
        #self.doubleClicked.connect(self.on_double_click)
        #self.connect_item_changed_to(self.on_item_changed)


    def on_item_selected(self, ind_sel, ind_desel):
        #logger.debug("ind   selected : ", ind_sel.row(),  ind_sel.column())
        #logger.debug("ind deselected : ", ind_desel.row(),ind_desel.column()) 
        item = self.model.itemFromIndex(ind_sel)
        logger.debug('on_item_selected: "%s" is selected' % (item.text() if item is not None else None))
        #logger.debug('on_item_selected: %s' % self.getFullNameFromItem(item))

        if not item.isEditable() : return

        selected = popup_select_item_from_list(self, self.LIST_OF_VALUES, min_height=80, dx=-46, dy=-33)
        msg = 'selected %s of the list %s' % (selected, str(self.LIST_OF_VALUES))
        logger.debug(msg)

        if selected is None : return

        item.setText(selected)

#----------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)

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
    w.setWindowTitle('CGWPartitionTable')
    w.move(100,50)
    w.show()
    app.exec_()
    print('%s\nOutput table:' % (50*'_'))
    for rec in w.fill_output_object() : print(rec)
    del w
    del app

#----------
