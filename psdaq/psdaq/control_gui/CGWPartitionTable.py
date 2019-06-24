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

from psdaq.control_gui.QWTableOfCheckBoxes import QWTableOfCheckBoxes, QStandardItem, Qt #icon
from psdaq.control_gui.QWPopupSelectItem import popup_select_item_from_list
#from re import search as re_search
from PyQt5.QtGui import QBrush, QColor

#----------

class CGWPartitionTable(QWTableOfCheckBoxes) :
    """Re-implemented connect_control and field editor using pull-down menu
    """
    LIST_OF_VALUES = [str(v) for v in range(8)]

    def __init__(self, **kwargs) :
        QWTableOfCheckBoxes.__init__(self, **kwargs)
        self.sort_items()
        self.collapse_all()


    def connect_control(self) :
        """re-implementation of QWTable.connect_control"""
        self.clicked.connect(self.on_click)
        self.connect_item_changed_to(self.on_item_changed)
        #self.doubleClicked.connect(self.on_double_click)
        #self.connect_item_selected_to(self.on_item_selected)


    def on_item_selected(self, ind_sel, ind_desel):
        """selection signal is not submitted if item is already selected.
        """
        item = self._si_model.itemFromIndex(ind_sel)
        logger.debug('CGWPartitionTable.on_item_selected: "%s" is selected' % (item.text() if item is not None else None))
        #logger.debug('on_item_selected: %s' % self.getFullNameFromItem(item))
        #logger.debug("ind   selected : ", ind_sel.row(),  ind_sel.column())
        #logger.debug("ind deselected : ", ind_desel.row(),ind_desel.column()) 


    def on_item_changed(self, item):
        if item.column() != self.column_cbx() : return
        item_id = self._si_model.item(item.row(), self.column_id())
        if item_id._is_collapser : self.set_group_check_state(item)

        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        logger.debug('CGWPartitionTable.on_item_changed: "%s" at state %s is_collapser: %s'%\
                     (self.getFullNameFromItem(item), state, item_id._is_collapser))


    def set_group_check_state(self, item_cbx) :
        """for group of the "collapser" check-box set check states""" 
        state = item_cbx.checkState()
        for it in item_cbx._group_cbx_items :
            if it == item_cbx : continue
            it.setCheckState(state)


    def on_click(self, index):
        item = self._si_model.itemFromIndex(index)
        model, row, col, txt = self._si_model, index.row(), index.column(), item.text()
        msg = 'CGWPartitionTable.on_click: item in row:%02d col:%02d text: %s' % (row, col, txt)
        logger.debug(msg)
        #print(msg)

        self.set_readout_group_number(item)

        self.expand_collapse_detector(item, do_collapse=(not self.in_collapsed_group(item)))


    def is_column_for_title(self, col, coltitle) :
        """checks consistency of the column index with expected column-title"""
        return self._si_model.horizontalHeaderItem(col).text() == coltitle


    def approved_column_for_title(self, col, coltitle) :
        if self.is_column_for_title(col, coltitle) : return col
        else :
            logger.error('CGWPartitionTable.approved_column_for_title:'\
                        +'column number:%d is not consistent with column-title %s' % (col,coltitle))
            return None


    def column_cbx(self) : return self.approved_column_for_title(0, 'sel')
    def column_grp(self) : return self.approved_column_for_title(1, 'grp')
    def column_id (self) : return self.approved_column_for_title(3, 'ID')


    def set_readout_group_number(self, item):
        if not self.is_column_for_title(item.column(), 'grp') : return
        if not item.isEditable() : return

        selected = popup_select_item_from_list(self, self.LIST_OF_VALUES, min_height=80, dx=-46, dy=-33)
        msg = 'selected %s' % selected
        logger.debug(msg)

        if selected is None : return
        item.setText(selected)

        # set group readout number for all detector segments
        item_id = self._si_model.item(item.row(), self.column_id())
        if item_id._is_collapser :
            self.disconnect_item_changed_from(self.on_item_changed)
            self.set_readout_group_number_for_detector_segments(item, selected)
            self.connect_item_changed_to(self.on_item_changed)


    def set_readout_group_number_for_detector_segments(self, item, selected) :
        column_grp = self.column_grp() # 2
        item_cbx = self._si_model.item(item.row(), self.column_cbx())
        for it in item_cbx._group_cbx_items :
            item_grp = self._si_model.item(it.row(), column_grp)
            if item_grp == item : continue
            item_grp.setText(selected)


    def detname(self, segname):
        """splits trailed segment number"""
        return segname.rsplit('_')[0]


    def in_collapsed_group(self, item):
        model, col, row  = self._si_model, item.column(), item.row()
        if not self.is_column_for_title(col, 'ID') : return False
        segname_sel = item.text()
        detname = self.detname(segname_sel)

        for r in range(model.rowCount()) :
            segname = model.item(r, col).text()
            if not self.detname(segname) == detname : continue
            if r == row : continue
            if self.isRowHidden(r) : return True
        return False


    def expand_collapse_detector(self, item, do_collapse=True):
        col, row  = item.column(), item.row()
        if not self.is_column_for_title(col, 'ID') : return

        col_cbx, model=self.column_cbx(), self._si_model
        segname_sel = item.text()
        detname = self.detname(segname_sel)
        #msg = 'CGWPartitionTable %s of detector: %s' % ({True:'COLLAPSE', False:'EXPAND'}[do_collapse], detname)
        #logger.debug(msg)
        #print(msg)

        group_cbx_items = []
        for r in range(model.rowCount()) :
            segname = model.item(r, col) .text()
            #print('r:%d t:%s' % (r, segname))

            if not self.detname(segname) == detname : continue
            group_cbx_items.append(model.item(r,col_cbx))
            if r == row : continue

            #if self.isRowHidden(r) :
            if do_collapse : self.hideRow(r)
            else           : self.showRow(r)

        if len(group_cbx_items)==1 : return

        check_state = 0 if all([i.checkState()==0 for i in group_cbx_items]) else\
                      2 if all([i.checkState()==2 for i in group_cbx_items]) else 1

        item_cbx = model.item(row, col_cbx)
        if do_collapse :
            item._is_collapser = True
            item.setToolTip('Click to EXPAND detector segmanes')
            item_cbx._old_check_state = item_cbx.checkState()
            item_cbx._group_cbx_items = tuple(group_cbx_items)
        else :
            if check_state == 1 : check_state = item_cbx._old_check_state
            item._is_collapser = False
            item_cbx._group_cbx_items = ()
            item.setToolTip('Click to COLLAPSE detector segmanes')

        self.disconnect_item_changed_from(self.on_item_changed)
        item_cbx.setCheckState(check_state)
        item.setBackground(QBrush(QColor('yellow' if item._is_collapser else 'white')))
        self.connect_item_changed_to(self.on_item_changed)


    def expand_all(self):
        col_id, model = self.column_id(), self._si_model
        for r in range(model.rowCount()) :
            item = model.item(r, col_id)
            if not item._is_collapser : continue
            self.expand_collapse_detector(item, do_collapse=False)


    def collapse_all(self):
        col_id, model = self.column_id(), self._si_model
        for r in range(model.rowCount()) :
            if self.isRowHidden(r) : continue
            item = model.item(r, col_id)
            self.expand_collapse_detector(item, do_collapse=True)


    def sort_items(self):
        col_id = self.column_id()
        self.sortByColumn(col_id,0) # Qt.AscendingOrder:0,  Qt.DescendingOrder:1

        # initialize items for expand/collapse opperation
        model = self._si_model
        for r in range(model.rowCount()) :
            model.item(r, col_id)._is_collapser = False
 

#    def insert_group_titles(self):
#        model = self._si_model
#        print('XXX rowCount   : %d' % model.rowCount())
#        print('XXX columnCount: %d' % model.columnCount())
        
#        for r in range(model.rowCount()) :
#            print('r:%d t:%s' % (r, model.item(r, 3).text()))

#        item_add = QStandardItem()
#        idx_add = model.indexFromItem(item_add)
#        model.insertRow(3, idx_add)

#----------

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
            self.expand_all()

        elif e.key() == Qt.Key_C : 
            self.collapse_all()

        else :
            logger.debug(self.key_usage())

#----------

def test00_CGWPartitionTable() :
    title_h = ['sel', 'grp', 'level/pid/host', 'ID']
    tableio = [\
               [[True,  ''], '1', 'drp/123456/drp-tst-dev008', 'cookie_9'],\
               [[True,  ''], '1', 'drp/123457/drp-tst-dev009', 'cookie_1'],\
               [[True,  ''], '1', 'drp/123456/drp-tst-dev008', 'cookie_8'],\
               [[True,  ''], '1', 'drp/123457/drp-tst-dev009', 'cookie_0'],\
               [[False, ''],  '', 'teb/123458/drp-tst-dev001', 'teb1'],\
               [[True,  ''], '1', 'drp/123456/drp-tst-dev008', 'tokie_5'],\
               [[True,  ''], '1', 'drp/123457/drp-tst-dev009', 'tokie_6'],\
               [[True,  ''], '1', 'drp/123456/drp-tst-dev008', 'tokie_8'],\
               [[True,  ''], '1', 'drp/123457/drp-tst-dev009', 'tokie_1'],\
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
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG, datefmt='%H:%M:%S')

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
