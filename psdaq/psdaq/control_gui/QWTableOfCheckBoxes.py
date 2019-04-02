#------------------------------
"""Class :py:class:`QWTableOfCheckBoxes` is derived as QWTableOfCheckBoxes->QWTable->QTableView
===============================================================================================

Usage ::

    # Run test: python lcls2/psdaq/psdaq/control_gui/QWTableOfCheckBoxes.py

    from psdaq.control_gui.QWTableOfCheckBoxes import QWTableOfCheckBoxes
    w = QWTableOfCheckBoxes(tableio=tableio, title_h=title_h, do_ctrl=True, do_edit=True, is_visv=True)

Created on 2019-03-11 by Mikhail Dubrovin
"""
#----------

import logging
logger = logging.getLogger(__name__)

from psdaq.control_gui.QWTable import QWTable, QStandardItem, Qt #icon

#----------

LIST_STR_CHECK_BOX_STATES = ['UNCHECKED', 'TRISTATE', 'CHECKED']
DICT_CHECK_BOX_STATES = {0:False, 1:True, 2:True}

#----------

class QWTableOfCheckBoxes(QWTable) :
    """Widget for table with fields containing text with check-boxes.
    """
    def __init__(self, **kwargs) :
        QWTable.__init__(self, **kwargs)
        #self._name = self.__class__.__name__


    def fill_table_model(self, **kwargs) :
        """tableio is an I/O list of lists, containing str or [bool,str] elements.
        """
        self.tableio = kwargs.get('tableio', None)  # I/O 2-d-list table with items "str" or "[bool, str]"
        self.do_live = kwargs.get('do_live', False) # live update of I/O tableio

        title_h = kwargs.get('title_h', None)  # list of horizontal header for cols
        title_v = kwargs.get('title_v', None)  # list of vertical header for rows
        do_ctrl = kwargs.get('do_ctrl', False) # allow to change check-box status
        do_edit = kwargs.get('do_edit', False) # allow to edit text of items
        is_visv = kwargs.get('is_visv', True)  # set visible vertical header, by def [1,2,3,...]

        self.clear_model()
        if title_h is not None : self.model.setHorizontalHeaderLabels(title_h) 

        self.verticalHeader().setVisible(is_visv)
        if title_v is not None : self.model.setVerticalHeaderLabels(title_v) 

        self.setFocusPolicy(Qt.NoFocus)

        for row,rec in enumerate(self.tableio):
            for col,fld in enumerate(rec):
                if isinstance(fld, list) :
                    item = QStandardItem(fld[1])
                    if do_ctrl :
                        item.setCheckable(True)
                        #item.setEnabled(do_ctrl)
                        item.setCheckState({False:0, True:2}[fld[0]])
                        item.setAccessibleDescription('type:list')
                else :
                    item = QStandardItem(fld)
                    item.setAccessibleDescription('type:str')

                item.setEditable(do_edit)
                self.model.setItem(row,col,item)

                #item.setIcon(icon.icon_table)
                #item.setText('Some text')

        self.set_exact_widget_size()

#----------

    def connect_control(self) :
        #self.connect_item_selected_to(self.on_item_selected)
        #self.clicked.connect(self.on_click)
        #self.doubleClicked.connect(self.on_double_click)
        self.connect_item_changed_to(self.on_item_changed)


    def on_item_changed(self, item):
        state = LIST_STR_CHECK_BOX_STATES[item.checkState()]
        index = self.model.indexFromItem(item)        
        row, col = index.row(), index.column()
        msg = 'on_item_changed: item(%d,%d) name: %s state: %s'%\
              (row, col, item.text(), state)
        logger.debug(msg)

        if self.do_live :
            if item.accessibleDescription() == 'type:list' :
                self.tableio[row][col][0] = DICT_CHECK_BOX_STATES[item.checkState()]
                self.tableio[row][col][1] = str(item.text())
            elif item.accessibleDescription() == 'type:str' :
                self.tableio[row][col] = str(item.text())


    def fill_output_object(self):
        """Fills output 2-d list from table of items"""
        model = self.model
        list2d_out = []
        for row in range(model.rowCount()) :
            list_row = []
            for col in range(model.columnCount()) :
                item = model.item(row, col)
                state = LIST_STR_CHECK_BOX_STATES[item.checkState()]
                #print('item(%d,%d) name: %s state: %s'% (row, col, item.text(), state))
                rec = str(item.text())
                if item.accessibleDescription() == 'type:list' :
                    rec = [DICT_CHECK_BOX_STATES[item.checkState()], rec]

                list_row.append(rec)

            list2d_out.append(list_row)
        return list2d_out

#----------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)

    title_h = ['proc/pid/host', 'alias']
    tableio = [\
      [[False,'name 1'], 'alias 1'],\
      [[True, 'name 2'], 'alias 2'],\
      [[True, 'name 3'], 'alias 3'],\
      ['name 4', [True, 'alias 4']],\
      ['name 5',         'alias 5'],\
    ]

    print('%s\nInput table:' % (50*'_'))
    for rec in tableio : print(rec)

    w = QWTableOfCheckBoxes(tableio=tableio, title_h=title_h, do_ctrl=True, do_edit=True, do_live=False)
    w.setWindowTitle('QWTableOfCheckBoxes')
    w.move(100,50)
    w.show()
    app.exec_()
    print('%s\nOutput table:' % (50*'_'))
    for rec in tableio : print(rec)
    del w
    del app

#----------
