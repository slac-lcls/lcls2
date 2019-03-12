#------------------------------
"""Class :py:class:`CGWPartitionList` is a QWList->QListView->QWidget for list model
====================================================================================

Usage ::

    # Run test: python lcls2/psdaq/psdaq/control_gui/CGWPartitionList.py

    #from psana.graphqt.CGWPartitionList import CGWPartitionList
    from psdaq.control_gui.CGWPartitionList import CGWPartitionList
    w = CGWPartitionList(lst)

Created on 2019-03-11 by Mikhail Dubrovin
"""
#----------

import logging
logger = logging.getLogger(__name__)

from psdaq.control_gui.QWList import QWList, QStandardItem #icon

#----------

class CGWPartitionList(QWList) :
    """Widget for List
    """

    def __init__(self, **kwargs) :
        QWList.__init__(self, **kwargs)
        #self._name = self.__class__.__name__


    def fill_list_model(self, **kwargs):
        self.clear_model()
        listio = kwargs.get('listio', [])
        for rec in listio:
            item = QStandardItem(rec)
            #item.setIcon(icon.icon_table)
            #item.setCheckable(True) 
            self.model.appendRow(item)

#----------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    #logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWPartitionList(listio=['rec1','rec2','rec3'])
    w.setGeometry(10, 25, 400, 600)
    w.setWindowTitle('CGWPartitionList')
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

#----------

