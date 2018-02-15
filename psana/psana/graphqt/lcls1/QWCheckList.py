#!@PYTHON@

"""
Created on 2017-07-27

@author: Mikhail Dubrovin

QListView/QWidget QWCheckList - works on the dictionary of checked items

Usage::
    from graphqt.QWCheckList import QWCheckList, print_dic
    import sys

    d = {'CSPAD1':True, 'CSPAD2x21':False, 'pNCCD1':True, 'Opal1':False}

    app = QtGui.QApplication(sys.argv)
    w = QWCheckList(parent=None, dic_item_state=d)
    w.setGeometry(100, 100, 200, 300)
    w.setWindowTitle('Test Check List')
    w.show()
    app.exec_()

    print_dic(d)
"""
#------------------------------
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
#------------------------------

#class QWCheckList(QtGui.QWidget):
class QWCheckList(QtGui.QListView):
    """Gets dict of item for checkbox GUI in format {'CSPAD1':True, 'CSPAD2x21':False, 'pNCCD1':True, 'Opal1':False}
    and modify this list in gui.
    """
    def __init__(self, parent=None, dic_item_state={}) :
        #QtGui.QWidget.__init__(self, parent)
        QtGui.QListView.__init__(self, parent)

        self.set_model(dic_item_state)
        #self.set_test_model()

        #self.view.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        #self.view.expandAll()
        #self.view.setAnimated(True)

        #self.connect(self.selectionModel(), QtCore.SIGNAL('currentChanged(QModelIndex,QModelIndex)'),self.itemSelected)
        #self.doubleClicked.connect(self.someMethod2) # This works
        #self.expanded.connect(self.itemExpanded)
        #self.collapsed.connect(self.itemCollapsed)

        self.set_style()


    def set_style(self):
        #self.setMinimumSize(100,400)
        self.setMinimumWidth(150)
        self.setMaximumWidth(500)
        self.setMinimumHeight(200)
        self.setContentsMargins(QtCore.QMargins(-9,-9,-9,-9))


    def set_test_model(self) :
        from random import randint
        model = QtGui.QStandardItemModel()
        for n in range(20):
            item = QtGui.QStandardItem('Item %02d' % n) #randint(1, 100))
            check = Qt.Checked
            if randint(0, 4) < 2 :
                check = Qt.Checked
            else :
                check = Qt.Unchecked
            item.setCheckState(check)
            item.setCheckable(True)
            model.appendRow(item)
        self.setModel(model)


    def get_dic_item_state(self) :
        return self.dic_item_state


    def set_dic_item_state(self, dic_item_state) :
        self.set_model(dic_item_state)


    def set_model(self, dic_item_state) :
        self.dic_item_state = dic_item_state
        #print 'self.model()', self.model()
        if self.model() is not None :
            self.model().itemChanged.disconnect(self.on_item_changed)
            #del self.model()
        model = QtGui.QStandardItemModel()
        for k,v in self.dic_item_state.iteritems() :
            item = QtGui.QStandardItem('%s' % k)
            item.setCheckState(Qt.Checked if v else Qt.Unchecked)
            item.setCheckable(True)
            #item.setSelectable(True)
            model.appendRow(item)
        self.setModel(model)
        self.model().itemChanged.connect(self.on_item_changed)


    def on_clicked_test(self, item):
        print "Clicked on item row:'%d' col:%d" % (item.row(), item.column())


    def on_item_changed(self, item):
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        item_txt = str(item.text())
        #print "Item with text '%s', is at state %s" % (item_txt, state)
        self.dic_item_state[item_txt] = [False, True, True][item.checkState()]


    def on_item_changed_test(self, item):
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        item_txt = str(item.text())
        print "Item with text '%s', is at state %s" % (item_txt, state)
        print_dic(self.dic_item_state)

        if item_txt == 'Opal1' :
            d = {'CSPAD2':True, 'CSPAD2x22':False, 'pNCCD2':False, 'Opal2':True, 'Opal3':True}
            self.set_model(d)

        if item_txt == 'Opal2' :
            d = {'CSPAD1':True, 'CSPAD2x21':False, 'pNCCD1':True, 'Opal1':False}
            self.set_model(d)

#-----------------------------

def print_dic(d, fmt='%s : %s') :
    for k,v in d.iteritems() : 
        print fmt % (k.ljust(32),str(v).ljust(32))

#-----------------------------

if __name__ == "__main__" :
    import sys
    d = {'CSPAD1':True, 'CSPAD2x21':False, 'pNCCD1':True, 'Opal1':False}
    #d = {}
    app = QtGui.QApplication(sys.argv)
    w = QWCheckList(parent=None, dic_item_state=d)
    w.setGeometry(100, 100, 200, 300)
    w.setWindowTitle('Test Check List')

    #w.model().itemChanged.connect(w.on_item_changed_test) # on checkbox
    #w.clicked.connect(w.on_clicked_test)                  # on item line

    w.show()
    app.exec_()

    print_dic(d)

#-----------------------------



