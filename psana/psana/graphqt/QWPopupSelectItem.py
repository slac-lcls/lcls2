#------------------------------
"""
:py:class:`QWPopupSelectItem` - Popup GUI for (str) item selection from the list of items
============================================================================================

Usage::

    # Import
    from psana.graphqt.QWPopupSelectItem import QWPopupSelectItem

    # Methods
    import sys
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication(sys.argv)
    exp_name = popup_select_item_from_list(None, lst)

See:
    - :py:class:`QWPopupSelectItem`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2017-01-26 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-15
"""
#------------------------------

import os

from PyQt5 import QtGui, QtCore, QtWidgets

#------------------------------  

class QWPopupSelectItem(QtWidgets.QDialog) :

    def __init__(self, parent=None, lst=[]):

        QtWidgets.QDialog.__init__(self, parent)

        self.name_sel = None
        self.list = QtWidgets.QListWidget(parent)

        self.fill_list(lst)
        #self.fill_list_icons(lst_icons)

        # Confirmation buttons
        #self.but_cancel = QtWidgets.QPushButton('&Cancel') 
        #self.but_apply  = QtWidgets.QPushButton('&Apply') 
        #cp.setIcons()
        #self.but_cancel.setIcon(cp.icon_button_cancel)
        #self.but_apply .setIcon(cp.icon_button_ok)
        #self.connect(self.but_cancel, QtCore.SIGNAL('clicked()'), self.onCancel)
        #self.connect(self.but_apply,  QtCore.SIGNAL('clicked()'), self.onApply)

        #self.hbox = QtWidgets.QVBoxLayout()
        #self.hbox.addWidget(self.but_cancel)
        #self.hbox.addWidget(self.but_apply)
        ##self.hbox.addStretch(1)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.list)
        #vbox.addLayout(self.hbox)
        self.setLayout(vbox)

        self.list.itemClicked.connect(self.onItemClick)

        self.showToolTips()
        self.setStyle()


    def fill_list(self, lst) :
        for exp in sorted(lst) :
            item = QtWidgets.QListWidgetItem(exp, self.list)
        self.list.sortItems(QtCore.Qt.AscendingOrder)


#    def fill_list_icons(self, lst_icons) :
#        for ind, icon in enumerate(lst_icons) :
#            item = QtWidgets.QListWidgetItem(icon, '%d'%ind, self.list) #'%d'%ind
#            item.setSizeHint(QtCore.QSize(80,30))
#        #self.list.sortItems(QtCore.Qt.AscendingOrder)


    def setStyle(self):
        self.setWindowTitle('Select')
        self.setFixedWidth(100)
        self.setMinimumHeight(600)
        #self.setMaximumWidth(600)
        #self.setStyleSheet(cp.styleBkgd)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        self.setContentsMargins(QtCore.QMargins(-9,-9,-9,-9))
        #self.setStyleSheet(cp.styleBkgd)
        #self.but_cancel.setStyleSheet(cp.styleButton)
        #self.but_apply.setStyleSheet(cp.styleButton)
        self.move(QtGui.QCursor.pos().__add__(QtCore.QPoint(-110,-50)))


    def showToolTips(self):
        #self.but_apply.setToolTip('Apply selection')
        #self.but_cancel.setToolTip('Cancel selection')
        self.setToolTip('Select item from the list')


    def onItemClick(self, item):
        #if item.isSelected(): item.setSelected(False)
        #widg = self.list.itemWidget(item)
        #item.checkState()
        self.name_sel = item.text()
        #if self.name_sel in self.years : return # ignore selection of year
        #print(self.name_sel)
        #logger.debug('Selected experiment %s' % self.name_sel, __name__)  
        self.accept()


    #def mousePressEvent(self, e):
    #    print('mousePressEvent')

        
    def event(self, e):
        #print('event.type', e.type())
        if e.type() == QtCore.QEvent.WindowDeactivate :
            self.reject()
        return QtWidgets.QDialog.event(self, e)
    

    def closeEvent(self, e) :
        #logger.info('closeEvent', __name__)
        self.reject()


    def selectedName(self):
        return self.name_sel

 
    def onCancel(self):
        #logger.debug('onCancel', __name__)
        self.reject()


    def onApply(self):
        #logger.debug('onApply', __name__)  
        self.accept()

#------------------------------  

def popup_select_item_from_list(parent, lst) :
    w = QWPopupSelectItem(parent, lst)
    ##w.show()
    resp=w.exec_()
    if   resp == QtWidgets.QDialog.Accepted : return w.selectedName()
    elif resp == QtWidgets.QDialog.Rejected : return None
    else : return None

#------------------------------
#------------------------------
#----------- TESTS ------------
#------------------------------
#------------------------------
 
def test_select_exp(tname) :
    lst = sorted(os.listdir('/reg/d/psdm/CXI/'))
    print('lst:', lst)
    app = QtWidgets.QApplication(sys.argv)
    exp_name = popup_select_item_from_list(None, lst)
    print('exp_name = %s' % exp_name) 

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    if   tname == '0': test_select_exp(tname)
    #elif tname == '1': test_select_icon(tname)
    else : sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of Test %s' % tname)

#------------------------------
