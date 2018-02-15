#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: QWPopupSelectColorBar.py 13109 2017-01-31 18:49:38Z dubrovin@SLAC.STANFORD.EDU $
#
# Description:
#  Module QWPopupSelectColorBar...
#------------------------------------------------------------------------

"""Popup GUI for (str) item selection from the list of items"""

#------------------------------

import os

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
import graphqt.ColorTable as ct
from graphqt.Styles import style

#------------------------------  

class QWPopupSelectColorBar(QtGui.QDialog) :

    def __init__(self, parent=None):

        QtGui.QDialog.__init__(self, parent)

        # Confirmation buttons
        self.but_cancel = QtGui.QPushButton('&Cancel') 
        #self.but_apply  = QtGui.QPushButton('&Apply') 
        #self.but_create = QtGui.QPushButton('&Create') 

        #cp.setIcons()
        #self.but_cancel.setIcon(cp.icon_button_cancel)
        #self.but_apply .setIcon(cp.icon_button_ok)
        #self.connect(self.but_cancel, QtCore.SIGNAL('clicked()'), self.onCancel)
        #self.connect(self.but_apply,  QtCore.SIGNAL('clicked()'), self.onApply)

        #self.hbox = QtGui.QVBoxLayout()
        #self.hbox.addWidget(self.but_cancel)
        #self.hbox.addWidget(self.but_apply)
        ##self.hbox.addStretch(1)

        self.ctab_selected = None
        self.list = QtGui.QListWidget(parent)

        self.fill_list()

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.list)
        self.setLayout(vbox)

        self.connect(self.but_cancel, QtCore.SIGNAL('clicked()'), self.onCancel)
        self.list.itemClicked.connect(self.onItemClick)

        self.showToolTips()
        self.setStyle()


    def fill_list(self) :
        for i in range(1,9) :
           item = QtGui.QListWidgetItem('%02d'%i, self.list)
           item.setSizeHint(QtCore.QSize(200,30))
           item._coltab_index = i
           #item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
           lab = QtGui.QLabel(parent=None)
           lab.setPixmap(ct.get_pixmap(i, size=(200,30)))
           self.list.setItemWidget(item, lab)

        item = QtGui.QListWidgetItem('cancel', self.list)
        self.list.setItemWidget(item, self.but_cancel)
        
  
    def onItemClick(self, item):
        self.ctab_selected = item._coltab_index
        self.accept()


#class QWPopupSelectColorBarV0(QtGui.QDialog) :

    def __init__V0(self, parent=None):

        QtGui.QDialog.__init__(self, parent)

        self.ctab_selected = None
        #self.list = QtGui.QListWidget(parent)

        #self.fill_list(lst)
        #self.fill_list_icons(lst_icons)

        # Confirmation buttons
        self.but_cancel = QtGui.QPushButton('&Cancel') 
        #self.but_apply  = QtGui.QPushButton('&Apply') 
        #self.but_create = QtGui.QPushButton('&Create') 

        #cp.setIcons()
        #self.but_cancel.setIcon(cp.icon_button_cancel)
        #self.but_apply .setIcon(cp.icon_button_ok)
        #self.connect(self.but_cancel, QtCore.SIGNAL('clicked()'), self.onCancel)
        #self.connect(self.but_apply,  QtCore.SIGNAL('clicked()'), self.onApply)

        #self.hbox = QtGui.QVBoxLayout()
        #self.hbox.addWidget(self.but_cancel)
        #self.hbox.addWidget(self.but_apply)
        ##self.hbox.addStretch(1)


        vbox = QtGui.QVBoxLayout()
        for i in range(1,9) :
           lab = QtGui.QLabel(parent=None)
           lab.setPixmap(ct.get_pixmap(i, size=(200,30)))
           #lab.setText('%02d'%i) # set text !!!OR!!! pixmam
           #lab.setContentsMargins(QtCore.QMargins(-5,-5,-5,-5))
           #lab.setFixedSize(200,10)
           lab._coltab_index = i
           vbox.addWidget(lab)

        vbox.addStretch()
        vbox.addWidget(self.but_cancel)
        self.setLayout(vbox)

        self.connect(self.but_cancel, QtCore.SIGNAL('clicked()'), self.onCancel)

        self.showToolTips()
        self.setStyle()


    def setStyle(self):
        self.setWindowTitle('Select')
        self.setFixedWidth(215)
        lst_len = self.list.__len__()        
        self.setMinimumHeight(30*lst_len+10)
        #self.setMaximumWidth(600)
        #self.setStyleSheet(style.styleBkgd)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        self.setContentsMargins(QtCore.QMargins(-9,-9,-9,-9))
        #self.setStyleSheet(style.styleBkgd)
        #self.but_create.setStyleSheet(style.styleButton)
        #self.but_apply.setStyleSheet(style.styleButton)
        self.but_cancel.setStyleSheet(style.styleButton)
        self.but_cancel.setFixedSize(200,30)
        self.move(QtGui.QCursor.pos().__add__(QtCore.QPoint(-110,-50)))


    def showToolTips(self):
        #self.but_apply.setToolTip('Apply selection')
        #self.but_cancel.setToolTip('Cancel selection')
        self.setToolTip('Select color table')


    def mousePressEvent(self, e):
        #print 'mousePressEvent'
        child = self.childAt(e.pos())
        if isinstance(child, QtGui.QLabel) :
            #print 'Selected color table index: %d' % child._coltab_index
            self.ctab_selected = child._coltab_index
            self.accept()


    def event(self, e):
        #print 'event.type', e.type()
        if e.type() == QtCore.QEvent.WindowDeactivate :
            self.reject()
        return QtGui.QDialog.event(self, e)
    

    def closeEvent(self, e) :
        #logger.info('closeEvent', __name__)
        self.reject()


    def selectedColorTable(self):
        return self.ctab_selected

 
    def onCancel(self):
        #logger.debug('onCancel', __name__)
        self.reject()


#    def onApply(self):
#        #logger.debug('onApply', __name__)  
#        self.accept()


#    def onSelect(self):
#        print 'onSelect'

#------------------------------  

def popup_select_color_table(parent) :
    w = QWPopupSelectColorBar(parent)
    ##w.show()
    resp=w.exec_()
    if   resp == QtGui.QDialog.Accepted : return w.selectedColorTable()
    elif resp == QtGui.QDialog.Rejected : return None
    else : return None

#------------------------------
#------------------------------
#----------- TESTS ------------
#------------------------------
#------------------------------
 
def test_select_color_table(tname) :
    #lst = sorted(os.listdir('/reg/d/psdm/CXI/'))
    #print 'lst:', lst 
    app = QtGui.QApplication(sys.argv)
    ctab_ind = popup_select_color_table(None)
    print 'Selected color table index = %s' % ctab_ind 

#------------------------------
 

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print 50*'_', '\nTest %s' % tname
    if   tname == '0': test_select_color_table(tname)
    #elif tname == '1': test_select_icon(tname)
    else : sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of Test %s' % tname)

#------------------------------
