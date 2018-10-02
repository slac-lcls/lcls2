
#------------------------------
"""
:py:class:`QWPopupSelectColorBar` - Popup GUI for selection color table
=======================================================================

Usage::

    # Import
    from psana.graphqt.QWPopupSelectColorBar import QWPopupSelectColorBar

    # Methods - see test

See:
    - :py:class:`QWPopupSelectColorBar`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2017-01-31 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-16 by Mikhail Dubrovin
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QDialog, QPushButton, QListWidget, QVBoxLayout,\
                            QListWidgetItem, QLabel
from PyQt5.QtCore import Qt, QPoint, QEvent, QSize
from PyQt5.QtGui import QCursor

import psana.graphqt.ColorTable as ct
from psana.graphqt.Styles import style

#------------------------------

class QWPopupSelectColorBar(QDialog) :

    def __init__(self, parent=None):

        QDialog.__init__(self, parent)

        # Confirmation buttons
        self.but_cancel = QPushButton('&Cancel') 
        #self.but_apply  = QPushButton('&Apply') 
        #self.but_create = QPushButton('&Create') 

        #cp.setIcons()
        #self.but_cancel.setIcon(cp.icon_button_cancel)
        #self.but_apply .setIcon(cp.icon_button_ok)
        #self.connect(self.but_cancel, QtCore.SIGNAL('clicked()'), self.onCancel)
        #self.connect(self.but_apply,  QtCore.SIGNAL('clicked()'), self.onApply)

        #self.hbox = QVBoxLayout()
        #self.hbox.addWidget(self.but_cancel)
        #self.hbox.addWidget(self.but_apply)
        ##self.hbox.addStretch(1)

        self.ctab_selected = None
        self.list = QListWidget(parent)

        self.fill_list()

        vbox = QVBoxLayout()
        vbox.addWidget(self.list)
        self.setLayout(vbox)

        self.but_cancel.clicked.connect(self.onCancel)
        self.list.itemClicked.connect(self.onItemClick)

        self.showToolTips()
        self.setStyle()


    def fill_list(self) :
        for i in range(1,9) :
           item = QListWidgetItem('%02d'%i, self.list)
           item.setSizeHint(QSize(200,30))
           item._coltab_index = i
           #item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
           lab = QLabel(parent=None)
           lab.setPixmap(ct.get_pixmap(i, size=(200,30)))
           self.list.setItemWidget(item, lab)

        item = QListWidgetItem('cancel', self.list)
        self.list.setItemWidget(item, self.but_cancel)
        
  
    def onItemClick(self, item):
        self.ctab_selected = item._coltab_index
        self.accept()


#class QWPopupSelectColorBarV0(QDialog) :

    def __init__V0(self, parent=None):

        QDialog.__init__(self, parent)

        self.ctab_selected = None
        #self.list = QListWidget(parent)

        #self.fill_list(lst)
        #self.fill_list_icons(lst_icons)

        # Confirmation buttons
        self.but_cancel = QPushButton('&Cancel') 
        #self.but_apply  = QPushButton('&Apply') 
        #self.but_create = QPushButton('&Create') 

        #cp.setIcons()
        #self.but_cancel.setIcon(cp.icon_button_cancel)
        #self.but_apply .setIcon(cp.icon_button_ok)
        #self.connect(self.but_cancel, QtCore.SIGNAL('clicked()'), self.onCancel)
        #self.connect(self.but_apply,  QtCore.SIGNAL('clicked()'), self.onApply)

        #self.hbox = QVBoxLayout()
        #self.hbox.addWidget(self.but_cancel)
        #self.hbox.addWidget(self.but_apply)
        ##self.hbox.addStretch(1)


        vbox = QVBoxLayout()
        for i in range(1,9) :
           lab = QLabel(parent=None)
           lab.setPixmap(ct.get_pixmap(i, size=(200,30)))
           #lab.setText('%02d'%i) # set text !!!OR!!! pixmam
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
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.layout().setContentsMargins(0,0,0,0)
        #self.setStyleSheet(style.styleBkgd)
        #self.but_create.setStyleSheet(style.styleButton)
        #self.but_apply.setStyleSheet(style.styleButton)
        self.but_cancel.setStyleSheet(style.styleButton)
        self.but_cancel.setFixedSize(200,30)
        self.move(QCursor.pos().__add__(QPoint(-110,-50)))


    def showToolTips(self):
        #self.but_apply.setToolTip('Apply selection')
        #self.but_cancel.setToolTip('Cancel selection')
        self.setToolTip('Select color table')


    def mousePressEvent(self, e):
        logger.debug('mousePressEvent')
        child = self.childAt(e.pos())
        if isinstance(child, QLabel) :
            logger.debug('Selected color table index: %d' % child._coltab_index)
            self.ctab_selected = child._coltab_index
            self.accept()


    def event(self, e):
        #logger.debug('event.type %s' % e.type())
        if e.type() == QEvent.WindowDeactivate :
            self.reject()
        return QDialog.event(self, e)
    

    def closeEvent(self, e) :
        logger.debug('closeEvent')
        self.reject()


    def selectedColorTable(self):
        return self.ctab_selected

 
    def onCancel(self):
        logger.debug('onCancel', __name__)
        self.reject()


#    def onApply(self):
#        #logger.debug('onApply', __name__)  
#        self.accept()


#    def onSelect(self):
#        logger.debug('onSelect')

#------------------------------  

def popup_select_color_table(parent) :
    w = QWPopupSelectColorBar(parent)
    ##w.show()
    resp=w.exec_()
    if   resp == QDialog.Accepted : return w.selectedColorTable()
    elif resp == QDialog.Rejected : return None
    else : return None

#------------------------------
#------------------------------
#----------- TESTS ------------
#------------------------------
#------------------------------
 
def test_select_color_table(tname) :
    from PyQt5.QtWidgets import QApplication
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    #lst = sorted(os.listdir('/reg/d/psdm/CXI/'))
    #logger.debug('lst: %s' % lst)
    app = QApplication(sys.argv)
    ctab_ind = popup_select_color_table(None)
    logger.debug('Selected color table index = %s' % ctab_ind)

#------------------------------
 

if __name__ == "__main__" :
    import sys; global sys

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.debug('%s\nTest %s' % (50*'_', tname))
    if   tname == '0': test_select_color_table(tname)
    #elif tname == '1': test_select_icon(tname)
    else : sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of Test %s' % tname)

#------------------------------
