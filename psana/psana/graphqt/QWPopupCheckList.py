#------------------------------
"""
:py:class:`QWPopupCheckList` - Popup GUI
========================================

Usage::

    # Import
    from psana.graphqt.QWPopupCheckList import QWPopupCheckList

    # Methods - see test

See:
    - :py:class:`QWPopupCheckList`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Adopted for LCLS2 on 2018-02-16 by Mikhail Dubrovin
"""
#------------------------------

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

#------------------------------
__version__ = "v2018-02-16"
#------------------------------

class QWPopupCheckList(QtWidgets.QDialog) :
    """Gets list of item for checkbox GUI in format [['name1',false], ['name2',true], ..., ['nameN',false]], 
    and modify this list in popup dialog gui.
    """

    def __init__(self, parent=None, list_in_out=[], win_title='Set check boxes'):
        QtWidgets.QDialog.__init__(self,parent)
        #self.setGeometry(20, 40, 500, 200)
        self.setWindowTitle(win_title)
 
        #self.setModal(True)
        self.list_in_out = list_in_out

        self.vbox = QtWidgets.QVBoxLayout()

        self.make_gui_checkbox()

        self.but_cancel = QtWidgets.QPushButton('&Cancel') 
        self.but_apply  = QtWidgets.QPushButton('&Apply') 
        
        self.but_cancel.clicked.connect(self.onCancel)
        self.but_apply.clicked.connect(self.onApply)

        self.hbox = QtWidgets.QHBoxLayout()
        self.hbox.addWidget(self.but_cancel)
        self.hbox.addWidget(self.but_apply)
        self.hbox.addStretch(1)

        self.vbox.addLayout(self.hbox)
        self.setLayout(self.vbox)

        self.but_cancel.setFocusPolicy(Qt.NoFocus)

        self.setStyle()
        self.setIcons()
        self.showToolTips()

#-----------------------------  

    def make_gui_checkbox(self) :
        self.dict_of_items = {}
        for k,[name,state] in enumerate(self.list_in_out) :        
            cbx = QtWidgets.QCheckBox(name) 
            if state : cbx.setCheckState(Qt.Checked)
            else     : cbx.setCheckState(Qt.Unchecked)
            #self.connect(cbx, QtCore.SIGNAL('stateChanged(int)'), self.onCBox)
            cbx.stateChanged[int].connect(self.onCBox)
            self.vbox.addWidget(cbx)

            self.dict_of_items[cbx] = [k,name,state] 

#-----------------------------  

    def showToolTips(self):
        self.but_apply.setToolTip('Apply changes to the list')
        self.but_cancel.setToolTip('Use default list')
        

    def setStyle(self):
        #self.setFixedWidth(200)
        self.setMinimumWidth(200)
        styleGray = "background-color: rgb(230, 240, 230); color: rgb(0, 0, 0);" # Gray
        styleDefault = ""

        self.setStyleSheet(styleDefault)
        self.but_cancel.setStyleSheet(styleGray)
        self.but_apply.setStyleSheet(styleGray)


    def setIcons(self):
        from psana.graphqt.QWIcons import icon
        icon.set_icons()
        self.but_cancel.setIcon(icon.icon_button_cancel)
        self.but_apply .setIcon(icon.icon_button_ok)

 
    #def resizeEvent(self, e):
        #logger.debug('resizeEvent', __name__) 
        #pass


    #def moveEvent(self, e):
        #pass


    #def closeEvent(self, event):
        #pass
        #logger.debug('closeEvent', __name__)
        #print('closeEvent')
        #try    : self.widg_pars.close()
        #except : pass

    #def event(self, event):
        #print('Event happens...:', event)

    
    def onCBox(self, tristate):
        for cbx in self.dict_of_items.keys() :
            if cbx.hasFocus() :
                k,name,state = self.dict_of_items[cbx]
                state_new = cbx.isChecked()
                msg = 'onCBox: Checkbox #%d:%s - state is changed to %s, tristate=%s'%(k, name, state_new, tristate)
                #print(msg)
                #logger.debug(msg, __name__)
                self.dict_of_items[cbx] = [k,name,state_new]


    def onCancel(self):
        #logger.debug('onCancel', __name__)
        self.reject()


    def onApply(self):
        #logger.debug('onApply', __name__)  
        self.fill_output_list()
        self.accept()


    def fill_output_list(self):
        """Fills output list"""
        for cbx,[k,name,state] in self.dict_of_items.items() :
            self.list_in_out[k] = [name,state]

#------------------------------

if __name__ == "__main__" :
    import sys
    app = QtWidgets.QApplication(sys.argv)
    list_in = [['CSPAD1',True], ['CSPAD2x21', False], ['pNCCD1', True], ['Opal1', False], \
               ['CSPAD2',True], ['CSPAD2x22', False], ['pNCCD2', True], ['Opal2', False]]
    for name,state in list_in : print( '%s checkbox is in state %s' % (name.ljust(10), state))
    w = QWPopupCheckList (None, list_in)
    #w.show()
    resp=w.exec_()
    print('resp=',resp)
    print('QtWidgets.QDialog.Rejected: ', QtWidgets.QDialog.Rejected)
    print('QtWidgets.QDialog.Accepted: ', QtWidgets.QDialog.Accepted)

    for name,state in list_in : print( '%s checkbox is in state %s' % (name.ljust(10), state))
    #app.exec_()

#------------------------------
