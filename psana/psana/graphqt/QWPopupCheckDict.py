#------------------------------
"""
:py:class:`QWPopupCheckDict` - Popup GUI
========================================

Usage::

    # Test: python lcls2/psana/psana/graphqt/QWPopupCheckDict.py

    # Import
    from psana.graphqt.QWPopupCheckDict import QWPopupCheckDict

    # Methods - see test

See:
    - :py:class:`QWPopupCheckDict`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Adopted for LCLS2 on 2018-02-16 by Mikhail Dubrovin
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QPushButton, QCheckBox
from PyQt5.QtCore import Qt

#------------------------------

class QWPopupCheckDict(QDialog) :
    """Gets dict of item for checkbox GUI in format {name:(bool)status,}, 
    e.g.: {'name1':False, 'name2':True, ..., 'nameN':False}, 
    and modify this dict in popup dialog gui.
    """
    def __init__(self, parent=None, dict_in_out={}, win_title=None):
        QDialog.__init__(self, parent)
 
        #self.setModal(True)
        if win_title is not None : self.setWindowTitle(win_title)

        self.dict_in_out = dict_in_out

        self.vbox = QVBoxLayout()

        self.make_gui_checkbox()

        self.but_cancel = QPushButton('&Cancel') 
        self.but_apply  = QPushButton('&Apply') 
        
        self.but_cancel.clicked.connect(self.onCancel)
        self.but_apply.clicked.connect(self.onApply)

        self.hbox = QHBoxLayout()
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
        for name,state in self.dict_in_out.items() :        
            cbx = QCheckBox(name) 
            if state : cbx.setCheckState(Qt.Checked)
            else     : cbx.setCheckState(Qt.Unchecked)
            #self.connect(cbx, QtCore.SIGNAL('stateChanged(int)'), self.onCBox)
            cbx.stateChanged[int].connect(self.onCBox)
            self.vbox.addWidget(cbx)

            self.dict_of_items[cbx] = [name,state] 

#-----------------------------  

    def showToolTips(self):
        self.but_apply.setToolTip('Apply changes to the dict')
        self.but_cancel.setToolTip('Use default dict')
        

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
        #logger.debug('resizeEvent') 

    #def moveEvent(self, e):
        #logger.debug('moveEvent') 

    #def closeEvent(self, event):
        #pass
        #logger.debug('closeEvent')
        #try    : self.widg_pars.close()
        #except : pass

    #def event(self, event):
        #logger.debug('Event happens...: %s' % str(event))

    
    def onCBox(self, tristate):
        for cbx in self.dict_of_items.keys() :
            if cbx.hasFocus() :
                name,state = self.dict_of_items[cbx]
                state_new = cbx.isChecked()
                msg = 'onCBox: Checkbox %s - state is changed to %s, tristate=%s'%(name, state_new, tristate)
                logger.debug(msg)
                self.dict_of_items[cbx] = [name,state_new]


    def onCancel(self):
        logger.debug('onCancel')
        self.reject()


    def onApply(self):
        logger.debug('onApply')  
        self.fill_output_dict()
        self.accept()


    def fill_output_dict(self):
        """Fills output dict"""
        for cbx,[name,state] in self.dict_of_items.items() :
            self.dict_in_out[name] = state

#------------------------------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    dict_in = {'CSPAD1':True, 'CSPAD2x21':False, 'pNCCD1':True, 'Opal1':False, \
               'CSPAD2':True, 'CSPAD2x22':False, 'pNCCD2':True, 'Opal2':False}
    for name,state in dict_in.items() : logger.debug('%s checkbox is in state %s' % (name.ljust(10), state))
    w = QWPopupCheckDict(None, dict_in)
    #w.setGeometry(20, 40, 500, 200)
    w.setWindowTitle('Set check boxes')
    #w.show()
    resp=w.exec_()
    logger.debug('resp=%s' % resp)
    logger.debug('QtWidgets.QDialog.Rejected: %d' % QDialog.Rejected)
    logger.debug('QtWidgets.QDialog.Accepted: %d' % QDialog.Accepted)

    for name,state in dict_in.items() : logger.debug('%s checkbox is in state %s' % (name.ljust(10), state))

    del w
    del app

#------------------------------
