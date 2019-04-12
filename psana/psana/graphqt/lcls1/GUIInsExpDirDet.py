#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GUIInsExpDirDet...
#------------------------------------------------------------------------

"""GUI sets the calib directory from the instrument & experiment or selected non-standard directory."""

#--------------------------------
__version__ = "$Revision$"
#--------------------------------

import os

from PyQt4 import QtGui, QtCore

from ConfigParametersForApp import cp
import GlobalUtils          as     gu
from FileNameManager        import fnm
from Logger                 import logger
from GUIPopupSelectExp      import select_experiment_v3

#------------------------------

class GUIInsExpDirDet(QtGui.QWidget) :
    """GUI sets the configuration parameters for instrument, experiment, and run number"""

    char_expand  = cp.char_expand
    #char_expand  = u' \u25BC' # down-head triangle
    #char_expand  = '' # down-head triangle

    def __init__(self, parent=None) :

        QtGui.QWidget.__init__(self, parent)

        cp.setIcons()

        self.instr_dir      = cp.instr_dir
        self.instr_name     = cp.instr_name
        self.exp_name       = cp.exp_name
        self.det_name       = cp.det_name
        self.calib_dir      = cp.calib_dir
        self.det_but_title  = cp.det_but_title
        self.but_current    = None

        self.setGeometry(100, 50, 700, 30)
        self.setWindowTitle('Select calibration directory')
 
        self.list_of_exp    = None

        self.titIns  = QtGui.QLabel('Ins:')
        self.titExp  = QtGui.QLabel('Exp:')
        self.titDet  = QtGui.QLabel('Det:')

        self.butIns  = QtGui.QPushButton(self.instr_name.value()    + self.char_expand)
        self.butExp  = QtGui.QPushButton(self.exp_name.value()      + self.char_expand)
        self.butDet  = QtGui.QPushButton(self.det_but_title.value() + self.char_expand)
        self.butBro  = QtGui.QPushButton('Browse' )

        self.ediDir = QtGui.QLineEdit(self.calib_dir.value())
        self.ediDir.setReadOnly(True) 

        self.hbox = QtGui.QHBoxLayout() 
        self.hbox.addWidget(self.titIns)
        self.hbox.addWidget(self.butIns)
        self.hbox.addWidget(self.titExp)
        self.hbox.addWidget(self.butExp)
        self.hbox.addWidget(self.ediDir)
        self.hbox.addWidget(self.butBro)
        self.hbox.addStretch(1)     
        self.hbox.addWidget(self.titDet)
        self.hbox.addWidget(self.butDet)
        self.hbox.addStretch(1)     

        self.setLayout(self.hbox)

        self.connect(self.butIns, QtCore.SIGNAL('clicked()'), self.onButIns)
        self.connect(self.butExp, QtCore.SIGNAL('clicked()'), self.onButExp)
        self.connect(self.butBro, QtCore.SIGNAL('clicked()'), self.onButBro)
        self.connect(self.butDet, QtCore.SIGNAL('clicked()'), self.onButDet)
        #self.connect(self.ediDir, QtCore.SIGNAL('editingFinished()'), self.onEdiDir)
        #self.connect(self.ediExp, QtCore.SIGNAL('editingFinished ()'), self.processEdiExp)
  
        self.showToolTips()
        self.setStyle()

        #self.setStatusMessage()
        #if cp.guistatus is not None : cp.guistatus.updateStatusInfo()

        cp.guiinsexpdirdet = self


    def showToolTips(self):
        # Tips for buttons and fields:
        #self        .setToolTip('This GUI deals with the configuration parameters.')
        self.butIns .setToolTip('Select the instrument name from the pop-up menu.')
        self.butExp .setToolTip('Select the experiment name from the pop-up menu.')
        self.butBro .setToolTip('Select non-default calibration directory.')
        self.butDet .setToolTip('Select the detector for calibration.')
        self.ediDir .setToolTip('Use buttons to change the calib derectory.')


    def setStyle(self):        
        #self.setStyleSheet(cp.styleYellow)
        self.titIns  .setStyleSheet (cp.styleLabel)
        self.titExp  .setStyleSheet (cp.styleLabel)
        self.titDet  .setStyleSheet (cp.styleLabel)

        self.        setFixedHeight(40)
        self.butIns .setFixedWidth(60)
        self.butExp .setFixedWidth(100)
        self.butBro .setFixedWidth(90)
        self.butDet .setFixedWidth(90)
        self.ediDir .setMinimumWidth(310)

        #self.ediDir.setStyleSheet(cp.styleGray)
        self.ediDir.setStyleSheet(cp.styleEditInfo)
        self.ediDir.setEnabled(False)            

        self.butBro .setIcon(cp.icon_browser)
        self.setContentsMargins(-5,-5,-5,-9) # (QtCore.QMargins(-9,-9,-9,-9))        

        self.setStyleButtons()
        

    def setStyleButtons(self):
        if self.instr_name.value() == 'Select' :
            self.butIns.setStyleSheet(cp.styleButtonGood)
            self.butExp.setStyleSheet(cp.styleDefault)
            self.butDet.setStyleSheet(cp.styleDefault)
            self.butExp.setEnabled(False)            
            self.butBro.setEnabled(False)            
            self.butDet.setEnabled(False)
            if cp.guistatus is not None :
                cp.guistatus.setStatusMessage('Select Instrument...')
            return

        self.butIns.setStyleSheet(cp.styleDefault)

        if self.exp_name.value() == 'Select' :
            self.butIns.setStyleSheet(cp.styleDefault)
            self.butExp.setStyleSheet(cp.styleButtonGood)
            self.butDet.setStyleSheet(cp.styleDefault)
            self.butExp.setEnabled(True)            
            self.butBro.setEnabled(False)            
            self.butDet.setEnabled(False)            
            if cp.guistatus is not None :
                cp.guistatus.setStatusMessage('Select Experiment...')
            return

        self.butExp.setStyleSheet(cp.styleDefault)
        self.butBro.setEnabled(True)            
        self.butDet.setStyleSheet(cp.styleDefault)
        self.butDet.setEnabled(True)            

        #if self.det_name.value() == '' :
        if self.det_but_title.value() == 'Select' :
            self.butDet.setStyleSheet(cp.styleButtonGood)
            if cp.guistatus is not None :
                cp.guistatus.setStatusMessage('Select Detector(s)...')
            return

        #self.but.setVisible(False)
        #self.but.setEnabled(True)
        #self.but.setFlat(True)
 

    def setParent(self,parent) :
        self.parent = parent


    def closeEvent(self, event):
        logger.info('closeEvent', __name__)
        #print 'closeEvent'
        #try: # try to delete self object in the cp
        #    del cp.guiselectcalibdir# GUIInsExpDirDet
        #except AttributeError:
        #    pass # silently ignore


    def processClose(self):
        #print 'Close button'
        self.close()

    #def resizeEvent(self, e):
        #print 'resizeEvent' 
        #pass


    #def moveEvent(self, e):
        #print 'moveEvent' 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())


    def onButIns(self):
        #print 'onButIns'
        self.but_current = self.butIns
        item_selected = gu.selectFromListInPopupMenu(cp.list_of_instr)
        if item_selected is None : return            # selection is cancelled
        if item_selected == self.instr_name.value() : return # selected the same item  

        self.setIns(item_selected)
        self.setExp('Select')
        self.setDir('Select')
        self.setDet('Select')
        self.setStyleButtons()


    def onButExp(self):
        #print 'onButExp'
        self.but_current = self.butExp
        dir = self.instr_dir.value() + '/' + self.instr_name.value()
        #print 'dir =', dir
        if self.list_of_exp is None : self.list_of_exp = sorted(os.listdir(dir))
        #item_selected = gu.selectFromListInPopupMenu(self.list_of_exp)
        item_selected = select_experiment_v3(self.butExp, self.list_of_exp)
        if item_selected is None : return          # selection is cancelled
        #if item_selected == self.exp_name.value() : return # selected the same item 

        self.setExp(item_selected)
        self.setDir(fnm.path_to_calib_dir_default())
        self.setDet('Select', clearList=True)
        self.setStyleButtons()

        path_to_xtc_dir = fnm.path_to_xtc_dir()
        if os.path.lexists(path_to_xtc_dir) : return        
        msg = 'XTC data are not seen on this computer for path: %s' % path_to_xtc_dir
        logger.warning(msg, __name__)
        print msg


    def onButBro(self):
        path0 = self.calib_dir.value()
        #print 'path0:', path0
        #dir, calib = self.calib_dir.value().rsplit('/',1)        
        dir, calib = os.path.split(path0)
        #print 'dir, calib =', dir, calib
        path1 = str(QtGui.QFileDialog.getExistingDirectory(self,
                      'Select non-standard calib directory',
                      dir,
                      QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks))
        if path1 == ''    : return # if nothing is selected
        if path1 == path0 : return # is selected the same directory
        if path1.rsplit('/',1)[1] != 'calib' :
            msg = 'Selection of non-"calib" directory "%s" IS FORBIDDEN!' % path1
            logger.warning(msg, __name__)
            return 
        self.setDir(path1)


    def onButDet(self):
        #print 'onButDet'
        self.but_current = self.butDet
        #item_selected = gu.selectFromListInPopupMenu(cp.list_of_dets)
        #if item_selected is None : return            # selection is cancelled
        #if item_selected == self.instr_name.value() : return # selected the same item  
        #self.setDet(item_selected)
        
        list_of_cbox = []
        #for det in cp.list_of_dets :
        for det_name, det_data_type, det_cbx_state in cp.list_of_det_pars :
            #print 'Detector list:', det_name, det_data_type, det_cbx_state.value()    
            list_of_cbox.append([det_name, det_cbx_state.value()])

        #list_of_cbox_out = gu.changeCheckBoxListInPopupMenu(list_of_cbox, win_title='Select Detectors')
        resp = gu.changeCheckBoxListInPopupMenu(list_of_cbox, win_title='Select detector(s)')

        if resp != 1 : return # at cancel

        str_of_detectors = ''

        for [name,state],state_par in zip(list_of_cbox, cp.det_cbx_states_list) :
            #print  '%s checkbox is in state %s' % (name.ljust(10), state)
            state_par.setValue(state)
            if state : str_of_detectors += name + ' '

        self.det_name.setValue(str_of_detectors.rstrip(' '))

        if self.det_name.value() == '' :
            self.butDet.setStyleSheet(cp.styleButtonBad)
            logger.warning('At least one detector needs to be selected !!!', __name__)
            self.setDet('Select')
            return

        self.setDet()
        self.setStyleButtons()


    def setIns(self, txt='Select'):
        self.instr_name.setValue( txt )
        self.butIns.setText( txt + self.char_expand )
        logger.info('Instrument selected: ' + str(txt), __name__)


    def setExp(self, txt='Select'):
        self.exp_name.setValue(txt)
        self.butExp.setText( txt + self.char_expand)
        if txt == 'Select' : self.list_of_exp = None        
        logger.info('Experiment selected: ' + str(txt), __name__)


    def setDir(self, txt='Select'):
        self.calib_dir.setValue(txt) 
        self.ediDir.setText(self.calib_dir.value())
        logger.info('Set calibration directory: ' + str(txt), __name__)


    def setDet(self, txt=None, clearList=True):        
        but_title = 'Select'
        if txt is None :
            but_title = 'Selected:%d' % len(cp.list_of_dets_selected())
            
        self.butDet.setText( but_title + self.char_expand )
        self.det_but_title.setValue(but_title)
        logger.info('Selected detector(s): ' + self.det_name.value(), __name__)

        if cp.guistatus is not None :
            cp.guistatus.updateStatusInfo()

        if cp.guitabs is not None and cp.current_tab.value() == cp.guitabs.list_of_tabs[0] and cp.guidarklist is not None :
            if self.but_current == self.butIns or self.but_current == self.butExp :
                cp.guidarklist.updateList(clearList)
            else :
                cp.guidarklist.updateList()

        if cp.guifilemanagersinglecontrol is not None :
            cp.guifilemanagersinglecontrol.resetFields()

        if cp.guifilemanagergroup is not None :
            cp.guifilemanagergroup.resetFields()

            #if txt=='Select' : cp.guidarklist.setFieldsEnabled(False)
            #else             : cp.guidarklist.setFieldsEnabled(True)

#------------------------------

if __name__ == "__main__" :
    import sys
    app = QtGui.QApplication(sys.argv)
    widget = GUIInsExpDirDet()
    widget.show()
    app.exec_()

#------------------------------
