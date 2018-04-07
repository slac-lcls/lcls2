#------------------------------
"""Class :py:class:`CMWConfigPars` is a QWidget for configuration parameters
==============================================================================

Usage ::
    # Test: python lcls2/psana/psana/graphqt/CMWConfigPars.py

    # Import
    from psana.graphqt.CMConfigParameters import

    # See test at the EOF

See:
  - :class:`CMWMain`
  - :class:`CMWConfig`
  - `on github <https://github.com/slac-lcls/lcls2>`_.

Created on 2017-04-05 by Mikhail Dubrovin
"""
#------------------------------

#import os

#from PyQt5.QtGui import QIntValidator, QDoubleValidator# QRegExpValidator
from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout, QComboBox
# QLineEdit,  QPushButton, QComboBox, QCheckBox, QFileDialog
# QTextEdit, QComboBox, QHBoxLayout, QVBoxLayout

from psana.graphqt.CMConfigParameters import cp
from psana.pyalgos.generic.Logger import logger
from psana.graphqt.Styles import style

#------------------------------

class CMWConfigPars(QWidget) :
    """QWidget for managements of configuration parameters"""

    def __init__(self, parent=None) :
        QWidget.__init__(self, parent)
        self._name = 'CMWConfigPars'

        self.lab_host = QLabel('Host:')
        self.lab_port = QLabel('Port:')

        self.cmb_host = QComboBox(self)        
        self.cmb_host.addItems(cp.list_of_hosts)
        self.cmb_host.setCurrentIndex(cp.list_of_hosts.index(cp.cdb_host.value()))

        self.cmb_port = QComboBox(self)        
        self.cmb_port.addItems(cp.list_of_str_ports)
        self.cmb_port.setCurrentIndex(cp.list_of_str_ports.index(str(cp.cdb_port.value())))

#        #self.tit_dir_work = QtGui.QLabel('Parameters:')
#
#        self.edi_dir_work = QLineEdit(cp.dir_work.value())        
#        self.but_dir_work = QPushButton('Dir work:')
#        self.edi_dir_work.setReadOnly(True)  
#
#        self.edi_dir_results = QLineEdit(cp.dir_results.value())        
#        self.but_dir_results = PushButton('Dir results:')
#        self.edi_dir_results.setReadOnly( True )  
#
#        self.lab_fname_prefix = QLabel('File prefix:')
#        self.edi_fname_prefix = QLineEdit(cp.fname_prefix.value())        
#
#        self.lab_bat_queue  = QLabel('Queue:') 
#        self.box_bat_queue  = QComboBox(self) 
#        self.box_bat_queue.addItems(cp.list_of_queues)
#        self.box_bat_queue.setCurrentIndex(cp.list_of_queues.index(cp.bat_queue.value()))
#
#        self.lab_dark_start = QLabel('Event start:') 

#        self.but_show_vers  = QPushButton('Soft Vers')
#        self.edi_dark_start = QLineEdit(str(cp.bat_dark_start.value()))#

#        self.edi_dark_start .setValidator(QIntValidator(0,9999999,self))
#        self.edi_rms_thr_min.setValidator(QDoubleValidator(0,65000,3,self))
#        #self.edi_events.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp("[0-9]\\d{0,3}|end$"),self))
#
#        self.cbx_deploy_hotpix = QCheckBox('Deploy pixel_status')
#        self.cbx_deploy_hotpix.setChecked(cp.dark_deploy_hotpix.value())

        self.grid = QGridLayout()
        self.grid.addWidget(self.lab_host, 0, 0)
        self.grid.addWidget(self.cmb_host, 0, 1, 1, 1)

        self.grid.addWidget(self.lab_port, 2, 0)
        self.grid.addWidget(self.cmb_port, 2, 1, 1, 1)

#        self.grid_row = 0
#        #self.grid.addWidget(self.tit_dir_work,      self.grid_row,   0, 1, 9)
#        self.grid.addWidget(self.but_dir_work,      self.grid_row+1, 0)
#        self.grid.addWidget(self.edi_dir_work,      self.grid_row+1, 1, 1, 8)
#        self.grid.addWidget(self.but_dir_results,   self.grid_row+2, 0)
#        self.grid.addWidget(self.edi_dir_results,   self.grid_row+2, 1, 1, 8)

        self.setLayout(self.grid)
#
#        self.vbox = QVBoxLayout() 
#        self.vbox.addLayout(self.grid)
#        self.vbox.addStretch(1)
#        self.setLayout(self.vbox)
#
#        self.connect(self.but_dir_work,     QtCore.SIGNAL('clicked()'),          self.onButDirWork)
#        self.connect(self.box_bat_queue,    QtCore.SIGNAL('currentIndexChanged(int)'), self.onBoxBatQueue)
#        self.connect(self.edi_fname_prefix, QtCore.SIGNAL('editingFinished ()'), self.onEditPrefix)
#        self.connect(self.cbx_deploy_hotpix,QtCore.SIGNAL('stateChanged(int)'),  self.on_cbx_deploy_hotpix) 

        #self.cbx_host.stateChanged[int].connect(self.on_cbx_host_changed)
        
        self.cmb_host.currentIndexChanged[int].connect(self.on_cmb_host_changed)
        self.cmb_port.currentIndexChanged[int].connect(self.on_cmb_port_changed)

        self.set_tool_tips()
        self.set_style()
#
    def set_tool_tips(self):
        self.cmb_host.setToolTip('Select DB host')
        self.cmb_port.setToolTip('Select DB port')

    def set_style(self):
        self.         setStyleSheet(style.styleBkgd)
        self.lab_host.setStyleSheet(style.styleLabel)

        self.setMaximumSize(200,120)
#        self.setMinimumSize(500,300)
#
#        self.tit_dir_work     .setStyleSheet(style.styleTitle)
#        self.edi_dir_work     .setStyleSheet(style.styleEditInfo)       
#        self.but_dir_work     .setStyleSheet(style.styleButton) 
#        self.lab_fname_prefix .setStyleSheet(style.styleLabel)
#        self.edi_fname_prefix .setStyleSheet(style.styleEdit)
#        self.lab_pix_status   .setStyleSheet(style.styleTitleBold)
#
#        self.edi_dir_work    .setAlignment(QtCore.Qt.AlignRight)
#        self.lab_pix_status  .setAlignment(QtCore.Qt.AlignLeft)
#
#        self.edi_dir_work    .setMinimumWidth(300)
#        self.but_dir_work    .setFixedWidth(80)
#
#    def set_parent(self,parent) :
#        self.parent = parent

    def resizeEvent(self, e):
        logger.debug('resizeEvent size: %s' % str(e.size()), self._name) 


    def moveEvent(self, e):
        logger.debug('moveEvent pos: %s' % str(e.pos()), self._name) 
#        #cp.posGUIMain = (self.pos().x(),self.pos().y())

    def closeEvent(self, event):
        logger.debug('closeEvent', self._name)
        #try    : del cp.guiworkresdirs # CMWConfigPars
        #except : pass # silently ignore
#
#
#   def onClose(self):
#       logger.debug('onClose', self._name)
#       self.close()
#
#    def onButShowVers(self):
#        #list_of_pkgs = ['CalibManager', 'ImgAlgos'] #, 'CSPadPixCoords', 'PSCalib', 'pdscalibdata']
#        #msg = 'Package versions:\n'
#        #for pkg in list_of_pkgs :
#        #    msg += '%s  %s\n' % (gu.get_pkg_version(pkg).ljust(10), pkg.ljust(32))
#
#        #msg = cp.package_versions.text_version_for_all_packages()
#        msg = cp.package_versions.text_rev_and_tag_for_all_packages()
#        logger.info(msg, self._name )
#
#
#    def onButLsfStatus(self):
#        queue = cp.bat_queue.value()
#        farm = cp.dict_of_queue_farm[queue]
#        msg, status = gu.msg_and_status_of_lsf(farm)
#        msgi = '\nLSF status for queue %s on farm %s: \n%s\nLSF status for %s is %s' % \
#               (queue, farm, msg, queue, {False:'bad',True:'good'}[status])
#        logger.info(msgi, self._name)
#
#        cmd, msg = gu.text_status_of_queues(cp.list_of_queues)
#        msgq = '\nStatus of queues for command: %s \n%s' % (cmd, msg)       
#        logger.info(msgq, self._name)
#
#    def onButDirWork(self):
#        self.selectDirectory(cp.dir_work, self.edi_dir_work, 'work')
#
#    def onButDirResults(self):
#        self.selectDirectory(cp.dir_results, self.edi_dir_results, 'results')
#
#    def selectDirectory(self, par, edi, label=''):        
#        logger.debug('Select directory for ' + label, self._name)
#        dir0 = par.value()
#        path, name = os.path.split(dir0)
#        dir = str(QtGui.QFileDialog.getExistingDirectory(None,'Select directory for '+label,path))
#
#        if dir == dir0 or dir == '' :
#            logger.info('Directiry for ' + label + ' has not been changed.', self._name)
#            return
#        edi.setText(dir)        
#        par.setValue(dir)
#        logger.info('Set directory for ' + label + str(par.value()), self._name)
#        gu.create_directory(dir)
#
#    def onBoxBatQueue(self):
#        queue_selected = self.box_bat_queue.currentText()
#        cp.bat_queue.setValue( queue_selected ) 
#        logger.info('onBoxBatQueue - queue_selected: ' + queue_selected, self._name)
#
#    def onEditPrefix(self):
#        logger.debug('onEditPrefix', self._name)
#        cp.fname_prefix.setValue(str(self.edi_fname_prefix.displayText()))
#        logger.info('Set file name common prefix: ' + str( cp.fname_prefix.value()), self._name)
#
#    def onEdiDarkStart(self):
#        str_value = str(self.edi_dark_start.displayText())
#        cp.bat_dark_start.setValue(int(str_value))      
#        logger.info('Set start event for dark run: %s' % str_value, self._name)
#
#    def onEdiDarkEnd(self):
#        str_value = str(self.edi_dark_end.displayText())
#        cp.bat_dark_end.setValue(int(str_value))      
#        logger.info('Set last event for dark run: %s' % str_value, self._name)
#
#    def onEdiDarkScan(self):
#        str_value = str(self.edi_dark_scan.displayText())
#        cp.bat_dark_scan.setValue(int(str_value))      
#        logger.info('Set the number of events to scan: %s' % str_value, self._name)
#
#    def onEdiTimeOut(self):
#        str_value = str(self.edi_timeout.displayText())
#        cp.job_timeout_sec.setValue(int(str_value))      
#        logger.info('Job execution timout, sec : %s' % str_value, self._name)
#
#    def onEdiDarkSele(self):
#        str_value = str(self.edi_dark_sele.displayText())
#        if str_value == '' : str_value = 'None'
#        cp.bat_dark_sele.setValue(str_value)      
#        logger.info('Set the event code for selector: %s' % str_value, self._name)
#
#    def onEdiRmsThrMin(self):
#        str_value = str(self.edi_rms_thr_min.displayText())
#        cp.mask_rms_thr_min.setValue(float(str_value))  
#        logger.info('Set hot pixel RMS MIN threshold: %s' % str_value, self._name)
#
#    def onEdiRmsThr(self):
#        str_value = str(self.edi_rms_thr_max.displayText())
#        cp.mask_rms_thr_max.setValue(float(str_value))  
#        logger.info('Set hot pixel RMS MAX threshold: %s' % str_value, self._name)
#
#    def onEdiMinThr(self):
#        str_value = str(self.edi_min_thr.displayText())
#        cp.mask_min_thr.setValue(float(str_value))  
#        logger.info('Set hot pixel intensity MIN threshold: %s' % str_value, self._name)
#
#
#    def onEdiMaxThr(self):
#        str_value = str(self.edi_max_thr.displayText())
#        cp.mask_max_thr.setValue(float(str_value))  
#        logger.info('Set hot pixel intensity MAX threshold: %s' % str_value, self._name)
#
#    def onEdiRmsNsigLo(self):
#        str_value = str(self.edi_rmsnlo.displayText())
#        cp.mask_rmsnlo.setValue(float(str_value))  
#        logger.info('Set nsigma low limit of rms: %s' % str_value, self._name)
#
#    def onEdiRmsNsigHi(self):
#        str_value = str(self.edi_rmsnhi.displayText())
#        cp.mask_rmsnhi.setValue(float(str_value))  
#        logger.info('Set nsigma high limit of rms: %s' % str_value, self._name)
#
#    def onEdiIntNsigLo(self):
#        str_value = str(self.edi_intnlo.displayText())
#        cp.mask_intnlo.setValue(float(str_value))  
#        logger.info('Set nsigma low limit of intensity: %s' % str_value, self._name)
#
#    def onEdiIntNsigHi(self):
#        str_value = str(self.edi_intnhi.displayText())
#        cp.mask_intnhi.setValue(float(str_value))  
#        logger.info('Set nsigma high limit of intensity: %s' % str_value, self._name)


#    def on_cbx(self, par, cbx):
#        #if cbx.hasFocus() :
#        par.setValue(cbx.isChecked())
#        msg = 'check box %s is set to: %s' % (cbx.text(), str(par.value()))
#        logger.info(msg, self._name)


#    def on_cbx_host_changed(self, i):
#        print('XXX:', str(type(i))
#        self.on_cbx(cp.cdb_host, self.cbx_host)


#    def on_cbx_port_changed(self, i):
#        self.on_cbx(cp.cdb_port, self.cbx_port)

    def on_cmb_host_changed(self):
        selected = self.cmb_host.currentText()
        cp.cdb_host.setValue(selected) 
        logger.info('on_cmb_host_changed - selected: %s' % selected, self._name)

    def on_cmb_port_changed(self):
        selected = self.cmb_port.currentText()
        cp.cdb_port.setValue(int(selected)) 
        logger.info('on_cmb_port_changed - selected: %s' % selected, self._name)

#-----------------------------

if __name__ == "__main__" :
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    logger.setPrintBits(0o177777)
    w = CMWConfigPars()
    #w.setGeometry(200, 400, 500, 200)
    w.setWindowTitle('Config Parameters')
    w.show()
    app.exec_()
    del w
    del app

#-----------------------------
