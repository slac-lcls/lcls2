
"""Class :py:class:`CMWConfigFile` is a QWidget for configuration parameters
==============================================================================

Usage ::
    # Test: python lcls2/psana/psana/graphqt/CMWConfigFile.py

    # Import
    from psana.graphqt.CMConfigParameters import

    # See test at the EOF

See:
  - :class:`CMWMain`
  - :class:`CMWConfig`
  - `on github <https://github.com/slac-lcls/lcls2>`_.

Created on 2017-04-05 by Mikhail Dubrovin
"""

import os

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QLineEdit, QCheckBox, QGridLayout, QVBoxLayout, QFileDialog
from psana.graphqt.CMConfigParameters import cp
from psana.graphqt.Styles import style


class CMWConfigFile(QWidget):
    """QWidget for configuration file parameters management"""

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self._name = 'CMWConfigFile'

        #self.parent = cp.guimain

        self.titFile     = QLabel('File with configuration parameters:')
        self.titPars     = QLabel('Operations on file:')
        self.butFile     = QPushButton('File:')
        self.butRead     = QPushButton('Read')
        self.butWrite    = QPushButton('Save')
        self.butDefault  = QPushButton('Reset default')
        self.butPrint    = QPushButton('Print current')
        self.ediFile     = QLineEdit(cp.fname_cp)
        self.cbxSave     = QCheckBox('Save at exit')
        self.cbxSave.setChecked(cp.save_cp_at_exit.value())

        grid = QGridLayout()
        grid.addWidget(self.titFile,       0, 0, 1, 5)
        grid.addWidget(self.butFile,       1, 0)
        grid.addWidget(self.ediFile,       1, 1, 1, 4)
        grid.addWidget(self.titPars,       2, 0, 1, 3)
        grid.addWidget(self.cbxSave,       2, 4)
        grid.addWidget(self.butRead,       3, 1)
        grid.addWidget(self.butWrite,      3, 2)
        grid.addWidget(self.butDefault,    3, 3)
        grid.addWidget(self.butPrint,      3, 4)
        #self.setLayout(grid)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(grid)
        self.vbox.addStretch(1)
        self.setLayout(self.vbox)

        self.ediFile.editingFinished .connect(self.onEditFile)
        self.butRead.clicked.connect(self.onRead)
        self.butWrite.clicked.connect(self.onSave)
        self.butPrint.clicked.connect(self.onPrint)
        self.butDefault.clicked.connect(self.onDefault)
        self.butFile.clicked.connect(self.onFile)
        self.cbxSave.stateChanged[int].connect(self.onCbxSave)

        self.set_tool_tips()
        self.set_style()


    def set_tool_tips(self):
        # Tips for buttons and fields:
        #self           .setToolTip('This GUI deals with the configuration parameters.')
        self.ediFile   .setToolTip('Type the file path name here,\nor better use "Browse" button.')
        self.butFile   .setToolTip('Select the file path name\nto read/write the configuration parameters.')
        self.butRead   .setToolTip('Read the configuration parameters from file.')
        self.butWrite  .setToolTip('Save (write) the configuration parameters in file.')
        self.butDefault.setToolTip('Reset the configuration parameters\nto their default values.')
        self.butPrint  .setToolTip('Print current values of the configuration parameters.')


    def set_style(self):
        #self.setMinimumSize(500,150)
        self.setMaximumSize(600,120)
        #width = 80
        #self.butFile .setFixedWidth(width)
        #self.edi_kin_win_size   .setAlignment(QtCore.Qt.AlignRight)

        self           .setStyleSheet(style.styleBkgd)
        self.titFile   .setStyleSheet(style.styleLabel)
        self.titPars   .setStyleSheet(style.styleLabel)
        self.ediFile   .setStyleSheet(style.styleEdit)
        self.ediFile   .setReadOnly(True)

        self.butFile   .setStyleSheet(style.styleButton)
        self.butRead   .setStyleSheet(style.styleButton)
        self.butWrite  .setStyleSheet(style.styleButton)
        self.butDefault.setStyleSheet(style.styleButton)
        self.butPrint  .setStyleSheet(style.styleButton)
        self.cbxSave   .setStyleSheet(style.styleLabel)
        #self.butClose  .setStyleSheet(style.styleButtonClose)

        self.butFile   .setFixedWidth(50)


    def setParent(self,parent):
        self.parent = parent


    def closeEvent(self, event):
        logger.debug('closeEvent')
        #try   : del cp.guiconfigparameters
        #except: pass


    def onClose(self):
        logger.debug('onClose')
        self.close()


    def onRead(self):
        fname = self.getFileNameFromEditField()
        logger.info('Load configuration parameters from file %s' % fname)
        cp.readParametersFromFile(fname)
        self.ediFile.setText(cp.fname_cp)
        #self.parent.ediFile.setText(cp.fname_cp)
        #self.refreshGUIWhatToDisplay()


    def onWrite(self):
        fname = self.getFileNameFromEditField()
        logger.info('Save configuration parameters in file: %s' % fname)
        cp.saveParametersInFile(fname)


    def onSave(self):
        fname = cp.fname_cp
        logger.info('Save configuration parameters in file: %s' % fname)
        cp.saveParametersInFile(fname)


    def onDefault(self):
        logger.info('Set default values of configuration parameters')
        cp.setDefaultValues()
        self.ediFile.setText(cp.fname_cp)
        #self.refreshGUIWhatToDisplay()


    def onPrint(self):
        logger.debug('onPrint')
        cp.printParameters()


    def onFile(self):
        logger.debug('onFile')
        self.path = self.getFileNameFromEditField()
        self.dname,self.fname = os.path.split(self.path)
        logger.debug('dname "%s"    fname "%s"' % (self.dname, self.fname))
        self.path = QFileDialog.getOpenFileName(self,'Open file',self.dname)[0]
        self.dname, self.fname = os.path.split(self.path)

        if self.dname == '' or self.fname == '':
            logger.info('Input directiry name or file name is empty... use default values')
        else:
            self.ediFile.setText(self.path)
            cp.fname_cp = self.path
            logger.info('Set configuration file name: %s' % cp.fname_cp)


    def onEditFile(self):
        logger.debug('onEditFile')
        self.path = self.getFileNameFromEditField()
        #cp.fname_cp.setValue(self.path)
        cp.fname_cp = self.path
        dname,fname = os.path.split(self.path)
        logger.info('Set dname: %s' % dname)
        logger.info('Set fname: %s' % fname)


    def getFileNameFromEditField(self):
        return str(self.ediFile.displayText())


    def onCbxSave(self):
        #if self.cbx.hasFocus():
        par = cp.save_cp_at_exit
        cbx = self.cbxSave
        tit = cbx.text()

        par.setValue(cbx.isChecked())
        msg = 'Check box "%s" is set to %s' % (tit, str(par.value()))
        logger.info(msg)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CMWConfigFile()
    w.setGeometry(370, 350, 500,150)
    w.setWindowTitle('Configuration File')
    w.show()
    app.exec_()
    del w
    del app

# EOF
