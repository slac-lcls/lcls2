import sys
import argparse
from psp import Pv
from PyQt5 import QtCore, QtGui, QtWidgets
from pvedit import *

try:
    QString = unicode
except NameError:
    # Python 3
    QString = str

try:
    QChar = unichr
except NameError:
    # Python 3
    QChar = chr

class PvPushButtonX(QtWidgets.QPushButton):

    valueSet = QtCore.pyqtSignal('QString',name='valueSet')

    def __init__(self, pvname, label):
        super(PvPushButtonX, self).__init__(label)
        self.setMaximumWidth(25) # Revisit

        self.clicked.connect(self.buttonClicked)

        self.pv = Pv.Pv(pvname)

    def buttonClicked(self):
        self.pv.put(1)          # Value is immaterial

class PvEditIntX(PvEditInt):

    def __init__(self, pv, label):
        super(PvEditIntX, self).__init__(pv, label)
#       self.setMaximumWidth(70)

class PvCmb(PvEditCmb):

    def __init__(self, pvname, choices):
        super(PvCmb, self).__init__(pvname, choices)
        self.setEnabled(False)

def LblPushButtonX(parent, pvbase, name, count=1, start=0, istart=0):
    return PvInput(PvPushButtonX, parent, pvbase, name, count, start, istart)

def LblEditIntX(parent, pvbase, name, count=1, start=0, istart=0, enable=True):
    return PvInput(PvEditIntX, parent, pvbase, name, count, start, istart, enable)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, title):
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        pvbase = title + ':'
        lol = QtWidgets.QVBoxLayout()
        lor = QtWidgets.QVBoxLayout()

        # left side

        LblPushButtonX(lol, pvbase, "CountClear"  )

        LblCheckBox   (lol, pvbase, "UsLinkEn",         7 )

        LblEditInt    (lol, pvbase, "UsLinkPartition",  7 )
        LblEditInt    (lol, pvbase, "UsLinkFwdMask",    7 )
        LblEditInt    (lol, pvbase, "UsLinkTrigDelay",  7 )

        # right side

        PvLabel(lor, pvbase, "UsLinkUp"  )
        PvLabel(lor, pvbase, "BpLinkUp"  )
        PvLabel(lor, pvbase, "DsLinkUp"  )

        PvLabel(lor, pvbase, "UsRxErrs", "dUsRxErrs"   )
        PvLabel(lor, pvbase, "UsRxFull", "dUsRxFull"   )
        PvLabel(lor, pvbase, "UsIbRecv", "dUsIbRecv"   )
        PvLabel(lor, pvbase, "UsIbEvt",  "dUsIbEvt"    )
        PvLabel(lor, pvbase, "DsRxErrs", "dDsRxErrs"   )
        PvLabel(lor, pvbase, "DsRxFull", "dDsRxFull"   )
        PvLabel(lor, pvbase, "DsObSent", "dDsObSent"   )

        PvLabel(lor, pvbase, "QpllLock"    )
        PvLabel(lor, pvbase, "MonClkRate"  )
        PvLabel(lor, pvbase, "MonClkSlow"  )
        PvLabel(lor, pvbase, "MonClkFast"  )
        PvLabel(lor, pvbase, "MonClkLock"  )

        PvLabel(lor, pvbase, "UsLinkObL0",  "dUsLinkObL0"    )
        PvLabel(lor, pvbase, "UsLinkObL1A", "dUsLinkObL1A"   )
        PvLabel(lor, pvbase, "UsLinkObL1R", "dUsLinkObL1R"   )
        PvLabel(lor, pvbase, "RxFrErrs", "dRxFrErrs"         )
        PvLabel(lor, pvbase, "RxFrames", "dRxFrames"         )
        PvLabel(lor, pvbase, "RxOpCodes", "dRxOpCodes"       )
        PvLabel(lor, pvbase, "TxFrErrs", "dTxFrErrs"         )
        PvLabel(lor, pvbase, "TxFrames", "dTxFrames"         )
        PvLabel(lor, pvbase, "TxOpCodes", "dTxOpCodes"       )

        ltable = QtWidgets.QWidget()
        ltable.setLayout(lol)
        rtable = QtWidgets.QWidget()
        rtable.setLayout(lor)

        lscroll = QtWidgets.QScrollArea()
        lscroll.setWidget(ltable)
        rscroll = QtWidgets.QScrollArea()
        rscroll.setWidget(rtable)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(lscroll)
        splitter.addWidget(rscroll)

        # splitter: left side stretch = 2x right side stretch
        splitter.setStretchFactor(0,2)
        splitter.setStretchFactor(1,1)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(splitter)

        self.centralWidget.setLayout(layout)

        MainWindow.resize(1500,550)
        MainWindow.setWindowTitle(title)
        MainWindow.setCentralWidget(self.centralWidget)

if __name__ == '__main__':
    print(QtCore.PYQT_VERSION_STR)

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("base", help="pv base to monitor", default="DAQ:LAB2:DTI")
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.base)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())
