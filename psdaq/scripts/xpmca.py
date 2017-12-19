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

NDsLinks    = 7
NAmcs       = 2
NPartitions = 16

frLMH       = { 'L':0, 'H':1, 'M':2, 'm':3 }
toLMH       = { 0:'L', 1:'H', 2:'M', 3:'m' }

class PvCString:
    def __init__(self, parent, pvbase, name, dName=None):
        layout = QtWidgets.QHBoxLayout()
        label  = QtWidgets.QLabel(name)
        label.setMinimumWidth(100)
        layout.addWidget(label)
        #layout.addStretch()
        self.__display = PvDisplay()
        self.__display.connect_signal()
        layout.addWidget(self.__display)
        parent.addLayout(layout)

        pvname = pvbase+name
        print(pvname)
        self.pv = Pv.Pv(pvname)
        self.pv.monitor_start()
        self.pv.add_monitor_callback(self.update)

    def update(self, err):
        q = self.pv.value
        if err is None:
            s = QString()
            slen = len(q)
            if slen > 64:
                slen = 64
            for i in range(slen):
                if q[i]==0:
                    break
                s.append(QChar(q[i]))
            self.__display.valueSet.emit(s)
        else:
            print(err)

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
        self.setMaximumWidth(70)

class PvCmb(PvEditCmb):

    def __init__(self, pvname, choices):
        super(PvCmb, self).__init__(pvname, choices)
        self.setEnabled(False)

def LblPushButtonX(parent, pvbase, name, count=1, start=0, istart=0):
    return PvInput(PvPushButtonX, parent, pvbase, name, count, start, istart)

def LblEditIntX(parent, pvbase, name, count=1, start=0, istart=0, enable=True):
    return PvInput(PvEditIntX, parent, pvbase, name, count, start, istart, enable)

def FrontPanelAMC(pvbase,iamc):
        dshbox = QtWidgets.QHBoxLayout()
        dsbox = QtWidgets.QGroupBox("Front Panel Links (AMC%d)"%iamc)
        dslo = QtWidgets.QVBoxLayout()
#        LblEditIntX   (lol, pvbase, "LinkTxDelay",    NAmcs * NDsLinks)
#        LblEditIntX   (lol, pvbase, "LinkPartition",  NAmcs * NDsLinks)
#        LblEditIntX   (lol, pvbase, "LinkTrgSrc",     NAmcs * NDsLinks)
        LblPushButtonX(dslo, pvbase, "TxLinkReset",    NDsLinks, start=iamc*NDsLinks)
        LblPushButtonX(dslo, pvbase, "RxLinkReset",    NDsLinks, start=iamc*NDsLinks)
        LblPushButtonX(dslo, pvbase, "RxLinkDump" ,    NDsLinks, start=iamc*NDsLinks)
        LblCheckBox   (dslo, pvbase, "LinkEnable",     NDsLinks, start=iamc*NDsLinks)
        LblCheckBox   (dslo, pvbase, "LinkRxReady",    NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox   (dslo, pvbase, "LinkTxReady",    NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox   (dslo, pvbase, "LinkIsXpm",      NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox   (dslo, pvbase, "LinkLoopback",   NDsLinks)
#        LblCheckBox  (dslo, pvbase, "LinkRxErr",      NAmcs * NDsLinks, enable=False)
        LblEditIntX   (dslo, pvbase, "LinkRxErr",      NDsLinks, start=iamc*NDsLinks, enable=False)
        LblEditIntX   (dslo, pvbase, "LinkRxRcv",      NDsLinks, start=iamc*NDsLinks, enable=False)
        dsbox.setLayout(dslo)
        dshbox.addWidget(dsbox)
        dshbox.addStretch()
        return dshbox

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, title):
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        pvbase = title + ':'
        lol = QtWidgets.QVBoxLayout()
        lor = QtWidgets.QVBoxLayout()

        PvLabel  (lol, pvbase, "PARTITIONS"  )
        PvLabel  (lol, pvbase, "PAddr"       , isInt=True)
        PvCString(lol, pvbase, "FwBuild"     )

        LblPushButtonX(lol, pvbase, "ModuleInit"      )
        LblPushButtonX(lol, pvbase, "DumpPll",        NAmcs)
        LblPushButtonX(lol, pvbase, "DumpTiming",     2)

        LblPushButtonX(lol, pvbase, "ClearLinks"      )

        LblPushButtonX(lol, pvbase, "Inhibit"         )
        LblPushButtonX(lol, pvbase, "TagStream"       )

        lol.addLayout(FrontPanelAMC(pvbase,0))
        lol.addLayout(FrontPanelAMC(pvbase,1))

        bthbox = QtWidgets.QHBoxLayout()
        btbox  = QtWidgets.QGroupBox("Backplane Tx Links")
        btlo   = QtWidgets.QVBoxLayout()
        LblPushButtonX(btlo, pvbase, "TxLinkReset16",    1, 16, 0)
        LblCheckBox   (btlo, pvbase, "LinkTxReady16",    1, 16, 0, enable=False)
        btbox.setLayout(btlo)
        bthbox.addWidget(btbox)
        lol.addLayout(bthbox)

        bphbox = QtWidgets.QHBoxLayout()
        bpbox  = QtWidgets.QGroupBox("Backplane Rx Links")
        bplo   = QtWidgets.QVBoxLayout()
#        LblEditIntX   (lol, pvbase, "LinkTxDelay",    5, 17, 3)
#        LblEditIntX   (lol, pvbase, "LinkPartition",  5, 17, 3)
#        LblEditIntX   (lol, pvbase, "LinkTrgSrc",     5, 17, 3)
#        LblPushButtonX(bplo, pvbase, "TxLinkReset",    5, 17, 3)
        LblPushButtonX(bplo, pvbase, "RxLinkReset",    5, 17, 3)
        LblCheckBox   (bplo, pvbase, "LinkEnable",     5, 17, 3)
        LblCheckBox   (bplo, pvbase, "LinkRxReady",    5, 17, 3, enable=False)
#        LblCheckBox  (bplo, pvbase, "LinkTxReady",    5, 17, 3, enable=False)
#        LblCheckBox  (bplo, pvbase, "LinkIsXpm",      5, 17, 3, enable=False)
#        LblCheckBox  (bplo, pvbase, "LinkLoopback",   5, 17, 3)
        LblEditIntX   (bplo, pvbase, "LinkRxErr",      5, 17, 3, enable=False)
        LblEditIntX   (bplo, pvbase, "LinkRxRcv",      5, 17, 3, enable=False)
        bpbox.setLayout(bplo)
        bphbox.addWidget(bpbox)
        bphbox.addStretch()
        lol.addLayout(bphbox)


        pllhbox = QtWidgets.QHBoxLayout()
        pllbox  = QtWidgets.QGroupBox("PLLs")
        pllvbox = QtWidgets.QVBoxLayout() 
        LblCheckBox  (pllvbox, pvbase, "PLL_LOS",        NAmcs, enable=False)
        LblCheckBox  (pllvbox, pvbase, "PLL_LOL",        NAmcs, enable=False)
        LblEditHML   (pllvbox, pvbase, "PLL_BW_Select",  NAmcs)
        LblEditHML   (pllvbox, pvbase, "PLL_FreqTable",  NAmcs)
        LblEditHML   (pllvbox, pvbase, "PLL_FreqSelect", NAmcs)
        LblEditHML   (pllvbox, pvbase, "PLL_Rate",       NAmcs)
        LblPushButtonX(pllvbox, pvbase, "PLL_PhaseInc",   NAmcs)
        LblPushButtonX(pllvbox, pvbase, "PLL_PhaseDec",   NAmcs)
        LblPushButtonX(pllvbox, pvbase, "PLL_Bypass",     NAmcs)
        LblPushButtonX(pllvbox, pvbase, "PLL_Reset",      NAmcs)
        LblPushButtonX(pllvbox, pvbase, "PLL_Skew",       NAmcs)
        pllbox.setLayout(pllvbox)
        pllhbox.addWidget(pllbox)
        pllhbox.addStretch()
        lol.addLayout(pllhbox)

        if (False):
            LblEditEvt   (lol, pvbase, "L0Select"        )
            LblCheckBox  (lol, pvbase, "SetL0Enabled"    )

            LblCheckBox  (lol, pvbase, "L1TrgClear",     NPartitions)
            LblCheckBox  (lol, pvbase, "L1TrgEnable",    NPartitions)
            LblEditTS    (lol, pvbase, "L1TrgSource",    NPartitions)
            LblEditIntX  (lol, pvbase, "L1TrgWord",      NPartitions)
            LblCheckBox  (lol, pvbase, "L1TrgWrite",     NPartitions)

            LblEditIntX   (lol, pvbase, "AnaTagReset",    NPartitions)
            LblEditIntX   (lol, pvbase, "AnaTag",         NPartitions)
            LblEditIntX   (lol, pvbase, "AnaTagPush",     NPartitions)

            LblEditIntX   (lol, pvbase, "PipelineDepth",  NPartitions)
            LblEditIntX   (lol, pvbase, "MsgHeader",      NPartitions)
            LblCheckBox   (lol, pvbase, "MsgInsert",      NPartitions)
            LblEditIntX   (lol, pvbase, "MsgPayload",     NPartitions)
            LblEditIntX   (lol, pvbase, "InhInterval",    NPartitions)
            LblEditIntX   (lol, pvbase, "InhLimit",       NPartitions)
            LblCheckBox   (lol, pvbase, "InhEnable",      NPartitions)

            #lol.addStretch()

            PvLabel(lor, pvbase, "L0InpRate"  )
            PvLabel(lor, pvbase, "L0AccRate"  )
            PvLabel(lor, pvbase, "L1Rate"     )
            PvLabel(lor, pvbase, "NumL0Inp"   )
            PvLabel(lor, pvbase, "NumL0Acc", None, True)
            PvLabel(lor, pvbase, "NumL1"      )
            PvLabel(lor, pvbase, "DeadFrac"   )
            PvLabel(lor, pvbase, "DeadTime"   )
            PvLabel(lor, pvbase, "DeadFLnk"   )

        PvLabel(lor, pvbase, "RxClks"     )
        PvLabel(lor, pvbase, "TxClks"     )
        PvLabel(lor, pvbase, "RxRsts"     )
        PvLabel(lor, pvbase, "CrcErrs"    )
        PvLabel(lor, pvbase, "RxDecErrs"  )
        PvLabel(lor, pvbase, "RxDspErrs"  )
        PvLabel(lor, pvbase, "BypassRsts" )
        PvLabel(lor, pvbase, "BypassDones")
        PvLabel(lor, pvbase, "RxLinkUp"   )
        PvLabel(lor, pvbase, "FIDs"       )
        PvLabel(lor, pvbase, "SOFs"       )
        PvLabel(lor, pvbase, "EOFs"       )

        lor.addStretch()

        PvLabel(lor, pvbase, "BpClk"      )
        
        #lor.addStretch()

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

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(splitter)

        self.centralWidget.setLayout(layout)
        self.centralWidget.resize(940,840)

        MainWindow.resize(940,840)
        MainWindow.setWindowTitle(title)
        MainWindow.setCentralWidget(self.centralWidget)

if __name__ == '__main__':
    print(QtCore.PYQT_VERSION_STR)

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("pv", help="pv to monitor")
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.pv)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())

