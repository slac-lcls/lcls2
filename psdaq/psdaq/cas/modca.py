import sys
import socket
import argparse
from PyQt5 import QtCore, QtGui, QtWidgets
from psdaq.cas.pvedit import *

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

linkType = []
for x in range(0xfb):
    linkType.append('0x%x'%x)
linkType.append('TDetSim')
linkType.append('HSD')
linkType.append('DRP')
linkType.append('DTI')
linkType.append('XPM')

class PvCString(QtWidgets.QWidget):
    def __init__(self, parent, pvbase, name, dName=None):
        super(PvCString,self).__init__()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        label  = QtWidgets.QLabel(name)
        label.setMinimumWidth(100)
        layout.addWidget(label)
        #layout.addStretch()
        self.__display = PvDisplay()
        self.__display.setWordWrap(True)
        self.__display.connect_signal()
        layout.addWidget(self.__display)
        self.setLayout(layout)
        parent.addWidget(self)

        pvname = pvbase+name
        initPvMon(self,pvname)
#        print(pvname)
#        self.pv = Pv.Pv(pvname)
#        self.pv.monitor_start()
#        self.pv.add_monitor_callback(self.update)

    def update(self, err):
        q = self.pv.value
        if err is None:
            s = QString()
            slen = len(q)
#            if slen > 64:
#                slen = 64
            for i in range(slen):
                if q[i]==0:
                    break
                s += QChar(q[i])
            self.__display.valueSet.emit(s)
        else:
            print(err)

class PvPushButtonX(QtWidgets.QPushButton):

    valueSet = QtCore.pyqtSignal('QString',name='valueSet')

    def __init__(self, pvname, label):
        super(PvPushButtonX, self).__init__(label)
        self.setMaximumWidth(70)

        self.clicked.connect(self.buttonClicked)

        self.pv = Pv(pvname)

    def buttonClicked(self):
        self.pv.put(1)
        self.pv.put(0)

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

class PvLinkId:

    def __init__(self,pvname):
        super(PvLinkId, self).__init__()

        self.linkType = QtWidgets.QLabel('-')
        self.linkType.setMaximumWidth(70)

        self.linkSrc  = QtWidgets.QLabel('-')
        self.linkSrc.setMaximumWidth(70)

        initPvMon(self,pvname)

    def update(self, err):
        value = self.pv.value
        print ('LinkId 0x%x'%value)
        itype = (int(value)>>24)&0xff
        self.linkType.setText(linkType[itype])
        if (itype == 0xfb or itype == 0xfc) and (value&0xffff)!=0:
            ip_addr = '172.21'+'.%u'%((int(value)>>8)&0xff)+'.%u'%((int(value)>>0)&0xff)
            host = socket.gethostbyaddr(ip_addr)[0].split('.')[0].split('-')[-1]
            if itype == 0xfc and value&0xff0000!=0:
                host = host+'.%x'%((value>>16)&0xff)
            self.linkSrc.setText(host)
        else:
            if itype > 0xfc:
                ip_addr = '10.%u'%((int(value)>>16)&0xff)+'.%u'%((int(value)>>8)&0xff)+'.%u'%((int(value)>>0)&0xff)
                self.linkSrc.setText(ip_addr)
            else:
                self.linkSrc.setText('0x%x'%(value&0xffffff))
                

class PvLinkIdV(QtWidgets.QWidget):

    def __init__(self,pvname,idx):
        super(PvLinkIdV, self).__init__()
        self.pvlink = PvLinkId(pvname)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.pvlink.linkType)
        layout.addWidget(self.pvlink.linkSrc)
        self.setLayout(layout)

class PvLinkIdG:

    def __init__(self,pvname,layout,row,col):
        super(PvLinkIdG, self).__init__()
        self.pvlink = PvLinkId(pvname)
        layout.addWidget(self.pvlink.linkType,row,col)
        layout.addWidget(self.pvlink.linkSrc ,row,col+1)

def FrontPanelAMC(pvbase,iamc):
        dsbox = QtWidgets.QWidget()
        dslo = QtWidgets.QVBoxLayout()
#        LblEditIntX   (lol, pvbase, "LinkTxDelay",    NAmcs * NDsLinks)
#        LblEditIntX   (lol, pvbase, "LinkPartition",  NAmcs * NDsLinks)
#        LblEditIntX   (lol, pvbase, "LinkTrgSrc",     NAmcs * NDsLinks)
        PvInput(PvLinkIdV, dslo, pvbase, "RemoteLinkId", NDsLinks, start=iamc*NDsLinks)
        LblPushButtonX(dslo, pvbase, "TxLinkReset",    NDsLinks, start=iamc*NDsLinks)
        LblPushButtonX(dslo, pvbase, "RxLinkReset",    NDsLinks, start=iamc*NDsLinks)
        LblPushButtonX(dslo, pvbase, "RxLinkDump" ,    NDsLinks, start=iamc*NDsLinks)
        LblCheckBox   (dslo, pvbase, "LinkEnable",     NDsLinks, start=iamc*NDsLinks)
        LblCheckBox   (dslo, pvbase, "LinkRxResetDone", NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox   (dslo, pvbase, "LinkRxReady",    NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox   (dslo, pvbase, "LinkTxResetDone", NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox   (dslo, pvbase, "LinkTxReady",    NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox   (dslo, pvbase, "LinkIsXpm",      NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox   (dslo, pvbase, "LinkLoopback",   NDsLinks)
#        LblCheckBox  (dslo, pvbase, "LinkRxErr",      NAmcs * NDsLinks, enable=False)
        LblEditIntX   (dslo, pvbase, "LinkRxErr",      NDsLinks, start=iamc*NDsLinks, enable=False)
        LblEditIntX   (dslo, pvbase, "LinkRxRcv",      NDsLinks, start=iamc*NDsLinks, enable=False)
        dslo.addStretch()
        dsbox.setLayout(dslo)
        return dsbox

def DeadTime(pvbase,parent):

    deadbox = QtWidgets.QWidget()
    deadgrid = QtWidgets.QGridLayout()

    textWidgets = []
    for j in range(7):
        ptextWidgets = []
        for i in range(32):
            ptextWidgets.append( PvDblArrayW() )
        textWidgets.append(ptextWidgets)

    parent.dtPvId = []
    deadgrid.addWidget( QtWidgets.QLabel('Partition'), 0, 0, 1, 2 )
    for j in range(7):
        deadgrid.addWidget( QtWidgets.QLabel('%d'%j ), 0, j+2 )
    for i in range(14):
        parent.dtPvId.append( PvLinkIdG(pvbase+'RemoteLinkId'+'%d'%i,
                                        deadgrid, i+1, 0) )
        for j in range(7):
            deadgrid.addWidget( textWidgets[j][i], i+1, j+2 )
    for i in range(16,21):
        k = i-1
        deadgrid.addWidget( QtWidgets.QLabel('BP-slot%d'%(i-13)), k, 0, 1, 2 )
        for j in range(7):
            deadgrid.addWidget( textWidgets[j][i], k, j+2 )
    for i in range(28,32):
        k = i-7
        deadgrid.addWidget( QtWidgets.QLabel('INH%d'%(i-28)), k, 0, 1, 2 )
        for j in range(7):
            deadgrid.addWidget( textWidgets[j][i], k, j+2 )

    parent.deadflnk = []
    for j in range(7):
        ppvbase = pvbase+'PART:%d:'%j
        print(ppvbase)
        parent.deadflnk.append( PvDblArray( ppvbase+'DeadFLnk', textWidgets[j] ) )

    deadbox.setLayout(deadgrid)
    return deadbox

def addTiming(tw, pvbase, title):
    lor = QtWidgets.QVBoxLayout()
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
    w = QtWidgets.QWidget()
    w.setLayout(lor)
    tw.addTab(w,title)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, title):
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        pvbase = title + ':'
        lol = QtWidgets.QVBoxLayout()

        tw  = QtWidgets.QTabWidget()
        if True:
            tb  = QtWidgets.QWidget()
            hl  = QtWidgets.QVBoxLayout()
            #        PvLabel  (hl, pvbase, "PARTITIONS"  )
            PvLabel  (hl, pvbase, "PAddr"       , isInt=True)
            PvCString(hl, pvbase, "FwBuild"     )

            LblPushButtonX(hl, pvbase, "ModuleInit"      )
            LblPushButtonX(hl, pvbase, "DumpPll",        NAmcs)
            LblPushButtonX(hl, pvbase, "DumpTiming",     2)
            LblPushButtonX(hl, pvbase, "DumpSeq"         )
#            LblEditIntX   (hl, pvbase, "SetVerbose"      )
            LblPushButtonX(hl, pvbase, "ClearLinks"      )
            LblPushButtonX(hl, pvbase, "Inhibit"         )
            LblPushButtonX(hl, pvbase, "TagStream"       )
            PvLabel(hl, pvbase, "RecClk"     )
            PvLabel(hl, pvbase, "FbClk"      )
            PvLabel(hl, pvbase, "BpClk"      )
            hl.addStretch()
            tb.setLayout(hl)
            tw.addTab(tb,"Global")

            addTiming(tw, pvbase+'Us:',"UsTiming")
            addTiming(tw, pvbase+'Cu:',"CuTiming")

        tw.addTab(FrontPanelAMC(pvbase,0),"AMC0")
        tw.addTab(FrontPanelAMC(pvbase,1),"AMC1")

        bpbox  = QtWidgets.QWidget()
        bplo   = QtWidgets.QVBoxLayout()
        LblPushButtonX(bplo, pvbase, "TxLinkReset16",    1, 16, 0)
        LblCheckBox   (bplo, pvbase, "LinkTxReady16",    1, 16, 0, enable=False)

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
        bplo.addStretch()
        bpbox.setLayout(bplo)
        tw.addTab(bpbox,"Bp")

        pllbox  = QtWidgets.QWidget()
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
        pllvbox.addStretch()
        pllbox.setLayout(pllvbox)
        tw.addTab(pllbox,"PLLs")

        tw.addTab(DeadTime(pvbase,self),"DeadTime")

        lol.addWidget(tw)

        ltable = QtWidgets.QWidget()
        ltable.setLayout(lol)

        lscroll = QtWidgets.QScrollArea()
        lscroll.setWidget(ltable)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(lscroll)

        self.centralWidget.setLayout(layout)
        self.centralWidget.resize(720,600)

        MainWindow.resize(720,600)
        MainWindow.setWindowTitle(title)
        MainWindow.setCentralWidget(self.centralWidget)

def main():
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

if __name__ == '__main__':
    main()
